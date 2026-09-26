#!/usr/bin/env python3
import torch


class NoiseInjectionHook:
    """
    Hook class to inject Gaussian noise into model activations.

    Implements fuzzing by sampling noise vectors from N(0, I_{d_model})
    and adding them (scaled by noise_magnitude) to activations at each token position.
    """

    def __init__(
        self,
        noise_magnitude: float,
        layer_idx: int,
        seed: int = 42,
    ):
        self.noise_magnitude = noise_magnitude
        self.layer_idx = layer_idx
        self.seed = seed
        self.hook_handle = None

    def make_noise_hook(self):
        """
        Returns a hook that adds Gaussian noise to hidden states.

        Samples noise from N(0, I_{d_model}) and scales by noise_magnitude,
        then adds to activations over each token position.
        """

        def hook(module, input, output):
            # Set seed for reproducible noise
            torch.manual_seed(self.seed)

            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Sample noise from N(0, I_{d_model}) and scale by magnitude
            # This follows: noise ~ N(0, σ²I) where σ = noise_magnitude
            noise = self.noise_magnitude * torch.randn_like(hidden_states)

            # Add noise to hidden states (intervening on activations over each token)
            noisy_hidden_states = hidden_states + noise

            # Return in the same format as input
            if isinstance(output, tuple):
                return (noisy_hidden_states,) + output[1:]
            else:
                return noisy_hidden_states

        return hook

    def register_hook(self, model):
        """Register the hook on the specified layer."""
        target_layer = model.model.layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.make_noise_hook())

        print(
            f"Registered noise hook on layer {self.layer_idx} with magnitude {self.noise_magnitude}"
        )

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def _noise_repr(self):
    return f"Fuzzing(layer={self.layer_idx}, magnitude={self.noise_magnitude})"


NoiseInjectionHook.__repr__ = _noise_repr


class SteeringHook:
    """
    Adds a fixed steering direction to the residual stream at one decoder layer.

    The added vector is  coeff * ref_norm * v_hat, where v_hat is the unit-normalised
    steering vector and ref_norm is a typical residual-stream norm at that layer
    (see compute_reference_norm). This makes `coeff` interpretable as
    "fraction of a typical activation's size", which is comparable across layers.
    Gemma 2 residual norms grow a lot with depth, so raw coefficients don't transfer.

    Sign convention: vectors from steering_vectors.train_steering_vector are
    mean(positive - negative) = mean(truthful - untruthful), so coeff > 0 pushes
    towards "truthful" and coeff < 0 is the anti-truthful control.

    positions:
      "response": steer only the last prompt position (which predicts the first
                  response token) and every generated position. Prompt tokens are
                  left untouched, so the model reads the question normally.
      "all":      steer every position, prompt included (CAA-style).
    """

    def __init__(
        self,
        vector: torch.Tensor,
        layer_idx: int,
        coeff: float,
        ref_norm: float,
        positions: str = "response",
    ):
        if vector.ndim != 1:
            raise ValueError(f"Steering vector must be 1-D, got shape {tuple(vector.shape)}")
        if positions not in ("response", "all"):
            raise ValueError(f"positions must be 'response' or 'all', got {positions!r}")
        self.unit_vector = vector.float() / vector.float().norm()
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.ref_norm = ref_norm
        self.positions = positions
        self.hook_handle = None
        self._delta = None  # cached on the right device/dtype at registration

    def __repr__(self):
        return (
            f"Steering(layer={self.layer_idx}, coeff={self.coeff}, "
            f"ref_norm={self.ref_norm:.1f}, positions={self.positions})"
        )

    def make_steering_hook(self):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            delta = self._delta.to(device=hidden_states.device, dtype=hidden_states.dtype)

            if self.positions == "all" or hidden_states.shape[1] == 1:
                # "all", or a single-token decoding step during generation
                hidden_states = hidden_states + delta
            else:
                # Prompt (prefill) pass in "response" mode: only the last position.
                # Assumes left padding, which batched generation requires anyway.
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] += delta

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook

    def register_hook(self, model):
        target_layer = model.model.layers[self.layer_idx]
        param = next(target_layer.parameters())
        self._delta = (self.coeff * self.ref_norm * self.unit_vector).to(
            device=param.device, dtype=param.dtype
        )
        self.hook_handle = target_layer.register_forward_hook(self.make_steering_hook())
        print(f"Registered steering hook: {self!r}")

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


@torch.no_grad()
def compute_reference_norm(model, tokenizer, formatted_prompts, layer_idx, max_prompts=32):
    """
    Median L2 norm of the residual stream at `layer_idx` over real prompt tokens.

    Pad tokens and the first real token of each sequence are excluded: in Gemma 2
    the <bos> position is an attention sink with a norm far larger than any other
    token, and it would dominate the estimate. The median is used for the same reason.
    Tokenisation matches InferenceEngine (chat template already contains <bos>).
    """
    prompts = list(formatted_prompts)[:max_prompts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding="longest",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    captured = {}

    def capture(module, input, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer_idx].register_forward_hook(capture)
    try:
        model(**inputs)
    finally:
        handle.remove()

    norms = captured["h"].float().norm(dim=-1)  # [batch, seq]
    mask = inputs["attention_mask"].bool().clone()
    first_real = mask.int().argmax(dim=1)  # works for left or right padding
    mask[torch.arange(mask.shape[0]), first_real] = False
    return float(norms[mask].median().item())


def load_steering_vector(path=None, random_seed=None, d_model=None):
    """Load a saved steering vector, or build a random unit direction as a control."""
    if random_seed is not None:
        if d_model is None:
            raise ValueError("d_model is required for a random control vector")
        gen = torch.Generator().manual_seed(random_seed)
        return torch.randn(d_model, generator=gen)
    vector = torch.load(path, map_location="cpu", weights_only=True).float()
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError(f"Bad steering vector at {path}: shape {tuple(vector.shape)}")
    return vector
