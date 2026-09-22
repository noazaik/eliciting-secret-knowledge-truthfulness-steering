import json
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from steering_vectors import extract_activations

# ---------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, REPO_ROOT)

from sampling.utils import load_model_and_tokenizer


# =====================================================================
# Configuration
# =====================================================================

MODEL_NAME = "google/gemma-2-9b-it"

LAYERS = [8, 12, 16, 20, 23, 26]

VECTOR_DIR = os.path.join(
    REPO_ROOT,
    "truthfulness",
    "vectors",
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "truthfulness",
    "evaluation",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 1
TEST_SIZE = 0.2

# Must match vector generation.
READ_TOKEN_INDEX = -3

# Same approach as vector generation.
BATCH_SIZE = 1


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Formatting
# =====================================================================

def format_example(
    tokenizer,
    question: str,
    answer: str,
) -> str:

    messages = [
        {
            "role": "user",
            "content": question,
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# =====================================================================
# Bootstrap
# =====================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 1,
    n_bootstrap: int = 5000,
):
    rng = np.random.default_rng(seed)

    n = len(values)

    samples = rng.choice(
        values,
        size=(n_bootstrap, n),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


# =====================================================================
# Main
# =====================================================================

def main():

    set_seed(SEED)

    print("=" * 80)
    print("HELD-OUT TRUTHFULNESS VECTOR EVALUATION")
    print("=" * 80)

    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Seed: {SEED}")
    print(f"Read token index: {READ_TOKEN_INDEX}")
    print(f"Batch size: {BATCH_SIZE}")

    # -----------------------------------------------------------------
    # Load vectors
    # -----------------------------------------------------------------

    print("\nLoading steering vectors...")

    vectors = {}

    for layer in LAYERS:

        path = os.path.join(
            VECTOR_DIR,
            f"truthfulness_l{layer}.pt",
        )

        vector = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).float()

        if vector.ndim != 1:
            raise ValueError(
                f"Layer {layer} vector has unexpected shape: "
                f"{tuple(vector.shape)}"
            )

        if not torch.isfinite(vector).all():
            raise ValueError(
                f"Layer {layer} vector contains NaN or Inf."
            )

        vectors[layer] = vector

        print(
            f"Layer {layer}: "
            f"shape={tuple(vector.shape)}, "
            f"norm={vector.norm().item():.6f}"
        )

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    print("\nLoading TruthfulQA...")

    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
    )

    split = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )

    test_data = split["test"]

    print(f"Total examples: {len(dataset)}")
    print(f"Held-out examples: {len(test_data)}")

    # -----------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------

    print("\nLoading model...")

    model, tokenizer = load_model_and_tokenizer(
        MODEL_NAME
    )

    model.eval()

    # -----------------------------------------------------------------
    # Construct held-out pairs
    # -----------------------------------------------------------------

    pairs = []

    for row in test_data:

        incorrect_answers = row["incorrect_answers"]

        if not incorrect_answers:
            continue

        truthful_text = format_example(
            tokenizer,
            row["question"],
            row["best_answer"],
        )

        incorrect_text = format_example(
            tokenizer,
            row["question"],
            incorrect_answers[0],
        )

        pairs.append(
            {
                "question": row["question"],
                "truthful_answer": row["best_answer"],
                "incorrect_answer": incorrect_answers[0],
                "truthful_text": truthful_text,
                "incorrect_text": incorrect_text,
            }
        )

    print(
        f"Valid held-out contrastive pairs: {len(pairs)}"
    )

    # -----------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------

    results = {
        layer: {
            "truthful_projection": [],
            "incorrect_projection": [],
            "margin": [],
        }
        for layer in LAYERS
    }

    # -----------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------

    print("\nEvaluating held-out examples...")

    for start in range(
        0,
        len(pairs),
        BATCH_SIZE,
    ):

        batch = pairs[
            start:start + BATCH_SIZE
        ]

        truthful_prompts = [
            x["truthful_text"]
            for x in batch
        ]

        incorrect_prompts = [
            x["incorrect_text"]
            for x in batch
        ]

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Use the SAME extraction function that generated the vectors.
        #
        # This takes care of:
        #   - padding
        #   - negative token indices
        #   - layer hooks
        #   - model invocation
        # -------------------------------------------------------------

        truthful_acts, incorrect_acts = (
            extract_activations(
                model=model,
                tokenizer=tokenizer,
                training_samples=list(
                    zip(
                        truthful_prompts,
                        incorrect_prompts,
                    )
                ),
                layers=LAYERS,
                layer_type="decoder_block",
                move_to_cpu=True,
                read_token_index=READ_TOKEN_INDEX,
                show_progress=False,
                batch_size=BATCH_SIZE,
                tqdm_desc="",
            )
        )

        # -------------------------------------------------------------
        # Calculate projections
        # -------------------------------------------------------------

        for layer in LAYERS:

            truthful_activation = torch.cat(
                truthful_acts[layer],
                dim=0,
            ).float()

            incorrect_activation = torch.cat(
                incorrect_acts[layer],
                dim=0,
            ).float()

            vector = vectors[layer]

            truthful_projection = (
                truthful_activation @ vector
            )

            incorrect_projection = (
                incorrect_activation @ vector
            )

            margin = (
                truthful_projection
                - incorrect_projection
            )

            results[layer][
                "truthful_projection"
            ].extend(
                truthful_projection.tolist()
            )

            results[layer][
                "incorrect_projection"
            ].extend(
                incorrect_projection.tolist()
            )

            results[layer][
                "margin"
            ].extend(
                margin.tolist()
            )

        processed = min(
            start + len(batch),
            len(pairs),
        )

        print(
            f"Processed {processed}/{len(pairs)} pairs",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    summary = {}

    for layer in LAYERS:

        truthful = np.asarray(
            results[layer]["truthful_projection"],
            dtype=np.float64,
        )

        incorrect = np.asarray(
            results[layer]["incorrect_projection"],
            dtype=np.float64,
        )

        margins = np.asarray(
            results[layer]["margin"],
            dtype=np.float64,
        )

        pairwise_accuracy = float(
            np.mean(margins > 0)
        )

        mean_margin = float(
            np.mean(margins)
        )

        median_margin = float(
            np.median(margins)
        )

        std_margin = float(
            np.std(margins, ddof=1)
        )

        ci_low, ci_high = bootstrap_mean_ci(
            margins,
            seed=SEED,
        )

        summary[str(layer)] = {
            "n_examples": len(margins),
            "pairwise_accuracy": pairwise_accuracy,
            "mean_truthful_projection": float(
                np.mean(truthful)
            ),
            "mean_incorrect_projection": float(
                np.mean(incorrect)
            ),
            "mean_margin": mean_margin,
            "median_margin": median_margin,
            "std_margin": std_margin,
            "bootstrap_95ci_mean_margin_low": ci_low,
            "bootstrap_95ci_mean_margin_high": ci_high,
        }

        print(f"\nLayer {layer}")
        print(
            f"  Pairwise accuracy: "
            f"{pairwise_accuracy:.4f}"
        )
        print(
            f"  Mean truthful projection: "
            f"{np.mean(truthful):.6f}"
        )
        print(
            f"  Mean incorrect projection: "
            f"{np.mean(incorrect):.6f}"
        )
        print(
            f"  Mean margin: "
            f"{mean_margin:.6f}"
        )
        print(
            f"  Median margin: "
            f"{median_margin:.6f}"
        )
        print(
            f"  95% CI: "
            f"[{ci_low:.6f}, {ci_high:.6f}]"
        )

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------

    output = {
        "model": MODEL_NAME,
        "dataset": "truthfulqa/truthful_qa:generation",
        "total_dataset_examples": len(dataset),
        "num_heldout_pairs": len(pairs),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "layers": LAYERS,
        "read_token_index": READ_TOKEN_INDEX,
        "incorrect_answer_selection": (
            "first incorrect_answers entry"
        ),
        "batch_size": BATCH_SIZE,
        "summary": summary,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "heldout_vector_summary.json",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(f"\nSaved: {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()