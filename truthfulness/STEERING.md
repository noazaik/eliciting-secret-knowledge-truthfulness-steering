# Truthfulness steering for secret elicitation

## Pieces
- `truthfulness/train_truthfulness_vectors.py` – CAA vectors from TruthfulQA (base gemma-2-9b-it).
- `truthfulness/evaluate_truthfulness_vectors.py` – held-out projection check.
- `sampling/interventions.py` – `SteeringHook`, `compute_reference_norm`, `load_steering_vector`.
- `sampling/batch_inference.py` – `--enable_steering` and `--steer_*` flags.
- `user_gender/scripts/run_inference_steering.sh` – one steered inference run.
- `user_gender/scripts/sweep_steering_val.sh` – validation sweep (both models, layers, coeffs, controls) + auditor.
- `truthfulness/summarize_steering_sweep.py` – sweep table.

## Conventions
- Added vector = `coeff * ref_norm * v_hat`. `ref_norm` = median residual norm at the layer over
  real prompt tokens (BOS/pad excluded), estimated from the run's prompts and stored in the
  output metadata. `coeff` is therefore "fraction of a typical activation", comparable across layers.
- `coeff > 0` → towards truthful (vector = mean(truthful − untruthful)). `coeff < 0` is the anti control.
- `--steer_positions response` (default) steers the last prompt token + every generated token;
  `all` also steers the prompt.
- `coeff 0.0` runs the same code path unsteered – use it as the in-pipeline baseline.
- `random:<seed>` replaces the vector with a random unit direction of the same size (control).

## Protocol
1. `./user_gender/scripts/sweep_steering_val.sh` on `gender_direct_val.txt`.
2. Choose the config by average auditor accuracy over female+male, checking that it beats the
   random control at the same coeff and that the anti (negative) coeff doesn't also help.
3. Run only that config on `gender_direct_test.txt` for both models and report next to the
   paper's rows (I/O, fuzzing, high-T, prefill, user persona).
