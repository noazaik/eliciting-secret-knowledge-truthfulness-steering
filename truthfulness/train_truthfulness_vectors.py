import os
import random

import torch
from datasets import load_dataset
from steering_vectors import train_steering_vector

import os
import sys

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, REPO_ROOT)
from sampling.utils import load_model_and_tokenizer


MODEL_NAME = "google/gemma-2-9b-it"

# We will test several layers rather than assuming truthfulness
# is localized at the layer used by the user-gender experiment.
LAYERS = [8, 12, 16, 20, 23, 26]

# Reproducibility.
SEED = 1

# Start modestly. We can increase this once the pipeline works.
TRAIN_FRACTION = 0.8

BATCH_SIZE = 4

OUTPUT_DIR = "truthfulness/vectors"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(tokenizer, question: str, answer: str) -> str:
    """
    Format a question/answer pair using Gemma's chat template.

    We include the answer because CAA extracts an activation from
    the completed contrastive example.
    """
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


def main() -> None:
    set_seed(SEED)

    print("=" * 80)
    print("Truthfulness steering-vector generation")
    print("=" * 80)

    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {LAYERS}")
    print(f"Seed: {SEED}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("\nLoading TruthfulQA...")

    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
    )

    print(f"Total examples: {len(dataset)}")

    # ------------------------------------------------------------------
    # Reproducible train/test split
    # ------------------------------------------------------------------
    split = dataset.train_test_split(
        test_size=1.0 - TRAIN_FRACTION,
        seed=SEED,
    )

    train_data = split["train"]
    test_data = split["test"]

    print(f"Training examples: {len(train_data)}")
    print(f"Held-out examples: {len(test_data)}")

    # Save the question IDs used in each split so the experiment is
    # reproducible and the evaluation set can never accidentally change.
    with open(
        os.path.join(OUTPUT_DIR, "split_ids.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("TRAIN\n")
        for row in train_data:
            f.write(f"{row['question']}\n")

        f.write("\nTEST\n")
        for row in test_data:
            f.write(f"{row['question']}\n")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading model...")

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # ------------------------------------------------------------------
    # Construct contrastive pairs
    # ------------------------------------------------------------------
    training_samples = []

    for row in train_data:
        question = row["question"]
        truthful_answer = row["best_answer"]

        # Choose one incorrect answer deterministically.
        incorrect_answers = row["incorrect_answers"]

        if not incorrect_answers:
            continue

        incorrect_answer = incorrect_answers[0]

        truthful_prompt = format_example(
            tokenizer,
            question,
            truthful_answer,
        )

        untruthful_prompt = format_example(
            tokenizer,
            question,
            incorrect_answer,
        )

        training_samples.append(
            (
                truthful_prompt,
                untruthful_prompt,
            )
        )

    print(f"\nContrastive training pairs: {len(training_samples)}")

    if not training_samples:
        raise RuntimeError("No valid contrastive training pairs found.")

    # ------------------------------------------------------------------
    # Sanity-check the tokenization of one pair
    # ------------------------------------------------------------------
    print("\nInspecting one contrastive pair:")

    example_prompt = training_samples[0][0]

    token_ids = tokenizer(
        example_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    print("Last 20 tokens:")
    print(tokenizer.convert_ids_to_tokens(token_ids[-20:].tolist()))

    print("\nDecoded ending:")
    print(tokenizer.decode(token_ids[-20:]))

    # The original CAA implementation commonly extracts the token
    # immediately before the final special token. We will verify the
    # tokenization above before relying on -2.
    READ_TOKEN_INDEX = -3

    # ------------------------------------------------------------------
    # Train CAA steering vectors
    # ------------------------------------------------------------------
    print("\nTraining steering vectors...")

    steering_vector = train_steering_vector(
        model=model,
        tokenizer=tokenizer,
        training_samples=training_samples,
        layers=LAYERS,
        layer_type="decoder_block",
        read_token_index=READ_TOKEN_INDEX,
        batch_size=BATCH_SIZE,
        move_to_cpu=True,
        show_progress=True,
    )

    # ------------------------------------------------------------------
    # Save each vector independently
    # ------------------------------------------------------------------
    print("\nSaving vectors...")

    for layer in LAYERS:
        vector = steering_vector.layer_activations[layer]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"truthfulness_l{layer}.pt",
        )

        torch.save(
            vector.cpu(),
            output_path,
        )

        print(
            f"Layer {layer}: "
            f"shape={tuple(vector.shape)}, "
            f"norm={vector.norm().item():.6f}, "
            f"saved={output_path}"
        )

        # Save metadata describing exactly how the vectors were generated.
        metadata = {
            "model": MODEL_NAME,
            "layers": LAYERS,
            "dataset": "truthfulqa/truthful_qa:generation",
            "seed": SEED,
            "train_fraction": TRAIN_FRACTION,
            "num_training_pairs": len(training_samples),
            "read_token_index": READ_TOKEN_INDEX,
            "incorrect_answer_selection": "first incorrect_answers entry",
        }

        metadata_path = os.path.join(
            OUTPUT_DIR,
            "metadata.pt",
        )

        torch.save(metadata, metadata_path)

    print("\nDone.")


if __name__ == "__main__":
    main()