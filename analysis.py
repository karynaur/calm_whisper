# analysis.py
import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import matplotlib.pyplot as plt
import numpy as np

def create_mask_head_hook(head_to_mask_idx: int):
    """Factory function to create a hook that masks a specific attention head."""

    def mask_head_hook(module, input, output):
        attn_output = output[0]
        head_dim = module.head_dim
        start_index = head_to_mask_idx * head_dim
        end_index = start_index + head_dim
        attn_output[:, :, start_index:end_index] = 0
        return (attn_output, output[1], output[2])

    return mask_head_hook


def analyze_hallucination_rate(
    model, processor, noise_dir: str, device: str, head_to_mask_idx = None
):
    """
    Calculates the hallucination rate for the given model on a directory of
    non-speech audio.
    """
    audio_files = list(Path(noise_dir).rglob("*.wav")) + list(
        Path(noise_dir).rglob("*.flac")
    )
    total_chunks = 0
    hallucinated_count = 0
    target_sr = processor.feature_extractor.sampling_rate

    for audio_file in tqdm(audio_files, desc="  Analyzing files", leave=False):
        try:
            waveform, sr = torchaudio.load(audio_file)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, sr, target_sr
                )
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if waveform.shape[1] == 0:
                continue

            # Process each file as a single chunk
            chunk_list = [waveform.squeeze(0).numpy()]
            total_chunks += len(chunk_list)

            # Pad all inputs to the model's required max length (3000 frames)
            inputs = processor(
                chunk_list,
                return_tensors="pt",
                sampling_rate=target_sr,
                padding="max_length",
            )
            input_features = inputs.input_features.to(device).to(model.dtype)

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=None,       
                    suppress_tokens=[],
                )

            transcriptions = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            for t in transcriptions:
                with open(f'txts/wlv3_pretrained/{head_to_mask_idx}.txt', 'a') as f:
                    f.write(f"{t}\n")
                if t.strip():
                    hallucinated_count += 1
        except Exception as e:
            # raise e
            print(f"Skipping file {audio_file.name} due to error: {e}")
            continue

    if total_chunks == 0:
        print("Warning: No audio chunks were processed.")
        return 0.0
    return (hallucinated_count / total_chunks) * 100


def plot_head_analysis(results, baseline_rate, output_path, model_name):
    """Plots the head analysis results in a sorted bar chart."""
    # Sort results by hallucination rate (ascending)
    sorted_results = sorted(results.items(), key=lambda item: item[1])
    heads = [f"Head #{item[0]}" for item in sorted_results]
    rates = [item[1] for item in sorted_results]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(heads, rates, color="skyblue", edgecolor="black")
    plt.axhline(
        y=baseline_rate,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Baseline Rate ({baseline_rate:.2f}%)",
    )

    plt.ylabel("Hallucination Rate (%)", fontsize=12)
    plt.xlabel("Decoder Attention Head (Masked)", fontsize=12)
    plt.title(
        f"Impact of Masking Heads on Hallucination Rate for {model_name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(baseline_rate, max(rates)) * 1.1)
    plt.legend()
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.savefig(output_path, dpi=300)
    print(f"\nAnalysis plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Identify hallucinatory heads in a Whisper model."
    )
    parser.add_argument(
        "--noise_dir",
        type=str,
        default="/data/adi/data/noise_data/urbansound/audio",
        help="Path to the root directory of non-speech audio files (e.g., UrbanSound8K).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/data/adi/checkpoints/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1/",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="whisper_largev3_head_analysis_us8k.png",
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} with dtype: {torch_dtype}")

    print(f"Loading model and processor for '{args.model_id}'...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, cache_dir="/data/adi/checkpoints",
    ).to(device)
    processor = WhisperProcessor.from_pretrained(args.model_id)

    # CRITICAL: Disable forced tokens to allow for empty transcriptions
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.eval()

    # --- Step 1: Get Baseline (No Masking) ---
    print("\n--- Step 1: Calculating Baseline Hallucination Rate (No Masking) ---")
    baseline_rate = analyze_hallucination_rate(
        model, processor, args.noise_dir, device, head_to_mask_idx="baseline"
    )
    print(f"Baseline Hallucination Rate: {baseline_rate:.2f}%")

    # --- Step 2: Analyze Impact of Masking Each Head ---
    print("\n--- Step 2: Analyzing Impact of Masking Each Head ---")
    num_heads = model.config.decoder_attention_heads
    head_results = {}

    for head_idx in tqdm(range(num_heads), desc="Testing All Heads"):
        hooks = []
        try:
            # Attach a hook to every decoder layer to mask the same head index
            hook_fn = create_mask_head_hook(head_idx)
            for layer in model.model.decoder.layers:
                handle = layer.self_attn.register_forward_hook(hook_fn)
                hooks.append(handle)

            rate = analyze_hallucination_rate(
                model, processor, args.noise_dir, device, head_idx
            )
            print(f"Head {head_idx} - Hallucination Rate: {rate:.2f}%")
            head_results[head_idx] = rate
        finally:
            # Always remove hooks to restore original model behavior
            for handle in hooks:
                handle.remove()

    # --- Step 3: Report Results ---
    print("\n--- Analysis Complete ---")
    sorted_heads = sorted(head_results.items(), key=lambda item: item[1])

    print(f"\n{'Head #':<10} | {'Resulting Hallucination Rate (%)'}")
    print("-" * 45)
    for head, rate in sorted_heads:
        change = rate - baseline_rate
        print(f"{head:<10} | {rate:>25.2f} ({change:+.2f}%)")

    # Identify the most impactful heads (those that reduce hallucination most)
    top_3_heads = [h for h, r in sorted_heads[:3]]
    print(
        f"\nConclusion: The most hallucinatory heads for '{args.model_id}' appear to be: {top_3_heads}"
    )
    print("These are the best candidates for 'calm-down fine-tuning'.")

    plot_head_analysis(
        head_results, baseline_rate, args.output_plot, args.model_id
    )


if __name__ == "__main__":
    main()
