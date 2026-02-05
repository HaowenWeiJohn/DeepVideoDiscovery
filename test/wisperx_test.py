"""
WhisperX Video Transcription Example

Usage:
    python test/wisperx_test.py path/to/video.mp4
    python test/wisperx_test.py path/to/audio.mp3 --model large-v3 --batch_size 16
    python test/wisperx_test.py path/to/video.mp4 --diarize --hf_token YOUR_HF_TOKEN
    python test/wisperx_test.py path/to/video.mp4 --device cpu --compute_type int8

Requirements:
    pip install whisperx
    ffmpeg must be installed system-wide
"""

import argparse
import gc
import json
import os

import torch
import whisperx


def transcribe_video(
    audio_file: str,
    model_name: str = "large-v2",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str = None,
    diarize: bool = False,
    hf_token: str = None,
    output_json: str = None,
):
    """Transcribe a video/audio file using WhisperX.

    Args:
        audio_file: Path to the video or audio file.
        model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3).
        device: Device to use ("cuda" or "cpu").
        compute_type: Compute type ("float16", "float32", or "int8").
        batch_size: Batch size for transcription. Reduce if low on GPU memory.
        language: Language code (e.g. "en"). Auto-detected if None.
        diarize: Whether to run speaker diarization.
        hf_token: HuggingFace token, required if diarize=True.
        output_json: Path to save results as JSON. Defaults to <audio_file>.json.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"File not found: {audio_file}")

    # ---- Stage 1: Transcribe ----
    print(f"Loading WhisperX model '{model_name}' on {device}...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)

    print(f"Loading audio: {audio_file}")
    audio = whisperx.load_audio(audio_file)

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result["language"]
    print(f"Detected language: {detected_lang}")
    print(f"Found {len(result['segments'])} segments")

    # Free model memory
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Stage 2: Word-level Alignment ----
    print(f"Loading alignment model for '{detected_lang}'...")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
    except ValueError:
        print(f"Language '{detected_lang}' not supported for alignment, falling back to English.")
        align_model, metadata = whisperx.load_align_model(
            language_code="en", device=device
        )

    print("Aligning word timestamps...")
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    del align_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Stage 3: Speaker Diarization (optional) ----
    if diarize:
        if not hf_token:
            print("Warning: --hf_token required for diarization. Skipping.")
        else:
            print("Running speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio_file)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            del diarize_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- Print Results ----
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    print("=" * 60)
    for seg in result["segments"]:
        speaker = seg.get("speaker", "")
        prefix = f"{speaker}: " if speaker else ""
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {prefix}{seg['text'].strip()}")

    # ---- Save to JSON ----
    if output_json is None:
        base, _ = os.path.splitext(audio_file)
        output_json = base + "_transcript.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\nSaved transcript to: {output_json}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe video/audio using WhisperX")
    parser.add_argument("audio_file", help="Path to video or audio file")
    parser.add_argument("--model", default="large-v2", help="Whisper model name (default: large-v2)")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto-detect)")
    parser.add_argument("--compute_type", default=None, help="Compute type: float16, float32, int8 (default: float16 for cuda, int8 for cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--language", default=None, help="Language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token for diarization (or set HF_TOKEN env var)")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <input>_transcript.json)")
    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {device}")

    # Set compute type based on device
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    transcribe_video(
        audio_file=args.audio_file,
        model_name=args.model,
        device=device,
        compute_type=compute_type,
        batch_size=args.batch_size,
        language=args.language,
        diarize=args.diarize,
        hf_token=args.hf_token,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
