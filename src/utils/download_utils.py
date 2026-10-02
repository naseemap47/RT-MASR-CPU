"""
Model Downloader for Qwen3-ASR ONNX Engine.

Downloads required ONNX model files and tokenizer configurations from HuggingFace Hub.
"""

import argparse
from pathlib import Path
import tempfile
import shutil
import os
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "Daumee/Qwen3-ASR-0.6B-ONNX-CPU"
REQUIRED_FILES = [
    "decoder_init.int8.onnx",
    "decoder_step.int8.onnx",
    "embed_tokens.bin",
    "encoder_conv.onnx",
    "encoder_conv.onnx.data",
    "encoder_transformer.onnx",
    "encoder_transformer.onnx.data",
]

def move_files(src: Path, dst: Path):
    files_list = os.listdir(src)
    for file in files_list:
        shutil.move(src / file, dst)

def download_qwen3_onnx_models(
    target_dir: str = "models/qwen3-asr-onnx",
    repo_id: str = DEFAULT_REPO_ID,
    force: bool = False,
) -> Path:
    """
    Download Qwen3-ASR ONNX models from Hugging Face Hub if missing.

    Args:
        target_dir: Directory where model files will be saved.
        repo_id: HuggingFace model repo ID.
        force: Force re-downloading even if files exist.

    Returns:
        Path to the target directory containing models.
    """
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    missing = [f for f in REQUIRED_FILES if not (out_dir / f).exists()]

    if not missing and not force:
        print(f"All required ONNX model files are already present in '{out_dir}'.")
        return out_dir

    print(f"Downloading missing Qwen3-ASR ONNX model files from '{repo_id}'...")
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["onnx_models/*", "tokenizer.json"],
            local_dir=temp_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded model files to '{temp_dir}'.")
        move_files(Path(temp_dir) / "onnx_models", out_dir)
        shutil.move(Path(temp_dir) / "tokenizer.json", out_dir)
        print(f"Moved files to '{out_dir}'.")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-ASR ONNX models from HuggingFace")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models/qwen3-asr-onnx",
        help="Target directory for ONNX models (default: onnx_models)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download all files",
    )
    args = parser.parse_args()

    download_qwen3_onnx_models(
        target_dir=args.output_dir,
        repo_id=args.repo_id,
        force=args.force,
    )


if __name__ == "__main__":
    main()
