"""
Model Downloader for RT-MASR backends.

Downloads model files from HuggingFace Hub using settings read directly
from the config YAML files.

Three download strategies (selected via download.method in the model YAML):

  "onnx"     — DownloadModels.download_onnx()
      Fetches only onnx_models/* + tokenizer.json from the ONNX repo,
      moves them into the target directory, and verifies required files.

  "snapshot" — DownloadModels.download_snapshot()
      Full snapshot_download() of a HuggingFace repo into a local dir.
      Used for Transformers (safetensors) model weights.

  "whisper"  — DownloadModels.download_whisper()
      Downloads Whisper ONNX model tarballs from Wasabi/S3 for a given
      precision (int8 | fp16 | fp32), extracts them into target_dir.

Config-driven entrypoint
------------------------
    downloader = DownloadModels()
    downloader.download_from_config("config/models/qwen3_onnx.yaml")
    downloader.download_from_config("config/models/qwen3_0.6b.yaml")
    downloader.download_from_config("config/models/qwen3_1.7b.yaml")
    downloader.download_from_config("config/models/whisper_onnx.yaml")

Or to download all registered models in one call:
    downloader.download_all("config/config.yaml")
"""

from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from core.config import load_config, load_model_registry, load_server_config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _move_files(src: Path, dst: Path) -> None:
    """Move every file in *src* into *dst* (one level deep)."""
    for name in os.listdir(src):
        shutil.move(str(src / name), str(dst))


# ── Downloader ────────────────────────────────────────────────────────────────

class DownloadModels:
    """Download RT-MASR model weights (Qwen3-ASR and Whisper ONNX) from remote sources."""

    # ------------------------------------------------------------------
    # Low-level strategies
    # ------------------------------------------------------------------

    def download_onnx(
        self,
        repo_id: str,
        target_dir: str,
        required_files: list[str] | None = None,
        force: bool = False,
    ) -> Path:
        """
        Download ONNX model artefacts from an HF repo.

        Fetches ``onnx_models/*`` and ``tokenizer.json`` from *repo_id*
        and moves them flat into *target_dir*.

        Args:
            repo_id:        HuggingFace repository ID.
            target_dir:     Local destination directory.
            required_files: List of filenames to check before skipping.
                            If None, uses the REQUIRED_FILES constant.
            force:          Re-download even if files already exist.

        Returns:
            Path to *target_dir*.
        """
        _required = required_files or [
            "decoder_init.int8.onnx",
            "decoder_step.int8.onnx",
            "embed_tokens.bin",
            "encoder_conv.onnx",
            "encoder_conv.onnx.data",
            "encoder_transformer.onnx",
            "encoder_transformer.onnx.data",
            "tokenizer.json",
        ]

        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        missing = [f for f in _required if not (out_dir / f).exists()]
        if not missing and not force:
            print(f"All required ONNX files already present in '{out_dir}'.")
            return out_dir

        print(f"Downloading ONNX model files from '{repo_id}' → '{out_dir}' ...")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=["onnx_models/*", "tokenizer.json"],
                local_dir=tmp,
            )
            _move_files(tmp_path / "onnx_models", out_dir)
            tokenizer_src = tmp_path / "tokenizer.json"
            if tokenizer_src.exists():
                shutil.move(str(tokenizer_src), str(out_dir))

        print(f"ONNX model ready in '{out_dir}'.")
        return out_dir

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        force: bool = False,
    ) -> Path:
        """
        Download a complete HuggingFace repo snapshot (Transformers weights).

        Args:
            repo_id:   HuggingFace repository ID (e.g. "Qwen/Qwen3-ASR-0.6B").
            local_dir: Local destination directory.
            force:     Re-download even if the directory already exists.

        Returns:
            Path to *local_dir*.
        """
        out_dir = Path(local_dir)

        if out_dir.exists() and any(out_dir.iterdir()) and not force:
            print(f"Model snapshot already present in '{out_dir}'.")
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading snapshot from '{repo_id}' → '{out_dir}' ...")
        snapshot_download(repo_id=repo_id, local_dir=str(out_dir))
        print(f"Snapshot ready in '{out_dir}'.")
        return out_dir

    # ------------------------------------------------------------------
    # Config-driven entrypoints
    # ------------------------------------------------------------------

    def download_from_config(
        self,
        model_config_path: str,
        force: bool = False,
    ) -> Path:
        """
        Download a model using settings from its per-model YAML file.

        Reads the ``download`` section of the given config YAML and
        dispatches to :meth:`download_onnx` or :meth:`download_snapshot`
        based on ``download.method``.

        Args:
            model_config_path: Path to a per-model config YAML, e.g.
                               ``"config/models/qwen3_onnx.yaml"``.
            force:             Force re-download.

        Returns:
            Path to the downloaded model directory.

        Example::

            downloader = DownloadModels()
            downloader.download_from_config("config/models/qwen3_onnx.yaml")
            downloader.download_from_config("config/models/qwen3_0.6b.yaml")
        """
        cfg = load_config(model_config_path)
        dl = cfg.get("download", {})
        method = dl.get("method", "snapshot")
        name = cfg.get("display_name", model_config_path)

        print(f"\n── {name} ({'method: ' + method}) ──")

        if method == "onnx":
            return self.download_onnx(
                repo_id=dl["repo_id"],
                target_dir=dl["target_dir"],
                required_files=dl.get("required_files"),
                force=force,
            )
        elif method == "snapshot":
            return self.download_snapshot(
                repo_id=dl["repo_id"],
                local_dir=dl["local_dir"],
                force=force,
            )
        elif method == "whisper":
            return self.download_whisper(
                target_dir=dl["target_dir"],
                precision=dl.get("precision", "int8"),
                force=force,
            )
        else:
            raise ValueError(
                f"Unknown download method '{method}' in '{model_config_path}'. "
                "Expected 'onnx', 'snapshot', or 'whisper'."
            )

    def download_all(
        self,
        config_path: str = "config/config.yaml",
        force: bool = False,
    ) -> dict[str, Path]:
        """
        Download every model listed in the model registry.

        Reads ``config.yaml`` → ``model_registry`` → iterates all entries
        and calls :meth:`download_from_config` for each.

        Args:
            config_path: Path to the top-level config.yaml.
            force:       Force re-download for all models.

        Returns:
            Dict mapping model name → downloaded directory path.
        """
        server_cfg = load_server_config(config_path)
        cfg_dir = Path(config_path).parent.parent
        registry_path = cfg_dir / server_cfg["model_registry"]

        entries = load_model_registry(str(registry_path))
        results: dict[str, Path] = {}

        print(f"Downloading {len(entries)} model(s) listed in '{registry_path}' ...")
        for entry in entries:
            model_name = entry["name"]
            model_cfg_path = cfg_dir / entry["config"]
            try:
                path = self.download_from_config(str(model_cfg_path), force=force)
                results[model_name] = path
            except Exception as exc:
                print(f"  [ERROR] Failed to download '{model_name}': {exc}")

        return results

    # ------------------------------------------------------------------
    # Whisper ONNX downloader
    # ------------------------------------------------------------------

    # Tarball URLs per precision (PINTO model-zoo, Wasabi S3)
    _WHISPER_URLS: dict = {
        "int8": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/resources_dynamic_range_quant_int8.tar.gz",
        "fp16": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/resources_float16.tar.gz",
        "fp32": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/resources_float32.tar.gz",
    }

    def download_whisper(
        self,
        target_dir: str = "models/whisper_int8",
        precision: str = "int8",
        force: bool = False,
    ) -> Path:
        """
        Download Whisper ONNX model files from the PINTO model-zoo (Wasabi S3).

        The tarball is extracted flat into *target_dir*.

        Args:
            target_dir: Local destination directory.
            precision:  One of ``"int8"``, ``"fp16"``, ``"fp32"``.
            force:      Re-download even if the directory already contains files.

        Returns:
            Path to *target_dir*.
        """
        supported = list(self._WHISPER_URLS)
        if precision not in supported:
            raise ValueError(
                f"Unknown Whisper precision '{precision}'. Supported: {supported}"
            )

        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_dir.exists() and any(out_dir.iterdir()) and not force:
            print(f"Whisper ONNX ({precision}) files already present in '{out_dir}'.")
            return out_dir

        url = self._WHISPER_URLS[precision]
        tar_file = out_dir.parent / f"whisper_{precision}.tar.gz"

        print(f"Downloading Whisper ONNX ({precision}) from '{url}' …")
        subprocess.run(["curl", "-L", url, "-o", str(tar_file)], check=True)
        print(f"Extracting to '{out_dir}' …")
        subprocess.run(["tar", "-zxvf", str(tar_file), "-C", str(out_dir)], check=True)
        tar_file.unlink(missing_ok=True)

        print(f"Whisper ONNX ({precision}) ready in '{out_dir}'.")
        return out_dir

    # ------------------------------------------------------------------
    # Legacy low-level methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    def qwen3_0_6b_asr_onnx_model(
        self,
        repo_id: str = "Daumee/Qwen3-ASR-0.6B-ONNX-CPU",
        target_dir: str = "models/qwen3-asr-onnx",
        force: bool = False,
    ) -> Path:
        """Backward-compatible wrapper → :meth:`download_onnx`."""
        return self.download_onnx(repo_id=repo_id, target_dir=target_dir, force=force)

    def qwen3_0_6b_asr_models(
        self,
        repo_id: str = "Qwen/Qwen3-ASR-0.6B",
        local_dir: str = "models/qwen3-asr-0.6b",
    ) -> Path:
        """Backward-compatible wrapper → :meth:`download_snapshot`."""
        return self.download_snapshot(repo_id=repo_id, local_dir=local_dir)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download RT-MASR model weights (Qwen3-ASR and Whisper ONNX)."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Name of a single model to download (must match a registry entry, "
            "e.g. 'qwen3_onnx', 'qwen3_0.6b', 'qwen3_1.7b', "
            "'whisper_int8_tiny|base|small|medium', 'whisper_fp16', 'whisper_fp32'; "
            "the alias 'whisper_int8' downloads all INT8 sizes). "
            "If omitted, all models in the registry are downloaded."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the top-level config.yaml (default: config/config.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model files already exist.",
    )
    args = parser.parse_args()

    downloader = DownloadModels()

    # Shorthand whisper aliases — dash and underscore forms both accepted.
    # Note: whisper_int8 / fp16 / fp32 are also full registry entries, so they
    # will be resolved via the registry path below. The aliases here act as a
    # fast-path that bypasses registry lookup for convenience.
    _WHISPER_ALIASES = {
        "whisper-int8": "int8",
        "whisper-fp16": "fp16",
        "whisper-fp32": "fp32",
        "whisper_int8": "int8",
        "whisper_fp16": "fp16",
        "whisper_fp32": "fp32",
    }
    if args.model in _WHISPER_ALIASES:
        precision = _WHISPER_ALIASES[args.model]
        downloader.download_whisper(
            target_dir=f"models/whisper_{precision}",
            precision=precision,
            force=args.force,
        )
    elif args.model:
        # Resolve per-model config path from the registry
        server_cfg = load_server_config(args.config)
        cfg_dir = Path(args.config).parent.parent
        registry_path = cfg_dir / server_cfg["model_registry"]
        entries = load_model_registry(str(registry_path))

        entry = next((e for e in entries if e["name"] == args.model), None)
        if entry is None:
            available = [e["name"] for e in entries]
            raise SystemExit(
                f"Unknown model '{args.model}'. Available: {available}"
            )
        model_cfg_path = cfg_dir / entry["config"]
        downloader.download_from_config(str(model_cfg_path), force=args.force)
    else:
        downloader.download_all(config_path=args.config, force=args.force)
