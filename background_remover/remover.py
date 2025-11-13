"""Core logic for removing image backgrounds using U2NetP."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

_MODEL_FILENAME = "u2netp.onnx"
_INPUT_SIZE = (320, 320)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the bundled ONNX model file is missing."""


@dataclass
class BackgroundRemover:
    """High-level interface for removing backgrounds from images.

    Parameters
    ----------
    model_path:
        Optional path to the ONNX model. When omitted the package looks for
        ``u2netp.onnx`` in the ``background_remover/models`` directory.
    providers:
        Optional list of ONNX Runtime execution providers. By default this
        enables GPU acceleration when available and falls back to CPU.
    """

    model_path: Optional[os.PathLike[str] | str] = None
    providers: Optional[list[str]] = None

    def __post_init__(self) -> None:
        self._session = self._load_session()
        self._input_name = self._session.get_inputs()[0].name

    @functools.cached_property
    def model_file(self) -> Path:
        if self.model_path is not None:
            return Path(self.model_path)
        return Path(__file__).resolve().parent / "models" / _MODEL_FILENAME

    def _load_session(self) -> ort.InferenceSession:
        model_file = self.model_file
        if not model_file.exists():
            raise ModelNotFoundError(
                "Could not find the background removal model. Expected to find "
                f"{_MODEL_FILENAME} next to the package."
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        providers = self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            return ort.InferenceSession(
                model_file.as_posix(),
                sess_options=session_options,
                providers=providers,
            )
        except Exception:  # pragma: no cover - falls back for systems without GPU
            return ort.InferenceSession(
                model_file.as_posix(),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Return a copy of *image* with an alpha channel created from the mask."""

        original_size = image.size
        image_rgb = image.convert("RGB")
        mask = self._predict_mask(image_rgb)
        mask = mask.resize(original_size, Image.BILINEAR)

        image_rgba = image_rgb.convert("RGBA")
        image_rgba.putalpha(mask)
        return image_rgba

    def _predict_mask(self, image: Image.Image) -> Image.Image:
        input_tensor = self._prepare_input(image)
        predictions = self._session.run(None, {self._input_name: input_tensor})
        mask = predictions[0][0, 0, :, :]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        return mask_image

    def _prepare_input(self, image: Image.Image) -> np.ndarray:
        resized = image.resize(_INPUT_SIZE, Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (array - mean) / std
        chw = normalized.transpose(2, 0, 1)
        return np.expand_dims(chw, 0)
