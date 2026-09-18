"""ATCAT light-curve wrapper (impl=atcat_lightcurve), against the real exported ONNX model:

    session = onnxruntime.InferenceSession("atcat_f32.onnx")
    emb, = session.run(["sequence"], {"flux": ..., "flux_err": ..., "time": ...,
                                       "mask": ..., "channel_index": ...})
    # emb: (B, 243, 384) — per-timestep transformer features, token-level

`light-curve/atcat` is **ATCAT: Astronomical Timeseries CAusal Transformer** (Tung 2025,
arXiv:2511.00614), exported to ONNX by the SNAD `light-curve` group *as an embedding model* — it
"exposes hidden representations before the final classifier head". We request only the `sequence`
output (onnxruntime prunes the unused `last`/`mean` pooling heads), because this codebase's encoder
contract is token-level and forbids pooling: the Q-Former exists to cross-attend over structure that
pooling destroys (see encoders/base.py, §5).

**Why a pretrained encoder rather than a hand-built featurizer.** AION cannot encode light curves at
all — `aion-base`/`large`/`xlarge` all declare the same 40 domains, none with a time axis, and its
photometry is single-epoch `Scalar` classes with `num_tokens = 1`. ATCAT fills that gap while
keeping every architectural property AION has here: frozen, run once offline, cached to fp16 shards,
covered by the existing shuffle/ablation groundedness gate. It is also tiny — ~3M parameters,
11.7 MB — so caching the whole dataset is near-instant on CPU.

**ATCAT is a causal transformer over the time axis**, which is why no positional encoding needs to be
smuggled into the token features: each of the 243 per-timestep hidden states already carries temporal
context before the Q-Former (which has no positional encoding of its own, and is asserted
permutation-invariant by tests/test_masking.py) ever sees it.

**243 is fixed, not a tunable.** Probing the export shows `batch` is the only symbolic axis. Input
preparation — accepted-point masking, detection-window trimming, seeded downsampling and padding to
exactly 243 — lives in data/transients_dataset.py's `prepare_lightcurve_arrays`, deliberately kept
numpy-only so that logic is testable without a GPU stack. This wrapper only runs the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from captioner.encoders.base import EncoderLoadError

ATCAT_OUT_DIM = 384  # ATCAT's hidden width
ATCAT_SEQ_LEN = 243  # static in the ONNX export
ATCAT_INPUTS = ("flux", "flux_err", "time", "mask", "channel_index")

# ONNX tensor type string -> numpy dtype. The export's declared dtypes are read at load time and
# inputs cast to match, rather than assumed: `mask`/`channel_index` could plausibly be exported as
# float, int32, int64 or bool, and guessing wrong is an onnxruntime type error on the first real
# batch rather than something caught here.
_ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(bfloat16)": np.float32,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(bool)": np.bool_,
}


class AtcatLightcurveEncoder:
    def __init__(
        self,
        name: str,
        hf_path: str,
        revision: str | None,
        kwargs: dict[str, Any],
        num_encoder_tokens: int,
    ) -> None:
        self.name = name
        self.hf_path = hf_path
        self.revision = revision
        self.onnx_file = kwargs.get("onnx_file", "atcat_f32.onnx")
        self.output_name = kwargs.get("output", "sequence")
        self.out_dim = ATCAT_OUT_DIM
        self.seq_len = ATCAT_SEQ_LEN

        if self.output_name != "sequence":
            raise ValueError(
                f"output={self.output_name!r} is not usable here. ATCAT's `last` and `mean` outputs "
                "are pooled (B, 384) vectors; this codebase's encoder contract requires token-level "
                "(B, T, out_dim) output and forbids pooling (see encoders/base.py, §5). Use "
                "'sequence'."
            )
        if int(num_encoder_tokens) != ATCAT_SEQ_LEN:
            raise ValueError(
                f"modalities.{name}.max_tokens={num_encoder_tokens} but ATCAT's ONNX export has a "
                f"static sequence length of {ATCAT_SEQ_LEN} (only `batch` is a symbolic axis). Set "
                f"max_tokens: {ATCAT_SEQ_LEN} in configs/modalities.yaml — it is a property of the "
                "model, not a free choice."
            )

        self._session = None
        self._input_dtypes: dict[str, Any] = {}

    def load(self, device: str = "cpu") -> None:
        try:
            import onnxruntime
        except ImportError as e:  # pragma: no cover - dependency guidance only
            raise EncoderLoadError(
                "onnxruntime is required for impl=atcat_lightcurve but is not installed. "
                "`pip install onnxruntime` (CPU is fine — this encoder runs once, offline, and the "
                "model is ~12MB)."
            ) from e

        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self.hf_path,
            filename=self.onnx_file,
            revision=self.revision,
        )

        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda"):
            if "CUDAExecutionProvider" in set(onnxruntime.get_available_providers()):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # Otherwise stay on CPU rather than failing: this encoder is cheap enough that a
            # missing onnxruntime-gpu build is not worth stopping a cache run over.

        self._session = onnxruntime.InferenceSession(local_path, providers=providers)
        self._input_dtypes = {i.name: i.type for i in self._session.get_inputs()}

        missing = [n for n in ATCAT_INPUTS if n not in self._input_dtypes]
        if missing:
            raise EncoderLoadError(
                f"{self.onnx_file} does not declare the expected inputs {missing}; it exposes "
                f"{sorted(self._input_dtypes)}. The export's input contract has changed — check "
                "the model card for light-curve/atcat before pinning a new revision."
            )
        output_names = {o.name for o in self._session.get_outputs()}
        if self.output_name not in output_names:
            raise EncoderLoadError(
                f"{self.onnx_file} has no output named {self.output_name!r}; it exposes "
                f"{sorted(output_names)}."
            )

    def _cast(self, name: str, arr: np.ndarray) -> np.ndarray:
        onnx_type = self._input_dtypes.get(name, "tensor(float)")
        dtype = _ONNX_TO_NUMPY.get(onnx_type)
        if dtype is None:
            raise EncoderLoadError(
                f"Unhandled ONNX input dtype {onnx_type!r} for input {name!r}. Add it to "
                "_ONNX_TO_NUMPY."
            )
        return np.ascontiguousarray(arr, dtype=dtype)

    @torch.no_grad()
    def encode(self, batch: dict[str, Any]) -> Tensor:
        """batch: the five ATCAT inputs, each (B, 243) — see data/transients_dataset.py's
        `prepare_lightcurve_arrays`, which is the only supported way to build them.

        `mask` is not optional and must not be all-zero for any row: the dataset's
        `atcat_band_id = 0` excluded-i sentinel is *u-band* to ATCAT, and only the mask keeps those
        points out of the model.
        """
        if self._session is None:
            raise EncoderLoadError("encode() called before load()")

        missing = [n for n in ATCAT_INPUTS if n not in batch]
        if missing:
            raise ValueError(
                f"batch is missing required ATCAT input(s) {missing}. Expected all of "
                f"{list(ATCAT_INPUTS)}, each shaped (B, {self.seq_len}); build them with "
                "data/transients_dataset.py:prepare_lightcurve_arrays."
            )

        feed = {}
        shapes = set()
        for name in ATCAT_INPUTS:
            arr = batch[name]
            arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != self.seq_len:
                raise ValueError(
                    f"batch[{name!r}] has shape {tuple(arr.shape)}; expected (B, {self.seq_len}). "
                    "ATCAT's export has a static sequence length, so inputs must be padded to "
                    "exactly that."
                )
            shapes.add(arr.shape)
            feed[name] = self._cast(name, arr)

        if len(shapes) != 1:
            raise ValueError(f"ATCAT inputs disagree on shape: {sorted(shapes)}")

        valid_per_row = np.asarray(feed["mask"], dtype=np.int64).sum(axis=1)
        if (valid_per_row == 0).any():
            n_bad = int((valid_per_row == 0).sum())
            raise ValueError(
                f"{n_bad} row(s) in this batch have an all-zero `mask`, i.e. no valid photometry. "
                "ATCAT would see an empty sequence, and the excluded-i-band sentinel "
                "(atcat_band_id=0) would be its only content — which ATCAT reads as u-band. Such "
                "objects must be excluded upstream, not encoded."
            )

        out = self._session.run([self.output_name], feed)[0]
        emb = torch.from_numpy(np.asarray(out, dtype=np.float32))

        if emb.dim() != 3:
            raise EncoderLoadError(
                f"Expected token-level (B, T, out_dim) from ATCAT's {self.output_name!r} output, "
                f"got shape {tuple(emb.shape)}. Do not pool in this wrapper — see §5/§10."
            )
        if emb.shape[-1] != self.out_dim:
            raise EncoderLoadError(
                f"ATCAT returned out_dim={emb.shape[-1]}, expected {self.out_dim}. Update "
                "configs/modalities.yaml's lightcurve.out_dim and re-cache."
            )
        return emb
