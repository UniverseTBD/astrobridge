"""name -> class, built from configs/modalities.yaml.

This is the only place `impl` strings are mapped to code. Adding a modality with an existing
`impl` (e.g. a second image-like modality reusing aion_image) requires no code change at all;
adding one with a genuinely new encoder family requires ~30 lines here plus a new wrapper file.
"""
from __future__ import annotations

import hashlib
import json

from omegaconf import DictConfig, OmegaConf

from captioner.encoders.aion_image import AionImageEncoder
from captioner.encoders.aion_spectrum import AionSpectrumEncoder
from captioner.encoders.atcat_lightcurve import AtcatLightcurveEncoder
from captioner.encoders.base import ModalityEncoder, assert_out_dim

IMPL_REGISTRY: dict[str, type] = {
    "aion_image": AionImageEncoder,
    "aion_spectrum": AionSpectrumEncoder,
    "atcat_lightcurve": AtcatLightcurveEncoder,
}


def encoder_spec(modality_cfg: DictConfig) -> dict:
    """The full resolved encoder spec — written into the cache index and asserted at train time."""
    enc = modality_cfg.encoder
    return {
        "impl": enc.impl,
        "hf_path": enc.hf_path,
        "revision": enc.revision,
        "kwargs": OmegaConf.to_container(enc.get("kwargs", {}), resolve=True),
    }


def encoder_hash(spec: dict) -> str:
    """Cache key must include encoder identity (impl, hf_path, revision, kwargs), not just the
    modality name — otherwise swapping an encoder silently reuses stale embeddings.
    """
    canonical = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_encoder(name: str, modality_cfg: DictConfig, device: str = "cpu") -> ModalityEncoder:
    spec = encoder_spec(modality_cfg)
    impl = spec["impl"]
    if impl not in IMPL_REGISTRY:
        raise KeyError(
            f"Unknown encoder impl={impl!r} for modality {name!r}. "
            f"Known impls: {sorted(IMPL_REGISTRY)}. Add a wrapper in encoders/ and register it here."
        )
    cls = IMPL_REGISTRY[impl]
    encoder = cls(
        name=name,
        hf_path=spec["hf_path"],
        revision=spec["revision"],
        kwargs=spec["kwargs"],
        num_encoder_tokens=int(modality_cfg.max_tokens),
    )
    encoder.load(device=device)
    assert_out_dim(int(modality_cfg.out_dim), encoder.out_dim, impl, spec["hf_path"])
    return encoder


def build_all_encoders(modalities_cfg: DictConfig, device: str = "cpu") -> dict[str, ModalityEncoder]:
    return {
        name: build_encoder(name, cfg, device=device)
        for name, cfg in modalities_cfg.modalities.items()
    }
