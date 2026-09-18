"""AION spectrum wrapper (impl=aion_spectrum), against the real `aion` package API:

    spectrum = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)  # or
    spectrum = SDSSSpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)  # same fields
    tokens = codec_manager.encode(spectrum)
    emb = model.encode(tokens, num_encoder_tokens=N)   # (B, N, 768) — token-level

Confirmed against a working linear-probe (`AionEncoder(SpectrumEncoder)`) that mean-pooled `emb`
over dim=1; this wrapper is that same call with pooling removed, since the Q-Former needs the
full token grid (§5). Where `ivar`/`mask` are missing from a batch, that probe defaulted to
`ivar=ones_like(flux)` / `mask=zeros_like(flux, dtype=bool)` (i.e. "fully trusted, fully
unmasked") — same default kept here. `wavelength` has no safe default: it's not derivable from
flux alone, so a batch missing it is a hard error rather than a silent guess.

**`DESISpectrum` vs `SDSSSpectrum` — confirmed real, not a hypothetical.** AION's own source
(github.com/PolymathicAI/AION, aion/modalities.py) defines these as two separate Modality
subclasses with identical fields but distinct AION-internal domains (`tok_spectrum_desi` vs
`tok_spectrum_sdss`, confirmed via aion-base's config.json). AstroBridge-Data's spectra come from
four crossmatch files — DESI-only, SDSS-only, and two DESI+SDSS combinations — so a batch can
contain both survey types. Routing every spectrum through `DESISpectrum` regardless of origin
(what this wrapper originally did) wouldn't error — it would silently tell AION "this is DESI
data" for spectra that are actually SDSS, which is wrong input, not a crash. `batch['survey']`
(populated by data/spectra_dataset.py from each file's own `survey` column, or inferred from
filename for the single-survey files) is required per object; the batch is split by survey value
and each group goes through its matching Modality class, then results are reassembled in the
original row order.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from captioner.encoders.aion_common import load_aion
from captioner.encoders.base import EncoderLoadError

AION_OUT_DIM = 768  # AION-base's shared embedding width — same for every modality's tokens


class AionSpectrumEncoder:
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
        self.num_encoder_tokens = num_encoder_tokens
        self.out_dim = AION_OUT_DIM
        self._model = None
        self._codec_manager = None
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        self._device = device
        self._model, self._codec_manager = load_aion(self.hf_path, self.revision, device)

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """batch['flux']: (B, L). Optional batch['ivar'], batch['mask']: (B, L).
        batch['wavelength']: (B, L) or (L,) — required, no safe default.
        batch['survey']: list[str] of length B, each "desi" or "sdss" — required, routes each
        object to the matching AION Modality class. See module docstring for why this matters.
        """
        if self._model is None:
            raise EncoderLoadError("encode() called before load()")

        from aion.modalities import DESISpectrum, SDSSSpectrum

        flux = batch["flux"].to(self._device)
        if "wavelength" not in batch:
            raise ValueError(
                "batch['wavelength'] is required to encode a spectrum — AstroBridge-Data's raw "
                "`spectrum` dict field name for the wavelength grid must be confirmed and wired "
                "into scripts/02_cache_embeddings.py's spectra batch loader; there is no safe "
                "default to fall back to."
            )
        wavelength = batch["wavelength"].to(self._device)
        if wavelength.dim() == 1:
            wavelength = wavelength.unsqueeze(0).expand(flux.shape[0], -1)

        ivar = batch["ivar"].to(self._device) if "ivar" in batch else torch.ones_like(flux)
        mask = batch["mask"].to(self._device) if "mask" in batch else torch.zeros_like(flux, dtype=torch.bool)

        if "survey" not in batch:
            raise ValueError(
                "batch['survey'] is required (one of 'desi'/'sdss' per object) — AION has "
                "separate DESISpectrum/SDSSSpectrum modality classes with distinct internal "
                "domains, and AstroBridge-Data's crossmatch files mix both survey origins. "
                "Wire this from data/spectra_dataset.py's survey column into "
                "scripts/02_cache_embeddings.py's spectra batch loader — there is no safe "
                "default (guessing wrong silently mislabels the spectrum's survey origin)."
            )
        survey = list(batch["survey"])
        unknown = sorted(set(survey) - {"desi", "sdss"})
        if unknown:
            raise ValueError(
                f"batch['survey'] contains unrecognized value(s) {unknown} — expected only "
                "'desi' or 'sdss'. AION has no modality class for other surveys; check "
                "data/spectra_dataset.py's survey-inference logic against the real data."
            )

        # AION's codec projects onto its latent grid with torch.searchsorted, which REQUIRES a
        # sorted wavelength row and silently returns undefined indices otherwise. AstroBridge-
        # Data pads `spectrum.lambda` with a `-1` sentinel on 94% of SDSS rows, which made every
        # cached v5/v6 spectrum embedding near-identical (99.6% shared token codes; a class probe
        # below its own majority baseline). That failed completely silently for two full training
        # runs, so it is a hard error here rather than a warning. Trim with
        # `captioner.data.spectra_dataset.trimmed_spectrum_arrays` before calling this.
        if wavelength.shape[1] > 1:
            steps = wavelength[:, 1:] - wavelength[:, :-1]
            bad_rows = (steps <= 0).any(dim=1)
            if bool(bad_rows.any()):
                i = int(bad_rows.nonzero()[0, 0])
                row = wavelength[i]
                j = int((row[1:] - row[:-1] <= 0).nonzero()[0, 0])
                raise ValueError(
                    f"batch['wavelength'] must be strictly increasing along dim=1: "
                    f"{int(bad_rows.sum())}/{wavelength.shape[0]} rows are not. First offender "
                    f"row {i} at index {j}: ...{row[max(0, j - 1):j + 3].tolist()}... "
                    "AION's interp1d uses torch.searchsorted, which gives undefined results on an "
                    "unsorted grid — this does not raise inside AION, it silently returns garbage "
                    "embeddings. Drop `wavelength <= 0` sentinels and sort ascending first (see "
                    "trimmed_spectrum_arrays in captioner.data.spectra_dataset)."
                )

        B = flux.shape[0]
        emb_by_index: dict[int, Tensor] = {}
        for survey_name, modality_cls in (("desi", DESISpectrum), ("sdss", SDSSSpectrum)):
            idx = [i for i in range(B) if survey[i] == survey_name]
            if not idx:
                continue
            idx_t = torch.tensor(idx, device=self._device)
            spectrum = modality_cls(
                flux=flux.index_select(0, idx_t),
                ivar=ivar.index_select(0, idx_t),
                mask=mask.index_select(0, idx_t),
                wavelength=wavelength.index_select(0, idx_t),
            )
            tokens = self._codec_manager.encode(spectrum)
            group_emb = self._model.encode(tokens, num_encoder_tokens=self.num_encoder_tokens)
            for local_i, global_i in enumerate(idx):
                emb_by_index[global_i] = group_emb[local_i]

        emb = torch.stack([emb_by_index[i] for i in range(B)], dim=0)

        if emb.dim() != 3:
            raise EncoderLoadError(
                f"Expected token-level (B, T, out_dim) from AION.encode, got shape "
                f"{tuple(emb.shape)}. Do not mean-pool in this wrapper — see §5/§10."
            )
        return emb
