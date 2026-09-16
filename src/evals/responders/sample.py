from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class SpectrumSample:
    sample_id: str
    wavelength: np.ndarray
    flux: np.ndarray
    mask: np.ndarray
    survey: str = "unknown"
    ivar: Optional[np.ndarray] = None

    @classmethod
    def from_row(cls, row: Any) -> SpectrumSample:
        sample_id = str(row.get("sample_id", row.get("object_id", "")))
        survey = str(row.get("survey", "unknown"))
        spec_data = row["spectrum"]

        flux = np.array(spec_data["flux"], dtype=float)
        wavelength = np.array(spec_data["lambda"], dtype=float)
        if "mask" in spec_data and spec_data["mask"] is not None:
            mask = np.array(spec_data["mask"], dtype=bool)
        else:
            mask = np.zeros_like(flux, dtype=bool)
        ivar = np.array(spec_data["ivar"], dtype=float) if "ivar" in spec_data and spec_data["ivar"] is not None else None

        return cls(
            sample_id=sample_id,
            wavelength=wavelength,
            flux=flux,
            mask=mask,
            survey=survey,
            ivar=ivar,
        )

