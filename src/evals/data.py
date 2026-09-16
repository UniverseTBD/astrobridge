from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from huggingface_hub import hf_hub_download

from .config import (
    get_default_cache_dir,
    get_hf_benchmark_subpath,
    get_hf_data_repo,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    filename: str
    hf_subpath: str
    expected_ground_truth_keys: Tuple[str, ...]
    required_columns: Tuple[str, ...] = (
        "sample_id",
        "survey",
        "spectrum",
        "ground_truth",
    )


_BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "redshift": BenchmarkSpec(
        name="redshift",
        filename="redshift.parquet",
        hf_subpath=get_hf_benchmark_subpath("redshift.parquet"),
        expected_ground_truth_keys=("z", "redshift_bin"),
    ),
    "source_class": BenchmarkSpec(
        name="source_class",
        filename="source_class.parquet",
        hf_subpath=get_hf_benchmark_subpath("source_class.parquet"),
        expected_ground_truth_keys=("source_class",),
    ),
    "subclass": BenchmarkSpec(
        name="subclass",
        filename="subclass.parquet",
        hf_subpath=get_hf_benchmark_subpath("subclass.parquet"),
        expected_ground_truth_keys=("subclass",),
    ),
    "emission_lines": BenchmarkSpec(
        name="emission_lines",
        filename="emission_lines.parquet",
        hf_subpath=get_hf_benchmark_subpath("emission_lines.parquet"),
        expected_ground_truth_keys=("detected_lines", "absent_lines"),
    ),
}


class BenchmarkDataManager:
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else get_default_cache_dir()
        self.repo_id = repo_id or get_hf_data_repo()
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._all_spectra_cache: Optional[pd.DataFrame] = None
        os.makedirs(self.cache_dir, exist_ok=True)

    def resolve_spec(self, benchmark_name: str) -> BenchmarkSpec:
        key = str(benchmark_name).strip().lower()
        if key in _BENCHMARK_SPECS:
            return _BENCHMARK_SPECS[key]
        valid_names = sorted(_BENCHMARK_SPECS.keys())
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Valid benchmarks: {valid_names}")

    def get_local_path(self, spec: BenchmarkSpec) -> Path:
        return self.cache_dir / spec.filename

    def is_cached_locally(self, spec: BenchmarkSpec) -> bool:
        return self.get_local_path(spec).is_file()

    def download_benchmark_file(self, spec: BenchmarkSpec, force_download: bool = False) -> Path:
        local_path = self.get_local_path(spec)
        if local_path.is_file() and not force_download:
            return local_path

        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=spec.hf_subpath,
                repo_type="dataset",
                force_download=force_download,
            )
            shutil.copyfile(downloaded, local_path)
            return local_path
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download benchmark '{spec.name}' from Hugging Face "
                f"({self.repo_id}:{spec.hf_subpath}): {exc}"
            ) from exc

    def ensure_benchmark_file(self, spec: BenchmarkSpec) -> Path:
        return self.download_benchmark_file(spec)

    def ensure_all_files(self, force_download: bool = False) -> Dict[str, Path]:
        return {
            spec.filename: self.download_benchmark_file(spec, force_download=force_download)
            for spec in _BENCHMARK_SPECS.values()
        }

    def _validate_and_normalize(self, df: pd.DataFrame, spec: BenchmarkSpec) -> pd.DataFrame:
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
        elif "object_id" in df.columns:
            df["sample_id"] = df["object_id"].astype(str)
        else:
            raise ValueError(f"Dataset for benchmark '{spec.name}' missing identifier column ('sample_id' or 'object_id').")

        if "survey" not in df.columns:
            raise ValueError(f"Dataset for benchmark '{spec.name}' missing required 'survey' column.")
        if "spectrum" not in df.columns:
            raise ValueError(f"Dataset for benchmark '{spec.name}' missing 'spectrum' column.")
        if "ground_truth" not in df.columns:
            raise ValueError(f"Dataset for benchmark '{spec.name}' missing 'ground_truth' column.")

        if len(df) > 0 and isinstance(df["ground_truth"].iloc[0], dict):
            first_gt = df["ground_truth"].iloc[0]
            for key in spec.expected_ground_truth_keys:
                if key not in first_gt:
                    raise ValueError(f"Dataset for benchmark '{spec.name}' missing ground truth key '{key}'.")

        return df

    def load_benchmark(self, benchmark_name: str, force_reload: bool = False) -> pd.DataFrame:
        spec = self.resolve_spec(benchmark_name)
        if not force_reload and spec.name in self._df_cache:
            return self._df_cache[spec.name].copy()

        local_path = self.ensure_benchmark_file(spec)
        df = pd.read_parquet(local_path)
        normalized_df = self._validate_and_normalize(df, spec)
        self._df_cache[spec.name] = normalized_df
        return normalized_df.copy()

    def load_all_unique_spectra(self, force_reload: bool = False) -> pd.DataFrame:
        if not force_reload and self._all_spectra_cache is not None:
            return self._all_spectra_cache.copy()

        canonical_names = ["redshift", "source_class", "subclass", "emission_lines"]
        dfs: List[pd.DataFrame] = []
        for name in canonical_names:
            df = self.load_benchmark(name)
            common_cols = [c for c in ["sample_id", "survey", "spectrum", "ra", "dec", "z"] if c in df.columns]
            dfs.append(df[common_cols].copy())

        combined = pd.concat(dfs, ignore_index=True)
        dedup = combined.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
        self._all_spectra_cache = dedup
        return dedup.copy()

    def list_available_benchmarks(self) -> List[str]:
        return sorted(_BENCHMARK_SPECS.keys())


_GLOBAL_DATA_MANAGER: Optional[BenchmarkDataManager] = None


def get_default_data_manager() -> BenchmarkDataManager:
    global _GLOBAL_DATA_MANAGER
    if _GLOBAL_DATA_MANAGER is None:
        _GLOBAL_DATA_MANAGER = BenchmarkDataManager()
    return _GLOBAL_DATA_MANAGER


def load_benchmark_dataset(benchmark_name: str) -> pd.DataFrame:
    return get_default_data_manager().load_benchmark(benchmark_name)


def load_all_benchmark_spectra() -> pd.DataFrame:
    return get_default_data_manager().load_all_unique_spectra()


def ensure_all_benchmark_files(
    target_dir: Optional[Union[str, Path]] = None,
    repo_id: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, Path]:
    mgr = BenchmarkDataManager(cache_dir=target_dir, repo_id=repo_id)
    return mgr.ensure_all_files(force_download=force_download)


__all__ = [
    "BenchmarkSpec",
    "BenchmarkDataManager",
    "get_default_data_manager",
    "load_benchmark_dataset",
    "load_all_benchmark_spectra",
    "ensure_all_benchmark_files",
]
