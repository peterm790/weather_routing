from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_PROVIDER = "dynamical"
DEFAULT_DATASET_ID = "gfs"

SUPPORTED_PROVIDERS = (DEFAULT_PROVIDER,)
SUPPORTED_DATASET_IDS = ("gfs", "aifs")
LEGACY_DATASET_ALIASES = {
    "gfs-dynamical": "gfs",
    "ecmwf-aifs-single": "aifs",
}


@dataclass(frozen=True)
class WeatherSource:
    provider: str
    dataset_id: str

    @property
    def dataset_name(self) -> str:
        return self.dataset_id.upper()

    @property
    def cache_key(self) -> str:
        return f"{self.provider}:{self.dataset_id}"

    def metadata(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
        }


class WeatherSourceValidationError(ValueError):
    def __init__(
        self,
        *,
        error: str,
        field: str,
        value: Optional[str],
        message: str,
        supported_values: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.error = error
        self.field = field
        self.value = value
        self.supported_values = supported_values

    def to_dict(self) -> dict[str, object]:
        return {
            "error": self.error,
            "message": str(self),
            "field": self.field,
            self.field: self.value,
            f"supported_{self.field}s": list(self.supported_values),
        }


def _clean(value: Optional[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text or default


def normalize_weather_source(
    provider: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> WeatherSource:
    normalized_provider = _clean(provider, DEFAULT_PROVIDER)
    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise WeatherSourceValidationError(
            error="invalid_provider",
            field="provider",
            value=normalized_provider,
            message=f"Unknown provider: {normalized_provider}",
            supported_values=SUPPORTED_PROVIDERS,
        )

    normalized_dataset_id = _clean(dataset_id, DEFAULT_DATASET_ID)
    normalized_dataset_id = LEGACY_DATASET_ALIASES.get(
        normalized_dataset_id, normalized_dataset_id
    )
    if normalized_dataset_id not in SUPPORTED_DATASET_IDS:
        raise WeatherSourceValidationError(
            error="invalid_dataset_id",
            field="dataset_id",
            value=normalized_dataset_id,
            message=f"Unknown dataset_id: {normalized_dataset_id}",
            supported_values=SUPPORTED_DATASET_IDS,
        )

    return WeatherSource(
        provider=normalized_provider,
        dataset_id=normalized_dataset_id,
    )
