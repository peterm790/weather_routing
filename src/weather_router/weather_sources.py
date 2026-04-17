from __future__ import annotations

from dataclasses import dataclass


DYNAMICAL_PROVIDER = "dynamical"

SUPPORTED_PROVIDERS = (DYNAMICAL_PROVIDER,)
SUPPORTED_DATASET_IDS = ("gfs", "aifs")


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
        value: str | None,
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


def _clean(value: object, field: str) -> str:
    if value is None:
        raise WeatherSourceValidationError(
            error=f"missing_{field}",
            field=field,
            value=None,
            message=f"{field} is required",
            supported_values=(
                SUPPORTED_PROVIDERS if field == "provider" else SUPPORTED_DATASET_IDS
            ),
        )

    text = str(value).strip().lower()
    if not text:
        raise WeatherSourceValidationError(
            error=f"missing_{field}",
            field=field,
            value=text,
            message=f"{field} is required",
            supported_values=(
                SUPPORTED_PROVIDERS if field == "provider" else SUPPORTED_DATASET_IDS
            ),
        )
    return text


def normalize_weather_source(
    provider: str,
    dataset_id: str,
) -> WeatherSource:
    normalized_provider = _clean(provider, "provider")
    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise WeatherSourceValidationError(
            error="invalid_provider",
            field="provider",
            value=normalized_provider,
            message=f"Unknown provider: {normalized_provider}",
            supported_values=SUPPORTED_PROVIDERS,
        )

    normalized_dataset_id = _clean(dataset_id, "dataset_id")
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
