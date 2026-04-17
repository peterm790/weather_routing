import pytest

from weather_router.weather_sources import (
    DYNAMICAL_PROVIDER,
    WeatherSourceValidationError,
    normalize_weather_source,
)


def test_weather_source_accepts_canonical_gfs():
    source = normalize_weather_source(provider="dynamical", dataset_id="gfs")

    assert source.provider == DYNAMICAL_PROVIDER
    assert source.dataset_id == "gfs"
    assert source.dataset_name == "GFS"
    assert source.cache_key == "dynamical:gfs"
    assert source.metadata() == {
        "provider": "dynamical",
        "dataset_id": "gfs",
        "dataset_name": "GFS",
    }


def test_weather_source_accepts_canonical_aifs():
    source = normalize_weather_source(provider="dynamical", dataset_id="aifs")

    assert source.provider == "dynamical"
    assert source.dataset_id == "aifs"
    assert source.dataset_name == "AIFS"
    assert source.cache_key == "dynamical:aifs"


def test_weather_source_rejects_unknown_provider_with_structured_error():
    with pytest.raises(WeatherSourceValidationError) as exc_info:
        normalize_weather_source(provider="static", dataset_id="gfs")

    assert exc_info.value.to_dict() == {
        "error": "invalid_provider",
        "message": "Unknown provider: static",
        "field": "provider",
        "provider": "static",
        "supported_providers": ["dynamical"],
    }


def test_weather_source_rejects_unknown_dataset_with_structured_error():
    with pytest.raises(WeatherSourceValidationError) as exc_info:
        normalize_weather_source(provider="dynamical", dataset_id="not-real")

    assert exc_info.value.to_dict() == {
        "error": "invalid_dataset_id",
        "message": "Unknown dataset_id: not-real",
        "field": "dataset_id",
        "dataset_id": "not-real",
        "supported_dataset_ids": ["gfs", "aifs"],
    }


def test_weather_source_rejects_missing_provider_with_structured_error():
    with pytest.raises(WeatherSourceValidationError) as exc_info:
        normalize_weather_source(provider="", dataset_id="gfs")

    assert exc_info.value.to_dict() == {
        "error": "missing_provider",
        "message": "provider is required",
        "field": "provider",
        "provider": "",
        "supported_providers": ["dynamical"],
    }


def test_weather_source_rejects_missing_dataset_with_structured_error():
    with pytest.raises(WeatherSourceValidationError) as exc_info:
        normalize_weather_source(provider="dynamical", dataset_id="")

    assert exc_info.value.to_dict() == {
        "error": "missing_dataset_id",
        "message": "dataset_id is required",
        "field": "dataset_id",
        "dataset_id": "",
        "supported_dataset_ids": ["gfs", "aifs"],
    }
