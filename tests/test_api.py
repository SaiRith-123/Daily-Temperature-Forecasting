import secrets

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main


@pytest.fixture
def client(monkeypatch):
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("DELHI_API_TOKEN", token)
    monkeypatch.setattr(
        main,
        "fetch_current_conditions",
        lambda **location: {
            "time": "2026-10-03T00:15",
            "timezone": location.get("timezone", "Asia/Kolkata"),
            "temperature_2m": 27.4,
            "relative_humidity_2m": 62.0,
            "apparent_temperature": 29.1,
            "precipitation": 0.2,
            "weather_code": 2,
            "weather_description": "Partly cloudy",
            "cloud_cover": 35.0,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 245.0,
            "source": "Open-Meteo current model conditions",
            "source_url": "https://open-meteo.com/",
        },
    )
    return TestClient(
        main.app,
        headers={"Authorization": f"Bearer {token}"},
    )


def test_health_returns_healthy_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_dashboard_html_is_served_from_api_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Temperature Forecast" in response.text
    assert "/dashboard.js" in response.text
    assert 'type="range"' in response.text
    assert 'max="365"' in response.text
    assert 'id="openai-key"' in response.text
    assert 'id="location-search"' in response.text
    assert 'id="location-map"' in response.text
    assert 'id="temperature-unit"' in response.text
    assert 'id="current-conditions"' in response.text
    assert 'id="ai-estimate-button" class="secondary-button" type="button">' in response.text
    assert "loads the current sourced forecast first if needed" in response.text


def test_dashboard_assets_are_served(client):
    response = client.get("/dashboard.js")

    assert response.status_code == 200
    assert "POST" in response.text
    assert "/forecast" in response.text
    assert "/forecast/question" in response.text
    assert "ai-estimate" in response.text
    assert "/locations/search" in response.text
    assert "displayTemperature" in response.text
    assert "9) / 5 + 32" in response.text
    assert "needsFreshForecast" in response.text
    assert "Loading forecast…" in response.text


def test_location_search_returns_geocoded_places(client, monkeypatch):
    captured = {}

    def fake_search(query):
        captured["query"] = query
        return [
            {
                "name": "Mumbai",
                "latitude": 19.076,
                "longitude": 72.8777,
                "country": "India",
                "admin1": "Maharashtra",
                "timezone": "Asia/Kolkata",
            }
        ]

    monkeypatch.setattr(main, "search_locations", fake_search)

    response = client.get("/locations/search", params={"q": "Mumbai"})

    assert response.status_code == 200
    assert captured["query"] == "Mumbai"
    assert response.json() == [
        {
            "name": "Mumbai",
            "latitude": 19.076,
            "longitude": 72.8777,
            "country": "India",
            "admin1": "Maharashtra",
            "timezone": "Asia/Kolkata",
        }
    ]


def test_dashboard_data_returns_real_history_model_and_metrics(client, monkeypatch):
    series = pd.Series(
        [float(value) for value in range(20, 40)],
        index=pd.date_range("2024-01-01", periods=20, name="date"),
        name="meantemp",
    )
    monkeypatch.setattr(main, "load_temperature_series", lambda: series)

    def fake_evaluate(observed, *, test_size):
        assert observed is series
        assert test_size == 19
        return {"mae": 1.25, "rmse": 1.5, "mape_percent": 4.0}

    monkeypatch.setattr(
        main,
        "evaluate_arima",
        fake_evaluate,
    )

    response = client.get("/dashboard-data")

    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == len(series)
    assert body["history"][0] == {"date": "2024-01-01", "meantemp": 20.0}
    assert body["model"] == "ARIMA"
    assert body["order"] == [1, 1, 0]
    assert body["metrics"] == {"mae": 1.25, "rmse": 1.5, "mape_percent": 4.0}
    assert body["training_start"] == "2024-01-01"
    assert body["training_end"] == "2024-01-20"


def test_forecast_requires_a_valid_bearer_token(monkeypatch):
    monkeypatch.setenv("DELHI_API_TOKEN", secrets.token_urlsafe(32))
    unauthenticated_client = TestClient(main.app)
    response = unauthenticated_client.post("/forecast", json={"days": 1})
    invalid_token_response = unauthenticated_client.post(
        "/forecast",
        json={"days": 1},
        headers={"Authorization": f"Bearer {secrets.token_urlsafe(32)}"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert invalid_token_response.status_code == 401


def test_forecast_rejects_an_unconfigured_server_token(client, monkeypatch):
    monkeypatch.delenv("DELHI_API_TOKEN")

    response = client.post("/forecast", json={"days": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "API authentication is not configured."


def test_post_forecast_returns_requested_daily_predictions(client, monkeypatch):
    observed = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.date_range("2024-01-01", periods=3, name="date"),
        name="meantemp",
    )
    expected_dates = pd.date_range("2024-01-04", periods=2, name="date")
    expected_forecast = pd.Series(
        [13.5, 14.0], index=expected_dates, name="forecast_meantemp"
    )
    monkeypatch.setattr(main, "load_temperature_series", lambda: observed)

    def forecast_temperatures(series, *, steps):
        assert series is observed
        assert steps == 2
        return expected_forecast

    monkeypatch.setattr(main, "forecast_temperatures", forecast_temperatures)

    response = client.post("/forecast", json={"days": 2})

    assert response.status_code == 200
    assert response.json() == {
        "forecast": [
            {"date": "2024-01-04", "meantemp": 13.5},
            {"date": "2024-01-05", "meantemp": 14.0},
        ]
    }


def test_weather_forecast_returns_provider_values_and_metadata(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "fetch_delhi_forecast",
        lambda days, **location: [
            (pd.Timestamp("2026-10-02").date(), 28.8),
            (pd.Timestamp("2026-10-03").date(), 29.1),
        ],
    )

    response = client.post("/weather-forecast", json={"days": 2})

    assert response.status_code == 200
    assert response.json() == {
        "forecast": [
            {"date": "2026-10-02", "meantemp": 28.8},
            {"date": "2026-10-03", "meantemp": 29.1},
        ],
        "source": "Open-Meteo",
        "location": "Delhi, India",
        "timezone": "Asia/Kolkata",
        "requested_days": 2,
        "forecast_days_available": 2,
        "current_conditions": {
            "time": "2026-10-03T00:15",
            "timezone": "Asia/Kolkata",
            "temperature_2m": 27.4,
            "relative_humidity_2m": 62.0,
            "apparent_temperature": 29.1,
            "precipitation": 0.2,
            "weather_code": 2,
            "weather_description": "Partly cloudy",
            "cloud_cover": 35.0,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 245.0,
            "source": "Open-Meteo current model conditions",
            "source_url": "https://open-meteo.com/",
        },
    }


def test_weather_forecast_uses_selected_location_coordinates(client, monkeypatch):
    captured = {}

    def fake_fetch(days, *, latitude, longitude, timezone):
        captured.update(
            days=days,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
        return [(pd.Timestamp("2026-10-02").date(), 31.2)]

    monkeypatch.setattr(main, "fetch_delhi_forecast", fake_fetch)
    response = client.post(
        "/weather-forecast",
        json={
            "days": 1,
            "latitude": 19.076,
            "longitude": 72.8777,
            "location": "Mumbai, Maharashtra, India",
            "timezone": "Asia/Kolkata",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "days": 1,
        "latitude": 19.076,
        "longitude": 72.8777,
        "timezone": "Asia/Kolkata",
    }
    assert response.json()["location"] == "Mumbai, Maharashtra, India"


def test_weather_forecast_caps_sourced_values_but_accepts_365_day_horizon(
    client, monkeypatch
):
    captured = {}

    def fake_fetch(days, **location):
        captured["days"] = days
        captured["location"] = location
        return [
            (pd.Timestamp("2026-10-02").date(), 28.8)
            for _ in range(days)
        ]

    monkeypatch.setattr(main, "fetch_delhi_forecast", fake_fetch)
    response = client.post("/weather-forecast", json={"days": 365})

    assert response.status_code == 200
    assert captured["days"] == 16
    assert response.json()["requested_days"] == 365
    assert response.json()["forecast_days_available"] == 16
    assert len(response.json()["forecast"]) == 16


def test_ai_explanation_uses_provider_forecast_and_requires_server_key(
    client, monkeypatch
):
    forecast = [(pd.Timestamp("2026-10-02").date(), 28.8)]
    observed = {}
    monkeypatch.setattr(
        main, "fetch_delhi_forecast", lambda days, **location: forecast
    )
    def explain(values, *, api_key, location, current_conditions):
        observed.setdefault("locations", []).append(location)
        observed["current_conditions"] = current_conditions
        return "Delhi's daily mean is forecast near 28.8 °C."

    monkeypatch.setattr(main, "explain_delhi_forecast", explain)

    response = client.post(
        "/forecast/explanation",
        json={
            "days": 1,
            "location": "Mumbai, Maharashtra, India",
            "latitude": 19.076,
            "longitude": 72.8777,
            "timezone": "Asia/Kolkata",
            "openai_api_key": "transient-test-key",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-test-key")
    server_key_response = client.post("/forecast/explanation", json={"days": 1})
    monkeypatch.delenv("OPENAI_API_KEY")
    no_key_response = client.post("/forecast/explanation", json={"days": 1})

    assert response.status_code == 200
    assert server_key_response.status_code == 200
    assert response.json() == {
        "explanation": "Delhi's daily mean is forecast near 28.8 °C."
    }
    assert "Mumbai, Maharashtra, India" in observed["locations"]
    assert observed["current_conditions"]["temperature_2m"] == 27.4
    assert no_key_response.status_code == 503
    assert "OpenAI API key" in no_key_response.json()["detail"]


def test_ai_forecast_returns_separate_experimental_series(client, monkeypatch):
    forecast = [
        (pd.Timestamp("2026-10-02").date(), 28.8),
        (pd.Timestamp("2026-10-03").date(), 29.1),
    ]
    ai_forecast = [
        (pd.Timestamp("2026-10-02").date(), 28.5),
        (pd.Timestamp("2026-10-03").date(), 29.4),
    ]
    monkeypatch.setattr(
        main, "fetch_delhi_forecast", lambda days, **location: forecast
    )
    captured = {}

    def fake_generate(source_forecast, *, days, api_key, location, current_conditions):
        captured["location"] = location
        captured["current_conditions"] = current_conditions
        return ai_forecast

    monkeypatch.setattr(main, "generate_ai_forecast", fake_generate)

    response = client.post(
        "/forecast/ai-estimate",
        json={
            "days": 2,
            "location": "Mumbai, Maharashtra, India",
            "latitude": 19.076,
            "longitude": 72.8777,
            "timezone": "Asia/Kolkata",
            "openai_api_key": "transient-test-key",
        },
    )

    assert response.status_code == 200
    assert captured["location"] == "Mumbai, Maharashtra, India"
    assert captured["current_conditions"]["time"] == "2026-10-03T00:15"
    assert response.json()["source"] == "OpenAI experimental estimate"
    assert "not more reliable or validated" in response.json()["disclaimer"]
    assert len(response.json()["forecast"]) == 2
    assert response.json()["forecast"] == [
        {"date": "2026-10-02", "meantemp": 28.5},
        {"date": "2026-10-03", "meantemp": 29.4},
    ]


def test_long_ai_forecast_returns_baseline_if_openai_fails(client, monkeypatch):
    start_date = pd.Timestamp("2026-10-02").date()
    source_forecast = [
        (start_date + pd.Timedelta(days=offset), 25.0 + offset / 10)
        for offset in range(16)
    ]
    monkeypatch.setattr(
        main, "fetch_delhi_forecast", lambda days, **location: source_forecast[:days]
    )
    def unavailable_current_conditions(**location):
        raise main.WeatherProviderError("Current conditions temporarily unavailable.")

    monkeypatch.setattr(main, "fetch_current_conditions", unavailable_current_conditions)

    def fail_ai_generation(*args, **kwargs):
        raise main.AIExplanationError("OpenAI could not generate valid JSON.")

    monkeypatch.setattr(main, "generate_ai_forecast", fail_ai_generation)
    response = client.post(
        "/forecast/ai-estimate",
        json={"days": 17, "openai_api_key": "transient-test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "Constant-temperature baseline (AI unavailable)"
    assert "not AI-generated" in body["disclaimer"]
    assert len(body["forecast"]) == 17
    assert body["forecast"][-1] == {
        "date": (start_date + pd.Timedelta(days=16)).isoformat(),
        "meantemp": source_forecast[-1][1],
    }


def test_long_ai_forecast_returns_365_days_without_openai_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    start_date = pd.Timestamp("2026-10-02").date()
    source_forecast = [
        (start_date + pd.Timedelta(days=offset), 25.0)
        for offset in range(16)
    ]
    monkeypatch.setattr(
        main, "fetch_delhi_forecast", lambda days, **location: source_forecast[:days]
    )
    monkeypatch.setattr(
        main,
        "generate_ai_forecast",
        lambda *args, **kwargs: pytest.fail("AI must not be called without a key."),
    )

    response = client.post("/forecast/ai-estimate", json={"days": 365})

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "Constant-temperature baseline (AI unavailable)"
    assert len(body["forecast"]) == 365
    assert body["forecast"][0]["date"] == start_date.isoformat()
    assert body["forecast"][-1]["date"] == (
        start_date + pd.Timedelta(days=364)
    ).isoformat()


def test_weather_question_uses_weather_context_and_api_key(client, monkeypatch):
    forecast = [(pd.Timestamp("2026-10-02").date(), 28.8)]
    observed = {}
    monkeypatch.setattr(
        main, "fetch_delhi_forecast", lambda days, **location: forecast
    )

    def answer(
        question, values, requested_days, *, api_key, location, current_conditions
    ):
        observed.update(
            question=question,
            values=values,
            requested_days=requested_days,
            api_key=api_key,
            location=location,
            current_conditions=current_conditions,
        )
        return "The sourced daily mean forecast is 28.8 °C."

    monkeypatch.setattr(main, "answer_weather_question", answer)
    response = client.post(
        "/forecast/question",
        json={
            "days": 365,
            "question": "Explain the forecast uncertainty in detail.",
            "openai_api_key": "transient-test-key",
            "location": "Mumbai, Maharashtra, India",
            "latitude": 19.076,
            "longitude": 72.8777,
            "timezone": "Asia/Kolkata",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The sourced daily mean forecast is 28.8 °C."
    }
    assert observed == {
        "question": "Explain the forecast uncertainty in detail.",
        "values": forecast,
        "requested_days": 365,
        "api_key": "transient-test-key",
        "location": "Mumbai, Maharashtra, India",
        "current_conditions": {
            "time": "2026-10-03T00:15",
            "timezone": "Asia/Kolkata",
            "temperature_2m": 27.4,
            "relative_humidity_2m": 62.0,
            "apparent_temperature": 29.1,
            "precipitation": 0.2,
            "weather_code": 2,
            "weather_description": "Partly cloudy",
            "cloud_cover": 35.0,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 245.0,
            "source": "Open-Meteo current model conditions",
            "source_url": "https://open-meteo.com/",
        },
    }


@pytest.mark.parametrize("days", [0, 366])
def test_post_forecast_rejects_days_outside_supported_range(client, days):
    response = client.post("/forecast", json={"days": days})

    assert response.status_code == 422


def test_post_forecast_rejects_extra_fields_and_non_integer_days(client):
    extra_field_response = client.post(
        "/forecast", json={"days": 1, "token": secrets.token_urlsafe(32)}
    )
    string_days_response = client.post("/forecast", json={"days": "7"})

    assert extra_field_response.status_code == 422
    assert string_days_response.status_code == 422


def test_forecast_rejects_non_finite_model_output(client, monkeypatch):
    observed = pd.Series(
        [10.0, 11.0],
        index=pd.date_range("2024-01-01", periods=2, name="date"),
        name="meantemp",
    )
    invalid_forecast = pd.Series(
        [float("nan")],
        index=pd.date_range("2024-01-03", periods=1, name="date"),
    )
    monkeypatch.setattr(main, "load_temperature_series", lambda: observed)

    def invalid_forecast_for_series(series, *, steps):
        assert not series.empty
        assert steps == 1
        return invalid_forecast

    monkeypatch.setattr(main, "forecast_temperatures", invalid_forecast_for_series)

    response = client.post("/forecast", json={"days": 1})

    assert response.status_code == 503
    assert response.json()["detail"] == "Forecast is temporarily unavailable."


def test_post_forecast_reports_missing_dataset(client, monkeypatch):
    def missing_dataset():
        raise FileNotFoundError("Temperature dataset not found.")

    monkeypatch.setattr(main, "load_temperature_series", missing_dataset)

    response = client.post("/forecast", json={"days": 7})

    assert response.status_code == 503
    assert response.json()["detail"] == "Forecast is temporarily unavailable."


def test_unexpected_forecast_errors_are_logged_and_sanitized(client, monkeypatch):
    observed = pd.Series(
        [10.0, 11.0],
        index=pd.date_range("2024-01-01", periods=2, name="date"),
        name="meantemp",
    )
    monkeypatch.setattr(main, "load_temperature_series", lambda: observed)

    def failed_forecast(series, *, steps):
        assert not series.empty
        assert steps == 1
        raise RuntimeError("private internal details")

    monkeypatch.setattr(main, "forecast_temperatures", failed_forecast)
    error_client = TestClient(
        main.app,
        raise_server_exceptions=False,
        headers=client.headers,
    )
    response = error_client.post("/forecast", json={"days": 1})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "private internal details" not in response.text


def test_get_forecast_remains_available(client, monkeypatch):
    dates = pd.date_range("2024-01-02", periods=1, name="date")
    predictions = pd.Series([20.0], index=dates)
    monkeypatch.setattr(
        main,
        "load_temperature_series",
        lambda: pd.Series(
            [19.0],
            index=pd.date_range("2024-01-01", periods=1, name="date"),
            name="meantemp",
        ),
    )

    def fake_forecast(series, *, steps):
        assert not series.empty
        return predictions.iloc[:steps]

    monkeypatch.setattr(main, "forecast_temperatures", fake_forecast)

    response = client.get("/forecast?steps=1")

    assert response.status_code == 200
    assert response.json()["forecast"] == [
        {"date": "2024-01-02", "meantemp": 20.0}
    ]


def test_forecast_is_rate_limited(client, monkeypatch):
    observed = pd.Series(
        [10.0, 11.0],
        index=pd.date_range("2024-01-01", periods=2, name="date"),
        name="meantemp",
    )
    monkeypatch.setattr(main, "load_temperature_series", lambda: observed)

    def fake_forecast(series, *, steps):
        assert not series.empty
        return pd.Series(
            [12.0],
            index=pd.date_range("2024-01-03", periods=steps, name="date"),
        )

    monkeypatch.setattr(main, "forecast_temperatures", fake_forecast)

    responses = [
        client.post("/forecast", json={"days": 1})
        for _ in range(31)
    ]

    assert any(response.status_code == 429 for response in responses)
