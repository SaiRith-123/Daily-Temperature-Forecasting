import io
import json
from datetime import date, timedelta
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from src.data import weather


def test_fetch_delhi_forecast_returns_provider_values(monkeypatch):
    payload = {
        "daily": {
            "time": ["2026-10-02", "2026-10-03"],
            "temperature_2m_mean": [28.8, 29.1],
        }
    }
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.fetch_delhi_forecast(2)

    assert result == [(date(2026, 10, 2), 28.8), (date(2026, 10, 3), 29.1)]
    assert isinstance(captured["request"], Request)
    assert "temperature_2m_mean" in captured["request"].full_url
    assert "timezone=Asia%2FKolkata" in captured["request"].full_url
    assert captured["timeout"] == 15


def test_provider_request_rejects_unapproved_urls():
    request = Request("http://api.open-meteo.com/v1/forecast")

    with pytest.raises(ValueError, match="approved HTTPS weather provider"):
        weather._open_provider_request(request, timeout=1)


def test_forecast_requests_selected_coordinates_and_timezone(monkeypatch):
    payload = {
        "daily": {
            "time": ["2026-10-02"],
            "temperature_2m_mean": [31.2],
        }
    }
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.fetch_delhi_forecast(
        1,
        latitude=19.076,
        longitude=72.8777,
        timezone="Asia/Kolkata",
    )

    query = parse_qs(urlparse(captured["request"].full_url).query)
    assert query["latitude"] == ["19.076"]
    assert query["longitude"] == ["72.8777"]
    assert query["timezone"] == ["Asia/Kolkata"]
    assert result == [(date(2026, 10, 2), 31.2)]


def test_fetch_current_conditions_requests_selected_live_model_context(monkeypatch):
    payload = {
        "timezone": "Asia/Kolkata",
        "current": {
            "time": "2026-10-03T00:15",
            "temperature_2m": 27.4,
            "relative_humidity_2m": 62,
            "apparent_temperature": 29.1,
            "precipitation": 0.2,
            "weather_code": 2,
            "cloud_cover": 35,
            "wind_speed_10m": 8.5,
            "wind_direction_10m": 245,
        },
    }
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.fetch_current_conditions(
        latitude=19.076,
        longitude=72.8777,
        timezone="Asia/Kolkata",
    )

    query = parse_qs(urlparse(captured["request"].full_url).query)
    assert query["latitude"] == ["19.076"]
    assert query["longitude"] == ["72.8777"]
    assert "temperature_2m,relative_humidity_2m,apparent_temperature" in query["current"][0]
    assert result["temperature_2m"] == 27.4
    assert result["apparent_temperature"] == 29.1
    assert result["weather_description"] == "Partly cloudy"
    assert result["time"] == "2026-10-03T00:15"
    assert "Open-Meteo" in result["source"]
    assert result["source_url"] == "https://open-meteo.com/"
    assert captured["timeout"] == 15


def test_fetch_current_conditions_rejects_invalid_provider_values(monkeypatch):
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: io.BytesIO(
            json.dumps(
                {
                    "timezone": "UTC",
                    "current": {
                        "time": "2026-10-03T00:00",
                        "temperature_2m": float("nan"),
                        "relative_humidity_2m": 50,
                        "apparent_temperature": 20,
                        "precipitation": 0,
                        "weather_code": 0,
                        "cloud_cover": 0,
                        "wind_speed_10m": 0,
                        "wind_direction_10m": 0,
                    },
                }
            ).encode()
        ),
    )

    with pytest.raises(weather.WeatherProviderError, match="invalid current conditions"):
        weather.fetch_current_conditions()


def test_search_locations_returns_open_meteo_places_and_encodes_query(monkeypatch):
    payload = {
        "results": [
            {
                "name": "Mumbai",
                "latitude": 19.076,
                "longitude": 72.8777,
                "country": "India",
                "admin1": "Maharashtra",
                "timezone": "Asia/Kolkata",
            }
        ]
    }
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.search_locations(" Mumbai ")

    query = parse_qs(urlparse(captured["request"].full_url).query)
    assert query["name"] == ["Mumbai"]
    assert result == [
        {
            "name": "Mumbai",
            "latitude": 19.076,
            "longitude": 72.8777,
            "country": "India",
            "admin1": "Maharashtra",
            "timezone": "Asia/Kolkata",
        }
    ]
    assert captured["timeout"] == 10


def test_search_locations_rejects_invalid_provider_coordinates(monkeypatch):
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: io.BytesIO(
            json.dumps(
                {
                    "results": [
                        {
                            "name": "Invalid",
                            "latitude": 95,
                            "longitude": 0,
                        }
                    ]
                }
            ).encode()
        ),
    )

    with pytest.raises(weather.WeatherProviderError, match="invalid locations"):
        weather.search_locations("Invalid")


@pytest.mark.parametrize("days", [0, 17, True])
def test_fetch_delhi_forecast_rejects_unsupported_horizons(days):
    with pytest.raises(ValueError, match="between 1 and 16"):
        weather.fetch_delhi_forecast(days)


def test_fetch_delhi_forecast_rejects_incomplete_or_non_finite_provider_data(
    monkeypatch,
):
    payload = {
        "daily": {
            "time": ["2026-10-02"],
            "temperature_2m_mean": [float("nan")],
        }
    }
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )

    with pytest.raises(weather.WeatherProviderError, match="invalid forecast"):
        weather.fetch_delhi_forecast(1)


def test_explain_delhi_forecast_requires_server_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(weather.AIExplanationError, match="not configured"):
        weather.explain_delhi_forecast([(date(2026, 10, 2), 28.8)])


def test_explain_delhi_forecast_sends_sourced_values_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-test-key")
    captured = {}
    response_body = {
        "choices": [{"message": {"content": "Delhi will be warm."}}],
    }

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(response_body).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.explain_delhi_forecast([(date(2026, 10, 2), 28.8)])

    sent_payload = json.loads(captured["request"].data)
    assert result == "Delhi will be warm."
    assert captured["request"].get_header("Authorization") == "Bearer server-side-test-key"
    assert "2026-10-02: 28.8 °C" in sent_payload["messages"][1]["content"]
    assert "do not change" in sent_payload["messages"][0]["content"]
    assert captured["timeout"] == 20


def test_explanation_includes_timestamped_current_source_context(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-test-key")
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: (
            captured.update(payload=json.loads(request.data))
            or io.BytesIO(
                json.dumps(
                    {"choices": [{"message": {"content": "Current conditions summary."}}]}
                ).encode()
            )
        ),
    )
    conditions = {
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
        "source": weather.CURRENT_WEATHER_SOURCE,
        "source_url": weather.OPEN_METEO_SOURCE_URL,
    }

    weather.explain_delhi_forecast(
        [(date(2026, 10, 3), 28.8)],
        location="Mumbai, India",
        current_conditions=conditions,
    )

    user_message = captured["payload"]["messages"][1]["content"]
    assert "27.4 °C" in user_message
    assert "2026-10-03T00:15 Asia/Kolkata" in user_message
    assert "Open-Meteo model estimate" in user_message
    assert weather.OPEN_METEO_SOURCE_URL in user_message


def test_explain_delhi_forecast_accepts_transient_key_without_environment_key(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "Forecast explanation."}}]}
            ).encode()
        )

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.explain_delhi_forecast(
        [(date(2026, 10, 2), 28.8)],
        api_key="one-request-key",
    )

    assert result == "Forecast explanation."
    assert captured["authorization"] == "Bearer one-request-key"


def test_generate_ai_forecast_validates_and_returns_separate_values(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}
    response_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "forecast": [
                                {"date": "2026-10-04", "meantemp": 29.5},
                            ]
                        }
                    )
                }
            }
        ]
    }

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(response_body).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)
    source = [(date(2026, 10, 2), 28.8), (date(2026, 10, 3), 29.1)]

    result = weather.generate_ai_forecast(
        source,
        days=3,
        api_key="tab-session-key",
        location="Mumbai, India",
    )

    request_payload = json.loads(captured["request"].data)
    assert result == [
        (date(2026, 10, 2), 28.8),
        (date(2026, 10, 3), 29.1),
        (date(2026, 10, 4), 29.5),
    ]
    assert captured["request"].get_header("Authorization") == "Bearer tab-session-key"
    assert request_payload["response_format"] == {"type": "json_object"}
    assert "experimental" in request_payload["messages"][0]["content"]
    assert "Selected location: Mumbai, India" in request_payload["messages"][1]["content"]
    assert captured["timeout"] == 25


@pytest.mark.parametrize("days", [1, 7, 16])
def test_generate_ai_forecast_creates_comparison_within_sourced_horizon(
    monkeypatch, days
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    first_date = date(2026, 10, 2)
    source_forecast = [
        (first_date + timedelta(days=offset), 28.0 + offset)
        for offset in range(16)
    ]
    current_conditions = {
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
        "source": weather.CURRENT_WEATHER_SOURCE,
        "source_url": weather.OPEN_METEO_SOURCE_URL,
    }
    captured = {}

    def fake_urlopen(request, *, timeout):
        payload = json.loads(request.data)
        captured["system"] = payload["messages"][0]["content"]
        captured["user"] = payload["messages"][1]["content"]
        start = first_date
        generated = [
            {
                "date": (start + timedelta(days=offset)).isoformat(),
                "meantemp": 27.5 + offset,
            }
            for offset in range(days)
        ]
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"forecast": generated})
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    result = weather.generate_ai_forecast(
        source_forecast,
        days=days,
        api_key="short-range-test-key",
        current_conditions=current_conditions,
    )

    assert len(result) == days
    assert result[0][0] == first_date
    assert result[-1][0] == first_date + timedelta(days=days - 1)
    assert result[0][1] == 27.5
    assert "experimental alternative estimate" in captured["system"]
    assert f"Generate {days} experimental daily mean estimates" in captured["user"]
    assert "27.4 °C" in captured["user"]
    assert weather.OPEN_METEO_SOURCE_URL in captured["user"]


def test_generate_ai_forecast_chunks_365_day_horizon(monkeypatch):
    captured_batches = []
    first_date = date(2026, 1, 1)

    def fake_urlopen(request, *, timeout):
        payload = json.loads(request.data)
        content = payload["messages"][1]["content"]
        batch_size = int(content.split("Generate ", 1)[1].split(" ", 1)[0])
        start = date.fromisoformat(content.split("beginning ", 1)[1].split(" ", 1)[0])
        captured_batches.append(batch_size)
        generated = [
            {
                "date": (start + timedelta(days=offset)).isoformat(),
                "meantemp": 25.0,
            }
            for offset in range(batch_size)
        ]
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"forecast": generated})
                            }
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)
    source_forecast = [
        (first_date + timedelta(days=offset), 25.0) for offset in range(16)
    ]

    result = weather.generate_ai_forecast(
        source_forecast,
        days=365,
        api_key="long-range-test-key",
    )

    assert len(result) == 365
    assert [forecast_date for forecast_date, _ in result] == [
        first_date + timedelta(days=offset) for offset in range(365)
    ]
    assert captured_batches == [60, 60, 60, 60, 60, 49]


def test_extend_forecast_with_baseline_fills_requested_horizon():
    source = [
        (date(2026, 10, 2) + timedelta(days=offset), 28.0 + offset)
        for offset in range(16)
    ]

    result = weather.extend_forecast_with_baseline(source, days=18)

    assert len(result) == 18
    assert result[:16] == source
    assert result[16:] == [
        (date(2026, 10, 18), 43.0),
        (date(2026, 10, 19), 43.0),
    ]


@pytest.mark.parametrize(
    "ai_result",
    [
        {"forecast": [{"date": "2026-10-04", "meantemp": 28.8}]},
        {"forecast": [{"date": "2026-10-02", "meantemp": float("nan")}]},
        {"forecast": [{"date": "2026-10-02", "meantemp": 61.0}]},
    ],
)
def test_generate_ai_forecast_rejects_invalid_model_values(monkeypatch, ai_result):
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-test-key")
    response_body = {
        "choices": [{"message": {"content": json.dumps(ai_result)}}],
    }
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(response_body).encode()),
    )

    with pytest.raises(weather.AIExplanationError, match="invalid forecast estimate"):
        weather.generate_ai_forecast([(date(2026, 10, 2), 28.8)], days=2)


def test_generate_ai_forecast_redacts_key_from_provider_error(monkeypatch):
    api_key = "sensitive-session-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    provider_error = HTTPError(
        "https://api.openai.com/v1/chat/completions",
        400,
        "Bad Request",
        hdrs=None,
        fp=io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": f"Invalid request containing {api_key}",
                    }
                }
            ).encode()
        ),
    )
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(provider_error),
    )

    with pytest.raises(weather.AIExplanationError) as error:
        weather.generate_ai_forecast(
            [(date(2026, 10, 2), 28.8)],
            days=2,
        )

    assert api_key not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_answer_weather_question_sends_question_and_limits_claims(monkeypatch):
    captured = {}
    response_body = {
        "choices": [
            {"message": {"content": "The mean temperature is forecast near 28.8 °C."}}
        ]
    }

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return io.BytesIO(json.dumps(response_body).encode())

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)
    result = weather.answer_weather_question(
        "Will it rain after day 16?",
        [(date(2026, 10, 2), 28.8)],
        requested_days=365,
        api_key="chat-test-key",
        current_conditions={
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
            "source": weather.CURRENT_WEATHER_SOURCE,
            "source_url": weather.OPEN_METEO_SOURCE_URL,
        },
    )

    payload = json.loads(captured["request"].data)
    assert result == "The mean temperature is forecast near 28.8 °C."
    assert "Will it rain after day 16?" in payload["messages"][1]["content"]
    assert "Do not invent current conditions" in payload["messages"][0]["content"]
    assert "Open-Meteo model estimate" in payload["messages"][1]["content"]
    assert "2026-10-03T00:15 Asia/Kolkata" in payload["messages"][1]["content"]
    assert weather.OPEN_METEO_SOURCE_URL in payload["messages"][1]["content"]
    assert "User-selected horizon: 365 days." in payload["messages"][1]["content"]
    assert captured["request"].get_header("Authorization") == "Bearer chat-test-key"
    assert captured["timeout"] == 30


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "rejected the server API key"),
        (403, "rejected the server API key"),
        (429, "rate limit or usage quota"),
    ],
)
def test_explain_delhi_forecast_reports_openai_auth_and_quota_errors(
    monkeypatch, status_code, message
):
    monkeypatch.setenv("OPENAI_API_KEY", "server-side-test-key")
    provider_error = HTTPError(
        "https://api.openai.com/v1/chat/completions",
        status_code,
        "Request failed",
        hdrs=None,
        fp=io.BytesIO(),
    )
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(provider_error),
    )

    with pytest.raises(weather.AIExplanationError, match=message):
        weather.explain_delhi_forecast([(date(2026, 10, 2), 28.8)])


def test_explain_delhi_forecast_surfaces_safe_openai_request_diagnostic(monkeypatch):
    api_key = "server-side-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    provider_error = HTTPError(
        "https://api.openai.com/v1/chat/completions",
        400,
        "Bad Request",
        hdrs=None,
        fp=io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": f"Invalid request for model using key {api_key}",
                        "param": "max_completion_tokens",
                        "code": "unsupported_parameter",
                    }
                }
            ).encode()
        ),
    )
    monkeypatch.setattr(
        weather,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(provider_error),
    )

    with pytest.raises(weather.AIExplanationError) as error:
        weather.explain_delhi_forecast([(date(2026, 10, 2), 28.8)])

    assert "unsupported_parameter" in str(error.value)
    assert "max_completion_tokens" in str(error.value)
    assert api_key not in str(error.value)
