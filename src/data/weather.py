import json
import math
import os
from collections.abc import Sequence
from datetime import date, timedelta
from http.client import HTTPResponse
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MAX_FORECAST_DAYS = 16
AI_FORECAST_BATCH_DAYS = 60
DEFAULT_LATITUDE = 28.6139
DEFAULT_LONGITUDE = 77.2090


class WeatherProviderError(RuntimeError):
    """Raised when the weather provider response is unavailable or invalid."""


class AIExplanationError(RuntimeError):
    """Raised when an optional OpenAI forecast explanation cannot be generated."""


class GeocodedLocation(TypedDict):
    name: str
    latitude: float
    longitude: float
    country: str
    admin1: str
    timezone: str


class CurrentConditions(TypedDict):
    time: str
    timezone: str
    temperature_2m: float
    relative_humidity_2m: float
    apparent_temperature: float
    precipitation: float
    weather_code: int
    weather_description: str
    cloud_cover: float
    wind_speed_10m: float
    wind_direction_10m: float
    source: str
    source_url: str


CURRENT_WEATHER_SOURCE = "Open-Meteo current model conditions"
OPEN_METEO_SOURCE_URL = "https://open-meteo.com/"
PROVIDER_HOSTS = frozenset(
    {
        "api.open-meteo.com",
        "geocoding-api.open-meteo.com",
        "api.openai.com",
    }
)


def _open_provider_request(request: Request, *, timeout: int) -> HTTPResponse:
    parsed_url = urlsplit(request.full_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in PROVIDER_HOSTS
        or parsed_url.port not in (None, 443)
    ):
        raise ValueError("Requests must target an approved HTTPS weather provider.")
    # Bandit cannot infer that the request URL is restricted to approved HTTPS providers.
    return urlopen(request, timeout=timeout)  # nosec B310


def search_locations(query: str) -> list[GeocodedLocation]:
    cleaned_query = query.strip()
    if not cleaned_query or len(cleaned_query) > 100:
        raise ValueError("Location search must be between 1 and 100 characters.")

    request = Request(
        f"{OPEN_METEO_GEOCODING_URL}?{urlencode({'name': cleaned_query, 'count': 8, 'language': 'en', 'format': 'json'})}",
        headers={"User-Agent": "DelhiTemperatureForecast/1.0"},
    )
    try:
        with _open_provider_request(request, timeout=10) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise WeatherProviderError("Location search is temporarily unavailable.") from error

    try:
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise TypeError("The geocoding provider returned invalid results.")
        locations: list[GeocodedLocation] = []
        for result in results:
            name = result["name"]
            latitude = result["latitude"]
            longitude = result["longitude"]
            if (
                not isinstance(name, str)
                or not name.strip()
                or isinstance(latitude, bool)
                or not isinstance(latitude, (int, float))
                or not math.isfinite(latitude)
                or not -90 <= latitude <= 90
                or isinstance(longitude, bool)
                or not isinstance(longitude, (int, float))
                or not math.isfinite(longitude)
                or not -180 <= longitude <= 180
            ):
                raise ValueError("The geocoding provider returned an invalid location.")
            locations.append(
                {
                    "name": name.strip(),
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "country": str(result.get("country", "")),
                    "admin1": str(result.get("admin1", "")),
                    "timezone": str(result.get("timezone", "auto")),
                }
            )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise WeatherProviderError("The geocoding provider returned invalid locations.") from error

    return locations


def fetch_delhi_forecast(
    days: int,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timezone: str = "Asia/Kolkata",
) -> list[tuple[date, float]]:
    if isinstance(days, bool) or not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"Forecast days must be between 1 and {MAX_FORECAST_DAYS}.")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Forecast coordinates are outside valid latitude/longitude bounds.")

    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_mean",
            "forecast_days": days,
            "timezone": timezone,
            "temperature_unit": "celsius",
        }
    )
    request = Request(
        f"{OPEN_METEO_URL}?{query}",
        headers={"User-Agent": "DelhiTemperatureForecast/1.0"},
    )
    try:
        with _open_provider_request(request, timeout=15) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise WeatherProviderError("Open-Meteo could not be reached.") from error

    try:
        daily = payload["daily"]
        dates = daily["time"]
        temperatures = daily["temperature_2m_mean"]
        if len(dates) != days or len(temperatures) != days:
            raise ValueError("The weather provider returned an incomplete forecast.")
        forecast = []
        for forecast_date, temperature in zip(dates, temperatures, strict=True):
            parsed_date = date.fromisoformat(forecast_date)
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
            ):
                raise ValueError("The weather provider returned an invalid temperature.")
            forecast.append((parsed_date, float(temperature)))
    except (KeyError, TypeError, ValueError) as error:
        raise WeatherProviderError("Open-Meteo returned an invalid forecast.") from error

    return forecast


def fetch_current_conditions(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    timezone: str = "Asia/Kolkata",
) -> CurrentConditions:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Forecast coordinates are outside valid latitude/longitude bounds.")

    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,cloud_cover,wind_speed_10m,"
                "wind_direction_10m"
            ),
            "timezone": timezone,
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
    )
    request = Request(
        f"{OPEN_METEO_URL}?{query}",
        headers={"User-Agent": "DelhiTemperatureForecast/1.0"},
    )
    try:
        with _open_provider_request(request, timeout=15) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise WeatherProviderError("Current weather conditions are unavailable.") from error

    try:
        current = payload["current"]
        current_time = current["time"]
        provider_timezone = payload["timezone"]
        weather_code = current["weather_code"]
        numeric_fields = (
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
        )
        values = {field: current[field] for field in numeric_fields}
        if not isinstance(current_time, str) or not current_time.strip():
            raise ValueError("The weather provider returned an invalid timestamp.")
        if not isinstance(provider_timezone, str) or not provider_timezone.strip():
            raise ValueError("The weather provider returned an invalid timezone.")
        if (
            isinstance(weather_code, bool)
            or not isinstance(weather_code, int)
            or not 0 <= weather_code <= 99
        ):
            raise ValueError("The weather provider returned an invalid weather code.")
        for field, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"The weather provider returned an invalid {field}.")
        humidity = values["relative_humidity_2m"]
        cloud_cover = values["cloud_cover"]
        if not 0 <= humidity <= 100 or not 0 <= cloud_cover <= 100:
            raise ValueError("The weather provider returned invalid percentages.")
    except (KeyError, TypeError, ValueError) as error:
        raise WeatherProviderError("Open-Meteo returned invalid current conditions.") from error

    weather_descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return {
        "time": current_time,
        "timezone": provider_timezone,
        "temperature_2m": float(values["temperature_2m"]),
        "relative_humidity_2m": float(values["relative_humidity_2m"]),
        "apparent_temperature": float(values["apparent_temperature"]),
        "precipitation": float(values["precipitation"]),
        "weather_code": weather_code,
        "weather_description": weather_descriptions.get(
            weather_code, f"Weather code {weather_code}"
        ),
        "cloud_cover": float(values["cloud_cover"]),
        "wind_speed_10m": float(values["wind_speed_10m"]),
        "wind_direction_10m": float(values["wind_direction_10m"]),
        "source": CURRENT_WEATHER_SOURCE,
        "source_url": OPEN_METEO_SOURCE_URL,
    }


def _format_current_conditions(
    conditions: CurrentConditions | None,
) -> str:
    if conditions is None:
        return "Current conditions: unavailable."
    return (
        "Latest available current conditions (Open-Meteo model estimate, "
        "not an independent station observation):\n"
        f"Time: {conditions['time']} {conditions['timezone']}\n"
        f"Temperature: {conditions['temperature_2m']:.1f} °C\n"
        f"Feels like: {conditions['apparent_temperature']:.1f} °C\n"
        f"Relative humidity: {conditions['relative_humidity_2m']:.0f}%\n"
        f"Conditions: {conditions['weather_description']} "
        f"(WMO code {conditions['weather_code']})\n"
        f"Precipitation: {conditions['precipitation']:.1f} mm\n"
        f"Cloud cover: {conditions['cloud_cover']:.0f}%\n"
        f"Wind: {conditions['wind_speed_10m']:.1f} km/h "
        f"from {conditions['wind_direction_10m']:.0f}°"
    )


def explain_delhi_forecast(
    forecast: Sequence[tuple[date, float]],
    api_key: str | None = None,
    location: str = "Delhi, India",
    current_conditions: CurrentConditions | None = None,
) -> str:
    effective_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not effective_api_key:
        raise AIExplanationError("OPENAI_API_KEY is not configured.")
    if not forecast:
        raise ValueError("A non-empty forecast is required.")

    forecast_lines = "\n".join(
        f"{forecast_date.isoformat()}: {temperature:.1f} °C"
        for forecast_date, temperature in forecast
    )
    current_context = _format_current_conditions(current_conditions)
    payload = json.dumps(
        {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Explain the supplied daily mean temperature forecast "
                        "in plain language. Treat the supplied values as the only "
                        "weather facts: do not change, extrapolate, or invent "
                        "temperatures, rain, or conditions. State that forecasts "
                        "are estimates and that the numbers come from Open-Meteo, "
                        "not from the language model. Current conditions, if "
                        "provided, are the latest available Open-Meteo weather-model "
                        "estimates, not independent station observations. Keep the "
                        "answer under 120 words."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Location: {location}\n"
                        f"{current_context}\n"
                        f"Forecast values:\n{forecast_lines}\n"
                        f"Source: {OPEN_METEO_SOURCE_URL}\n"
                        "Use only these supplied source facts; you have no "
                        "independent live web access."
                    ),
                },
            ],
            "temperature": 0.2,
            "max_completion_tokens": 180,
        }
    ).encode("utf-8")
    request = Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _open_provider_request(request, timeout=20) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("OpenAI returned a non-text explanation.")
        explanation = content.strip()
        if not explanation:
            raise ValueError("OpenAI returned an empty explanation.")
    except HTTPError as error:
        if error.code in (401, 403):
            message = "OpenAI rejected the server API key."
        elif error.code == 429:
            message = "OpenAI rate limit or usage quota was reached."
        else:
            try:
                error_payload = json.load(error)
                provider_error = error_payload.get("error", {})
                provider_message = provider_error.get("message")
                provider_code = provider_error.get("code")
                provider_param = provider_error.get("param")
                if isinstance(provider_message, str):
                    message = f"OpenAI request failed: {provider_message}"
                    if isinstance(provider_param, str):
                        message += f" (parameter: {provider_param})"
                    if isinstance(provider_code, str):
                        message += f" (code: {provider_code})"
                    if effective_api_key in message:
                        message = message.replace(effective_api_key, "[redacted]")
                    message = message[:500]
                else:
                    message = f"OpenAI request failed with HTTP {error.code}."
            except (json.JSONDecodeError, AttributeError, TypeError):
                message = f"OpenAI request failed with HTTP {error.code}."
        raise AIExplanationError(message) from error
    except (
        URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise AIExplanationError("OpenAI could not generate an explanation.") from error

    return explanation


def _generate_ai_forecast_batch(
    forecast: Sequence[tuple[date, float]],
    days: int,
    api_key: str | None = None,
    location: str = "Delhi, India",
    *,
    comparison: bool = False,
    context_forecast: Sequence[tuple[date, float]] | None = None,
    current_conditions: CurrentConditions | None = None,
) -> list[tuple[date, float]]:
    effective_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not effective_api_key:
        raise AIExplanationError("OPENAI_API_KEY is not configured.")
    if not forecast:
        raise ValueError("A non-empty forecast is required.")
    if isinstance(days, bool) or not 1 <= days <= 365:
        raise ValueError("AI estimate days must be between 1 and 365.")
    forecast = forecast[:days]
    if days == len(forecast):
        return list(forecast)

    context = context_forecast or forecast
    source_lines = "\n".join(
        f"{forecast_date.isoformat()}: {temperature:.1f} °C"
        for forecast_date, temperature in context
    )
    extension_start = forecast[-1][0] + timedelta(days=1)
    extension_days = days - len(forecast)
    payload = json.dumps(
        {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        (
                            "Create an experimental alternative estimate for "
                            "the requested near-term dates, using the supplied "
                            "Open-Meteo daily means as context. These estimates "
                            "are not more authoritative or validated than the "
                            "Open-Meteo values. "
                            if comparison
                            else
                            "Create a highly uncertain, experimental long-range "
                            "temperature projection from the supplied daily mean "
                            "temperature values. Only Open-Meteo values for the "
                            "first 16 forecast days are sourced; later values may "
                            "be experimental estimates generated in earlier "
                            "batches. You have no live observations or validated "
                            "long-range weather model for the later dates. Use "
                            "current conditions only as timestamped context; do "
                            "not present them as future daily observations. Do "
                            "not claim later values are reliable predictions. "
                        )
                        + "Provide "
                        "plausible broad seasonal daily mean estimates, avoid "
                        "random day-to-day noise, and keep values within -10 to "
                        "50 Celsius. Return exactly the requested number of later "
                        "daily values in a JSON object shaped as "
                        '{"forecast":[{"date":"YYYY-MM-DD","meantemp":number}]}. '
                        "Dates must be consecutive and start at the requested "
                        "extension start date."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Selected location: {location}\n"
                        f"{_format_current_conditions(current_conditions)}\n"
                        f"Current conditions source: {OPEN_METEO_SOURCE_URL}\n"
                        f"Recent temperature context:\n{source_lines}\n\n"
                        f"Generate {extension_days} experimental daily mean "
                        f"estimates beginning {extension_start.isoformat()} and "
                        f"ending {(extension_start + timedelta(days=extension_days - 1)).isoformat()}."
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": min(6000, extension_days * 30 + 300),
        }
    ).encode("utf-8")
    request = Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with _open_provider_request(request, timeout=25) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("OpenAI returned a non-text estimate.")
        parsed = json.loads(content)
        generated = parsed["forecast"]
        if not isinstance(generated, list) or len(generated) != extension_days:
            raise ValueError("OpenAI returned the wrong number of forecast days.")

        ai_forecast: list[tuple[date, float]] = list(forecast)
        expected_date = extension_start
        for item in generated:
            if not isinstance(item, dict):
                raise TypeError("OpenAI returned an invalid forecast item.")
            generated_date = date.fromisoformat(item["date"])
            temperature = item["meantemp"]
            if generated_date != expected_date:
                raise ValueError("OpenAI changed a forecast date.")
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not -10 <= temperature <= 50
            ):
                raise ValueError("OpenAI returned an invalid temperature.")
            ai_forecast.append((generated_date, float(temperature)))
            expected_date += timedelta(days=1)
    except HTTPError as error:
        if error.code in (401, 403):
            message = "OpenAI rejected the API key."
        elif error.code == 429:
            message = "OpenAI rate limit or usage quota was reached."
        else:
            try:
                error_payload = json.load(error)
                provider_error = error_payload.get("error", {})
                provider_message = provider_error.get("message")
                if isinstance(provider_message, str):
                    message = f"OpenAI request failed: {provider_message}"
                    if effective_api_key in message:
                        message = message.replace(effective_api_key, "[redacted]")
                    message = message[:500]
                else:
                    message = f"OpenAI request failed with HTTP {error.code}."
            except (json.JSONDecodeError, AttributeError, TypeError):
                message = f"OpenAI request failed with HTTP {error.code}."
        raise AIExplanationError(message) from error
    except (
        URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise AIExplanationError("OpenAI returned an invalid forecast estimate.") from error

    return ai_forecast


def generate_ai_forecast(
    forecast: Sequence[tuple[date, float]],
    days: int,
    api_key: str | None = None,
    location: str = "Delhi, India",
    current_conditions: CurrentConditions | None = None,
) -> list[tuple[date, float]]:
    if not forecast:
        raise ValueError("A non-empty forecast is required.")
    if isinstance(days, bool) or not 1 <= days <= 365:
        raise ValueError("AI estimate days must be between 1 and 365.")
    if not (api_key or os.environ.get("OPENAI_API_KEY")):
        raise AIExplanationError("OPENAI_API_KEY is not configured.")

    forecast = forecast[:days]
    if days <= len(forecast):
        seed_date = forecast[0][0] - timedelta(days=1)
        seeded_forecast = [(seed_date, forecast[0][1])]
        comparison_forecast = _generate_ai_forecast_batch(
            seeded_forecast,
            days=days + 1,
            api_key=api_key,
            location=location,
            comparison=True,
            context_forecast=forecast,
            current_conditions=current_conditions,
        )
        return comparison_forecast[1:]

    ai_forecast = list(forecast)
    while len(ai_forecast) < days:
        batch_end = min(days, len(ai_forecast) + AI_FORECAST_BATCH_DAYS)
        ai_forecast = _generate_ai_forecast_batch(
            ai_forecast,
            days=batch_end,
            api_key=api_key,
            location=location,
            current_conditions=current_conditions,
        )
    return ai_forecast


def extend_forecast_with_baseline(
    forecast: Sequence[tuple[date, float]],
    days: int,
) -> list[tuple[date, float]]:
    """Extend a sourced forecast with a constant last-value baseline."""
    if not forecast:
        raise ValueError("A non-empty forecast is required.")
    if isinstance(days, bool) or not 1 <= days <= 365:
        raise ValueError("Forecast days must be between 1 and 365.")

    baseline = list(forecast[:days])
    if not baseline:
        raise ValueError("A non-empty forecast is required.")
    last_date, last_temperature = baseline[-1]
    while len(baseline) < days:
        last_date += timedelta(days=1)
        baseline.append((last_date, last_temperature))
    return baseline


def answer_weather_question(
    question: str,
    forecast: Sequence[tuple[date, float]],
    requested_days: int,
    api_key: str | None = None,
    location: str = "Delhi, India",
    current_conditions: CurrentConditions | None = None,
) -> str:
    effective_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not effective_api_key:
        raise AIExplanationError("OPENAI_API_KEY is not configured.")
    if not question.strip():
        raise ValueError("A weather question is required.")
    if not forecast:
        raise ValueError("A weather forecast is required to answer questions.")

    forecast_lines = "\n".join(
        f"{forecast_date.isoformat()}: {temperature:.1f} °C"
        for forecast_date, temperature in forecast
    )
    payload = json.dumps(
        {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a detailed assistant for a temperature forecast. "
                        "Answer the user's question clearly "
                        "and with useful detail. The only weather facts you have "
                        "are the supplied Open-Meteo daily mean forecasts and, "
                        "when provided, timestamped current model conditions. "
                        "Do not invent current conditions, highs, "
                        "lows, rain, or values for dates beyond the supplied "
                        "forecast. Open-Meteo data is available only for the first "
                        "16 days. If the question asks about later dates or facts "
                        "not present, do not invent current conditions or future "
                        "weather facts; say what evidence is missing and explain the "
                        "uncertainty. Distinguish daily mean temperature from "
                        "daytime high and nighttime low. Do not claim to have "
                        "performed external research."
                        " The current conditions provided are Open-Meteo model "
                        "estimates, not independent station observations. You "
                        "have no independent live web access; do not claim that "
                        "you verified these conditions against another source."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Selected location: {location}\n"
                        f"{_format_current_conditions(current_conditions)}\n"
                        f"Current conditions source: {OPEN_METEO_SOURCE_URL}\n"
                        f"User-selected horizon: {requested_days} days.\n"
                        f"Available Open-Meteo daily mean forecasts:\n{forecast_lines}\n\n"
                        f"Source: {OPEN_METEO_SOURCE_URL}\n"
                        f"Question: {question.strip()}"
                    ),
                },
            ],
            "max_completion_tokens": 1200,
        }
    ).encode("utf-8")
    request = Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _open_provider_request(request, timeout=30) as response:
            result = json.load(response)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI returned an empty or non-text answer.")
        return content.strip()
    except HTTPError as error:
        if error.code in (401, 403):
            message = "OpenAI rejected the API key."
        elif error.code == 429:
            message = "OpenAI rate limit or usage quota was reached."
        else:
            message = f"OpenAI request failed with HTTP {error.code}."
        raise AIExplanationError(message) from error
    except (
        URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise AIExplanationError("OpenAI could not answer the weather question.") from error
