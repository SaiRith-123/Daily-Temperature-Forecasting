import hmac
import logging
import os
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data.load_data import load_temperature_series
from src.data.weather import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    AIExplanationError,
    WeatherProviderError,
    answer_weather_question,
    explain_delhi_forecast,
    extend_forecast_with_baseline,
    fetch_current_conditions,
    fetch_delhi_forecast,
    generate_ai_forecast,
    search_locations,
)
from src.evaluation.evaluate import evaluate_arima
from src.models.arima import DEFAULT_ARIMA_ORDER, forecast_temperatures

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
bearer_scheme = HTTPBearer(auto_error=False)
MAX_FORECAST_DAYS = 16
app = FastAPI(title="Delhi Daily Temperature Forecast API", version="0.1.0")
app.state.limiter = limiter


def handle_rate_limit_error(
    request: Request, error: Exception
) -> JSONResponse:
    if not isinstance(error, RateLimitExceeded):
        raise error
    logger.warning("Rate limit exceeded for %s %s.", request.method, request.url.path)
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please retry later."},
    )


app.add_exception_handler(RateLimitExceeded, handle_rate_limit_error)


def require_api_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
) -> None:
    expected_token = os.environ.get("DELHI_API_TOKEN")
    if expected_token is None or len(expected_token) < 32:
        logger.error("DELHI_API_TOKEN must be configured with at least 32 characters.")
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


protected_router = APIRouter(dependencies=[Depends(require_api_token)])


class ForecastRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    days: int = Field(default=7, ge=1, le=365)


class ForecastLocationRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    latitude: float = Field(default=DEFAULT_LATITUDE, ge=-90, le=90)
    longitude: float = Field(default=DEFAULT_LONGITUDE, ge=-180, le=180)
    location: str = Field(default="Delhi, India", min_length=1, max_length=160)
    timezone: str = Field(default="Asia/Kolkata", min_length=1, max_length=64)


class WeatherForecastRequest(ForecastLocationRequest):
    model_config = ConfigDict(strict=True, extra="forbid")

    days: int = Field(default=7, ge=1, le=365)


class ForecastExplanationRequest(ForecastLocationRequest):
    model_config = ConfigDict(strict=True, extra="forbid")

    days: int = Field(default=7, ge=1, le=365)
    openai_api_key: SecretStr | None = None


class WeatherQuestionRequest(ForecastExplanationRequest):
    question: str = Field(min_length=1, max_length=1200)


class ForecastPoint(BaseModel):
    date: date
    meantemp: float


class CurrentConditionsResponse(BaseModel):
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


class WeatherForecastResponse(BaseModel):
    forecast: list[ForecastPoint]
    source: str
    location: str
    timezone: str
    requested_days: int
    forecast_days_available: int
    current_conditions: CurrentConditionsResponse


class LocationSearchResult(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: str
    admin1: str
    timezone: str


class AIForecastResponse(BaseModel):
    forecast: list[ForecastPoint]
    source: str
    disclaimer: str


class ForecastResponse(BaseModel):
    forecast: list[ForecastPoint]


class DashboardData(BaseModel):
    history: list[ForecastPoint]
    model: str
    order: list[int]
    metrics: dict[str, float]
    training_start: date
    training_end: date


def _forecast(days: int) -> ForecastResponse:
    logger.info("Forecast requested for %d day(s).", days)
    try:
        series = load_temperature_series()
        predictions = forecast_temperatures(series, steps=days)
    except (FileNotFoundError, ValueError) as error:
        logger.exception("Forecast could not be produced.")
        raise HTTPException(
            status_code=503,
            detail="Forecast is temporarily unavailable.",
        ) from error

    forecast_values = predictions.to_numpy(dtype=float)
    if len(forecast_values) != days or not np.isfinite(forecast_values).all():
        logger.error("Forecaster returned an invalid prediction payload.")
        raise HTTPException(
            status_code=503,
            detail="Forecast is temporarily unavailable.",
        )

    forecast_dates = pd.DatetimeIndex(predictions.index)
    return ForecastResponse(
        forecast=[
            ForecastPoint(date=timestamp.date(), meantemp=float(value))
            for timestamp, value in zip(forecast_dates, forecast_values, strict=True)
        ]
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@protected_router.get("/locations/search", response_model=list[LocationSearchResult])
@limiter.limit("30/minute")
def locations_search(
    request: Request, q: str = Query(min_length=1, max_length=100)
) -> list[LocationSearchResult]:
    try:
        return [
            LocationSearchResult(
                name=location["name"],
                latitude=location["latitude"],
                longitude=location["longitude"],
                country=location["country"],
                admin1=location["admin1"],
                timezone=location["timezone"],
            )
            for location in search_locations(q)
        ]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except WeatherProviderError as error:
        logger.warning("Location geocoding request failed.")
        raise HTTPException(
            status_code=502,
            detail="Location search is temporarily unavailable.",
        ) from error


@protected_router.get("/forecast")
@limiter.limit("30/minute")
def forecast(
    request: Request, steps: int = Query(default=7, ge=1, le=365)
) -> ForecastResponse:
    logger.info("GET forecast requested via %s.", request.method)
    return _forecast(steps)


@protected_router.post("/forecast", response_model=ForecastResponse)
@limiter.limit("30/minute")
def forecast_request(
    request: Request, payload: ForecastRequest
) -> ForecastResponse:
    logger.info("POST forecast requested via %s.", request.method)
    return _forecast(payload.days)


@protected_router.post("/weather-forecast", response_model=WeatherForecastResponse)
@limiter.limit("30/minute")
def weather_forecast(
    request: Request, payload: WeatherForecastRequest
) -> WeatherForecastResponse:
    logger.info(
        "Open-Meteo forecast requested for %d day(s) via %s.",
        payload.days,
        request.method,
    )
    try:
        values = fetch_delhi_forecast(
            min(payload.days, MAX_FORECAST_DAYS),
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
        current_conditions = fetch_current_conditions(
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except WeatherProviderError as error:
        logger.warning("Weather provider request failed.")
        raise HTTPException(
            status_code=502,
            detail="The weather forecast is temporarily unavailable.",
        ) from error

    return WeatherForecastResponse(
        forecast=[
            ForecastPoint(date=forecast_date, meantemp=temperature)
            for forecast_date, temperature in values
        ],
        source="Open-Meteo",
        location=payload.location,
        timezone=payload.timezone,
        requested_days=payload.days,
        forecast_days_available=len(values),
        current_conditions=CurrentConditionsResponse(**current_conditions),
    )


@protected_router.post("/forecast/explanation")
@limiter.limit("10/minute")
def forecast_explanation(
    request: Request, payload: ForecastExplanationRequest
) -> dict[str, str]:
    logger.info(
        "AI forecast explanation requested for %d day(s) via %s.",
        payload.days,
        request.method,
    )
    api_key = (
        payload.openai_api_key.get_secret_value()
        if payload.openai_api_key
        else os.environ.get("OPENAI_API_KEY")
    )
    api_key = _validate_openai_api_key(api_key)
    try:
        values = fetch_delhi_forecast(
            min(payload.days, MAX_FORECAST_DAYS),
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
        current_conditions = fetch_current_conditions(
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
    except WeatherProviderError as error:
        logger.warning("Weather provider request failed for AI explanation.")
        raise HTTPException(
            status_code=502,
            detail="The weather forecast is temporarily unavailable.",
        ) from error

    try:
        explanation = explain_delhi_forecast(
            values,
            api_key=api_key,
            location=payload.location,
            current_conditions=current_conditions,
        )
    except AIExplanationError as error:
        logger.warning("OpenAI explanation request failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error

    return {"explanation": explanation}


@protected_router.post("/forecast/ai-estimate", response_model=AIForecastResponse)
@limiter.limit("10/minute")
def ai_forecast_estimate(
    request: Request, payload: ForecastExplanationRequest
) -> AIForecastResponse:
    logger.info(
        "Experimental AI forecast requested for %d day(s) via %s.",
        payload.days,
        request.method,
    )
    api_key = (
        payload.openai_api_key.get_secret_value()
        if payload.openai_api_key
        else os.environ.get("OPENAI_API_KEY")
    )
    if api_key:
        api_key = _validate_openai_api_key(api_key)
    elif payload.days <= MAX_FORECAST_DAYS:
        api_key = _validate_openai_api_key(api_key)

    try:
        source_forecast = fetch_delhi_forecast(
            min(payload.days, MAX_FORECAST_DAYS),
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
    except WeatherProviderError as error:
        logger.warning("Weather provider request failed for AI forecast.")
        raise HTTPException(
            status_code=502,
            detail="The weather forecast is temporarily unavailable.",
        ) from error

    try:
        current_conditions = fetch_current_conditions(
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
    except WeatherProviderError as error:
        logger.warning("Current conditions unavailable for AI estimate: %s", error)
        current_conditions = None

    estimate_source = "OpenAI experimental estimate"
    disclaimer = (
        (
            "This is an experimental AI-generated comparison for dates that "
            "also have sourced Open-Meteo values. It is not more reliable or "
            "validated than the Open-Meteo forecast."
        )
        if payload.days <= len(source_forecast)
        else (
            "This estimate includes Open-Meteo values for the first 16 days. "
            "Later values are experimental AI-generated projections and are not "
            "validated weather predictions."
        )
    )
    fallback_reason = "No OpenAI key was provided." if not api_key else None
    ai_values: list[tuple[date, float]] | None = None
    if fallback_reason is None:
        try:
            ai_values = generate_ai_forecast(
                source_forecast,
                days=payload.days,
                api_key=api_key,
                location=payload.location,
                current_conditions=current_conditions,
            )
        except AIExplanationError as error:
            if payload.days <= len(source_forecast):
                logger.warning("OpenAI comparison forecast failed: %s", error)
                raise HTTPException(status_code=502, detail=str(error)) from error
            fallback_reason = str(error)

    if fallback_reason is not None:
        if payload.days <= len(source_forecast):
            raise HTTPException(
                status_code=503,
                detail="An OpenAI API key is required for AI comparisons within the sourced forecast.",
            )
        logger.warning(
            "OpenAI long-range estimate unavailable; returning a labeled baseline: %s",
            fallback_reason,
        )
        ai_values = extend_forecast_with_baseline(source_forecast, payload.days)
        estimate_source = "Constant-temperature baseline (AI unavailable)"
        disclaimer = (
            "OpenAI could not generate the long-range estimate. Values after "
            "the 16 sourced Open-Meteo days repeat the last sourced daily mean "
            "as a simple baseline; they are not AI-generated or weather predictions."
        )
    if ai_values is None:
        raise RuntimeError("AI estimate generation returned no forecast values.")

    return AIForecastResponse(
        forecast=[
            ForecastPoint(date=forecast_date, meantemp=temperature)
            for forecast_date, temperature in ai_values
        ],
        source=estimate_source,
        disclaimer=disclaimer,
    )


@protected_router.post("/forecast/question")
@limiter.limit("10/minute")
def weather_question(
    request: Request, payload: WeatherQuestionRequest
) -> dict[str, str]:
    logger.info(
        "Weather question submitted for %d-day horizon via %s.",
        payload.days,
        request.method,
    )
    api_key = (
        payload.openai_api_key.get_secret_value()
        if payload.openai_api_key
        else os.environ.get("OPENAI_API_KEY")
    )
    api_key = _validate_openai_api_key(api_key)
    try:
        values = fetch_delhi_forecast(
            min(payload.days, MAX_FORECAST_DAYS),
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
        current_conditions = fetch_current_conditions(
            latitude=payload.latitude,
            longitude=payload.longitude,
            timezone=payload.timezone,
        )
    except WeatherProviderError as error:
        logger.warning("Weather provider request failed for user question.")
        raise HTTPException(
            status_code=502,
            detail="The weather forecast is temporarily unavailable.",
        ) from error

    try:
        answer = answer_weather_question(
            payload.question,
            values,
            payload.days,
            api_key=api_key,
            location=payload.location,
            current_conditions=current_conditions,
        )
    except AIExplanationError as error:
        logger.warning("OpenAI weather question failed: %s", error)
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {"answer": answer}


def _validate_openai_api_key(api_key: str | None) -> str:
    if api_key and len(api_key) > 512:
        raise HTTPException(
            status_code=422,
            detail="The OpenAI API key must be 512 characters or fewer.",
        )
    if api_key:
        api_key = api_key.strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Enter an OpenAI API key in the dashboard or configure OPENAI_API_KEY on the server.",
        )
    return api_key


@protected_router.get("/evaluate")
@limiter.limit("30/minute")
def evaluate(
    request: Request, test_size: int = Query(default=30, ge=1, le=365)
) -> dict[str, float]:
    logger.info(
        "Evaluation requested with test_size=%d via %s.",
        test_size,
        request.method,
    )
    try:
        series = load_temperature_series()
        return evaluate_arima(series, test_size=test_size)
    except (FileNotFoundError, ValueError) as error:
        logger.exception("Evaluation could not be produced.")
        raise HTTPException(
            status_code=503,
            detail="Evaluation is temporarily unavailable.",
        ) from error


app.include_router(protected_router)


@protected_router.get("/dashboard-data", response_model=DashboardData)
@limiter.limit("30/minute")
def dashboard_data(request: Request) -> DashboardData:
    logger.info("Dashboard data requested via %s.", request.method)
    try:
        series = load_temperature_series()
        if len(series) < 2:
            raise ValueError("At least two observations are required.")
        metrics = evaluate_arima(series, test_size=min(30, len(series) - 1))
    except (FileNotFoundError, ValueError) as error:
        logger.exception("Dashboard data could not be produced.")
        raise HTTPException(
            status_code=503,
            detail="Dashboard data is temporarily unavailable.",
        ) from error

    history = series.tail(30)
    history_dates = pd.DatetimeIndex(history.index)
    history_values = history.to_numpy(dtype=float)
    return DashboardData(
        history=[
            ForecastPoint(date=timestamp.date(), meantemp=float(value))
            for timestamp, value in zip(history_dates, history_values, strict=True)
        ],
        model="ARIMA",
        order=list(DEFAULT_ARIMA_ORDER),
        metrics=metrics,
        training_start=series.index.min().date(),
        training_end=series.index.max().date(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.error(
        "Unhandled API error for %s %s.",
        request.method,
        request.url.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")
