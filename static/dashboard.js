const form = document.querySelector("#forecast-form");
const tokenInput = document.querySelector("#api-token");
const daysInput = document.querySelector("#days");
const daysValue = document.querySelector("#days-value");
const button = document.querySelector("#forecast-button");
const buttonLabel = document.querySelector(".button-label");
const notice = document.querySelector("#notice");
const chart = document.querySelector("#forecast-chart");
const emptyState = document.querySelector("#chart-empty");
const forecastValues = document.querySelector("#forecast-values");
const forecastValueList = document.querySelector("#forecast-value-list");
const explainButton = document.querySelector("#explain-button");
const aiEstimateButton = document.querySelector("#ai-estimate-button");
const explanation = document.querySelector("#ai-explanation");
const aiDisclaimer = document.querySelector("#ai-disclaimer");
const aiForecastLegend = document.querySelector("#ai-forecast-legend");
const aiForecastLegendLabel = document.querySelector("#ai-forecast-legend-label");
const temperatureUnitInput = document.querySelector("#temperature-unit");
const chartUnitLabel = document.querySelector("#chart-unit-label");
const apiKeyDialog = document.querySelector("#api-key-dialog");
const openAiKeyInput = document.querySelector("#openai-key");
const saveApiKeyButton = document.querySelector("#save-api-key");
const skipApiKeyButton = document.querySelector("#skip-api-key");
const apiKeySettingsButton = document.querySelector("#api-key-settings");
const weatherChatForm = document.querySelector("#weather-chat-form");
const weatherQuestionInput = document.querySelector("#weather-question");
const weatherChatButton = document.querySelector("#weather-chat-button");
const weatherChatAnswer = document.querySelector("#weather-chat-answer");
const locationSearchForm = document.querySelector("#location-search-form");
const locationSearchInput = document.querySelector("#location-search");
const locationSearchButton = document.querySelector("#location-search-button");
const locationResults = document.querySelector("#location-results");
const selectedLocationLabel = document.querySelector("#selected-location");
const heroLocation = document.querySelector("#hero-location");
const mapStatus = document.querySelector("#map-status");
const currentConditionsTime = document.querySelector("#current-conditions-time");

let apiToken = "";
let lastForecast = [];
let lastForecastInfo = null;
let aiForecast = [];
let openAiApiKey = "";
let lastRequestedDays = Number(daysInput.value);
let temperatureUnit = "C";
let locationRevision = 0;
let selectedLocation = {
  name: "Delhi, India",
  latitude: 28.6139,
  longitude: 77.209,
  timezone: "Asia/Kolkata",
};
let locationMap;
let locationMarker;

function updateDaysValue() {
  const days = Number(daysInput.value);
  daysValue.value = `${days} ${days === 1 ? "day" : "days"}`;
  daysValue.textContent = daysValue.value;
}

const svgNamespace = "http://www.w3.org/2000/svg";
const chartColors = {
  grid: "#e9eef3",
  label: "#91a0ae",
  forecast: "#ed9a4a",
  aiForecast: "#477fb5",
};

function setStatus(id, text, state) {
  const element = document.querySelector(`#${id}`);
  element.textContent = text;
  element.className = `status-value ${state}`;
}

function setNotice(message = "", success = false) {
  notice.textContent = message;
  notice.classList.toggle("success", success);
}

function locationPayload() {
  return {
    latitude: selectedLocation.latitude,
    longitude: selectedLocation.longitude,
    location: selectedLocation.name,
    timezone: selectedLocation.timezone,
  };
}

function clearForecastResults() {
  locationRevision += 1;
  lastForecast = [];
  lastForecastInfo = null;
  aiForecast = [];
  chart.setAttribute("hidden", "");
  emptyState.hidden = false;
  forecastValues.hidden = true;
  explainButton.disabled = true;
  explainButton.textContent = "Explain forecast";
  aiEstimateButton.disabled = false;
  aiEstimateButton.textContent = "Add AI estimate";
  weatherChatButton.disabled = true;
  weatherChatButton.textContent = "Ask AI";
  aiForecastLegend.hidden = true;
  aiDisclaimer.hidden = true;
  explanation.hidden = true;
  weatherChatAnswer.hidden = true;
  document.querySelector("#model-name").textContent =
    `Open-Meteo — awaiting forecast for ${selectedLocation.name}`;
  document.querySelector("#model-dot").classList.remove("ready");
  document.querySelector("#forecast-days").textContent = "—";
  document.querySelector("#forecast-average").textContent = "—";
  document.querySelector("#forecast-range").textContent = "—";
  document.querySelector("#training-range").textContent =
    "Generate a forecast to view data for this location.";
  for (const id of [
    "current-temperature",
    "current-apparent-temperature",
    "current-description",
    "current-humidity",
    "current-precipitation",
    "current-wind",
  ]) {
    document.querySelector(`#${id}`).textContent = "—";
  }
  currentConditionsTime.textContent =
    "Generate a forecast to load current conditions.";
  setStatus("model-status", "Waiting for forecast", "pending");
  setNotice("");
}

function selectLocation(location, zoom = 9) {
  selectedLocation = location;
  const coordinateLabel = `${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}`;
  selectedLocationLabel.textContent = location.name;
  heroLocation.textContent = `Current weather forecast · ${location.name}`;
  locationSearchInput.value = location.name;
  locationResults.replaceChildren();
  locationResults.hidden = true;
  if (locationMap && locationMarker) {
    locationMarker.setLatLng([location.latitude, location.longitude]);
    locationMap.setView([location.latitude, location.longitude], zoom);
  }
  if (location.name.startsWith("Map pin")) {
    selectedLocationLabel.textContent = `Map point · ${coordinateLabel}`;
    heroLocation.textContent = `Current weather forecast · Map point ${coordinateLabel}`;
  }
  clearForecastResults();
}

function initializeLocationMap() {
  if (!window.L) {
    mapStatus.textContent =
      "The map could not load. You can still search for a city or region above.";
    return;
  }

  locationMap = window.L.map("location-map").setView(
    [selectedLocation.latitude, selectedLocation.longitude],
    5,
  );
  window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(locationMap);
  locationMarker = window.L.marker([
    selectedLocation.latitude,
    selectedLocation.longitude,
  ]).addTo(locationMap);
  locationMarker.bindPopup("Selected forecast location");
  locationMap.on("click", (event) => {
    const { lat, lng } = event.latlng;
    selectLocation(
      {
        name: `Map pin (${lat.toFixed(4)}, ${lng.toFixed(4)})`,
        latitude: lat,
        longitude: lng,
        timezone: "auto",
      },
      Math.max(locationMap.getZoom(), 8),
    );
  });
  mapStatus.textContent = "Map ready. Click anywhere to choose a forecast point.";
}

async function apiRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.headers || {}),
      Authorization: `Bearer ${apiToken}`,
    },
  });

  if (response.status === 401) {
    apiToken = "";
    setStatus("auth-status", "Token rejected", "bad");
    throw new Error("Authentication failed. Check the API token and try again.");
  }
  if (response.status === 429) {
    throw new Error("The API rate limit was reached. Wait a minute, then retry.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `The API returned HTTP ${response.status}.`);
  }

  return response.json();
}

locationSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = locationSearchInput.value.trim();
  if (!query) {
    mapStatus.textContent = "Enter a city, town, or region to search.";
    locationSearchInput.focus();
    return;
  }
  apiToken = tokenInput.value.trim();
  if (apiToken.length < 32) {
    mapStatus.textContent = "Enter the API access token before searching for places.";
    tokenInput.focus();
    return;
  }

  locationSearchButton.disabled = true;
  locationSearchButton.textContent = "Searching…";
  mapStatus.textContent = "";
  locationResults.replaceChildren();
  locationResults.hidden = true;
  try {
    const results = await apiRequest(
      `/locations/search?q=${encodeURIComponent(query)}`,
    );
    if (!results.length) {
      mapStatus.textContent = `No places found for “${query}”. Try a nearby city or region.`;
      return;
    }
    for (const result of results) {
      const parts = [result.name, result.admin1, result.country].filter(Boolean);
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("role", "option");
      button.textContent = [...new Set(parts)].join(", ");
      button.addEventListener("click", () => {
        selectLocation(
          {
            name: [...new Set(parts)].join(", "),
            latitude: result.latitude,
            longitude: result.longitude,
            timezone: result.timezone || "auto",
          },
          10,
        );
        mapStatus.textContent = "Place selected. Generate a forecast to view its temperatures.";
      });
      const item = document.createElement("li");
      item.setAttribute("role", "presentation");
      item.append(button);
      locationResults.append(item);
    }
    locationResults.hidden = false;
    mapStatus.textContent = `${results.length} matching place(s). Choose one to set the forecast location.`;
  } catch (error) {
    mapStatus.textContent = error.message || "Could not search for this location.";
  } finally {
    locationSearchButton.disabled = false;
    locationSearchButton.textContent = "Search places";
  }
});

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function formatDate(value, includeYear = false) {
  const date = new Date(`${value}T00:00:00`);
  return new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    ...(includeYear ? { year: "numeric" } : {}),
  }).format(date);
}

function displayTemperature(celsius) {
  return temperatureUnit === "F" ? (celsius * 9) / 5 + 32 : celsius;
}

function formatTemperature(celsius) {
  return `${displayTemperature(celsius).toFixed(1)} °${temperatureUnit}`;
}

function chartTickStep(span) {
  const roughStep = span / 4;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalizedStep = roughStep / magnitude;
  const niceStep = normalizedStep <= 1
    ? 1
    : normalizedStep <= 2
      ? 2
      : normalizedStep <= 5
        ? 5
        : 10;
  return niceStep * magnitude;
}

function renderChart(forecast, experimentalForecast = []) {
  if (!forecast.length) {
    throw new Error("The weather provider returned no forecast values.");
  }
  const width = Math.max(chart.parentElement.clientWidth, 320);
  const height = chart.clientHeight || 285;
  const margin = { top: 15, right: 15, bottom: 40, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const forecastValues = forecast.map((point) => displayTemperature(point.meantemp));
  const experimentalValues = experimentalForecast.map((point) =>
    displayTemperature(point.meantemp),
  );
  const allValues = [...forecastValues, ...experimentalValues];
  const span = Math.max(...allValues) - Math.min(...allValues);
  const padding = Math.max(span * 0.2, temperatureUnit === "F" ? 2 : 1);
  const tickStep = chartTickStep(span + padding * 2);
  const minValue =
    Math.floor((Math.min(...allValues) - padding) / tickStep) * tickStep;
  const maxValue =
    Math.ceil((Math.max(...allValues) + padding) / tickStep) * tickStep;
  const totalPoints = Math.max(forecast.length, experimentalForecast.length);
  const x = (index) => margin.left + (index / Math.max(totalPoints - 1, 1)) * plotWidth;
  const y = (value) => margin.top + ((maxValue - value) / (maxValue - minValue)) * plotHeight;

  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  chart.replaceChildren();
  chart.setAttribute(
    "aria-label",
    `${experimentalForecast.length ? "Open-Meteo and experimental AI estimates" : "Open-Meteo daily mean temperature forecast"} for ${selectedLocation.name} in degrees ${temperatureUnit === "C" ? "Celsius" : "Fahrenheit"} from ${forecast[0].date} through ${forecast.at(-1).date}.`,
  );
  chartUnitLabel.textContent = `°${temperatureUnit}`;
  aiForecastLegend.hidden = experimentalForecast.length === 0;

  for (
    let value = minValue;
    value <= maxValue + tickStep * 0.001;
    value += tickStep
  ) {
    const tickY = y(value);
    chart.append(
      svgElement("line", {
        x1: margin.left,
        x2: width - margin.right,
        y1: tickY,
        y2: tickY,
        stroke: chartColors.grid,
        "stroke-width": 1,
      }),
    );
    const label = svgElement("text", {
      x: margin.left - 9,
      y: tickY + 4,
      fill: chartColors.label,
      "font-size": 10,
      "text-anchor": "end",
    });
    label.textContent = `${value.toFixed(1)}°${temperatureUnit}`;
    chart.append(label);
  }

  const forecastPoints = forecast.map(
    (point, index) => `${x(index)},${y(displayTemperature(point.meantemp))}`,
  );
  const fullHorizonAiComparison =
    experimentalForecast.length > 0 &&
    experimentalForecast.length <= forecast.length;
  chart.append(
    svgElement("polyline", {
      points: forecastPoints.join(" "),
      fill: "none",
      stroke: chartColors.forecast,
      "stroke-width": 2.8,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }),
  );
  if (experimentalForecast.length) {
    chart.append(
      svgElement("polyline", {
        points: experimentalForecast
          .map((point, index) => ({ point, index }))
          .filter(({ index }) =>
            fullHorizonAiComparison
              ? true
              : index >= Math.max(0, forecast.length - 1),
          )
          .map(({ point, index }) => `${x(index)},${y(displayTemperature(point.meantemp))}`)
          .join(" "),
        fill: "none",
        stroke: chartColors.aiForecast,
        "stroke-width": 2.8,
        "stroke-dasharray": "6 4",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      }),
    );
  }
  forecast.forEach((point, index) => {
    chart.append(
      svgElement("circle", {
        cx: x(index),
        cy: y(displayTemperature(point.meantemp)),
        r: 3.5,
        fill: chartColors.forecast,
        stroke: "#fff",
        "stroke-width": 1.5,
      }),
    );
  });
  experimentalForecast.forEach((point, index) => {
    if (!fullHorizonAiComparison && index < forecast.length) return;
    chart.append(
      svgElement("circle", {
        cx: x(index),
        cy: y(displayTemperature(point.meantemp)),
        r: 3.5,
        fill: chartColors.aiForecast,
        stroke: "#fff",
        "stroke-width": 1.5,
      }),
    );
  });

  const labels = [
    { index: 0, text: formatDate(forecast[0].date, true) },
    {
      index: totalPoints - 1,
      text: formatDate(
        experimentalForecast.at(-1)?.date || forecast.at(-1).date,
        true,
      ),
    },
  ];
  for (const labelData of labels) {
    const label = svgElement("text", {
      x: x(labelData.index),
      y: height - 12,
      fill: chartColors.label,
      "font-size": 10,
      "text-anchor": labelData.index === 0 ? "start" : labelData.index === totalPoints - 1 ? "end" : "middle",
    });
    label.textContent = labelData.text;
    chart.append(label);
  }

  forecastValueList.replaceChildren(
    ...Array.from({ length: totalPoints }, (_, index) => {
      const item = document.createElement("li");
      const point = forecast[index];
      const aiPoint = experimentalForecast[index];
      const sourcedValue = point
        ? `Open-Meteo ${formatTemperature(point.meantemp)}`
        : "Open-Meteo unavailable beyond day 16";
      const aiValue = aiPoint
        ? ` · ${aiForecastLegendLabel.textContent} ${formatTemperature(aiPoint.meantemp)}`
        : "";
      item.textContent = `${formatDate(point?.date || aiPoint.date, true)} — ${sourcedValue}${aiValue}`;
      return item;
    }),
  );
  forecastValues.hidden = false;
  emptyState.hidden = true;
  chart.removeAttribute("hidden");
}

function renderForecastInfo(data) {
  const values = data.forecast.map((point) => point.meantemp);
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  document.querySelector("#model-name").textContent = `${data.source} — ${data.location}`;
  document.querySelector("#model-dot").classList.add("ready");
  document.querySelector("#forecast-days").textContent =
    `${data.forecast_days_available}/${data.requested_days}`;
  document.querySelector("#forecast-average").textContent = formatTemperature(average);
  document.querySelector("#forecast-range").textContent =
    `${formatTemperature(Math.min(...values))}–${formatTemperature(Math.max(...values))}`;
  document.querySelector("#training-range").textContent =
    `${formatDate(data.forecast[0].date, true)} – ${formatDate(data.forecast.at(-1).date, true)} · Open-Meteo available for first 16 days`;
  const current = data.current_conditions;
  document.querySelector("#current-temperature").textContent =
    formatTemperature(current.temperature_2m);
  document.querySelector("#current-apparent-temperature").textContent =
    formatTemperature(current.apparent_temperature);
  document.querySelector("#current-description").textContent =
    current.weather_description;
  document.querySelector("#current-humidity").textContent =
    `${current.relative_humidity_2m.toFixed(0)}%`;
  document.querySelector("#current-precipitation").textContent =
    `${current.precipitation.toFixed(1)} mm`;
  document.querySelector("#current-wind").textContent =
    `${current.wind_speed_10m.toFixed(1)} km/h`;
  currentConditionsTime.textContent =
    `Open-Meteo current model conditions for ${current.time} (${current.timezone}). WMO code ${current.weather_code}.`;
  setStatus("model-status", "Available", "good");
}

async function connectAndForecast(days) {
  apiToken = tokenInput.value.trim();
  if (apiToken.length < 32) {
    setStatus("auth-status", "Token required", "bad");
    setNotice("Enter the DELHI_API_TOKEN used to start the API (at least 32 characters).");
    tokenInput.focus();
    return false;
  }

  button.disabled = true;
  buttonLabel.textContent = "Connecting…";
  setNotice("");
  setStatus("auth-status", "Authenticating…", "pending");

  const requestRevision = locationRevision;
  try {
    lastRequestedDays = days;
    const result = await apiRequest("/weather-forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...locationPayload(), days }),
    });
    if (requestRevision !== locationRevision) return false;
    lastForecast = result.forecast;
    lastForecastInfo = result;
    renderForecastInfo(result);
    aiForecast = [];
    renderChart(lastForecast);
    setStatus("auth-status", "Connected", "good");
    explainButton.disabled = false;
    aiEstimateButton.disabled = false;
    weatherChatButton.disabled = false;
    aiDisclaimer.hidden = true;
    explanation.hidden = true;
    weatherChatAnswer.hidden = true;
    setNotice(
      `${result.source} provides ${lastForecast.length} day(s), through ${formatDate(lastForecast.at(-1).date, true)}.${days > lastForecast.length ? ` The remaining ${days - lastForecast.length} day(s) require an experimental AI projection.` : ""}`,
      true,
    );
    return true;
  } catch (error) {
    if (requestRevision === locationRevision) {
      setNotice(error.message || "Could not connect to the forecast API.");
    }
    return false;
  } finally {
    button.disabled = false;
    buttonLabel.textContent = "Generate forecast";
  }
}

function aiRequestBody() {
  const payload = { ...locationPayload(), days: lastRequestedDays };
  if (openAiApiKey) {
    payload.openai_api_key = openAiApiKey;
  }
  return payload;
}

aiEstimateButton.addEventListener("click", async () => {
  const days = Number(daysInput.value);
  const needsFreshForecast =
    !lastForecast.length ||
    lastRequestedDays !== days ||
    lastForecastInfo?.location !== selectedLocation.name;
  const requestRevision = locationRevision;
  aiEstimateButton.disabled = true;
  aiEstimateButton.textContent = needsFreshForecast
    ? "Loading forecast…"
    : "Generating AI estimate…";
  explanation.hidden = false;
  explanation.textContent = "";
  try {
    if (needsFreshForecast && !(await connectAndForecast(days))) {
      explanation.hidden = true;
      return;
    }
    if (requestRevision !== locationRevision) return;
    explanation.hidden = false;
    aiEstimateButton.disabled = true;
    aiEstimateButton.textContent = "Generating AI estimate…";
    const result = await apiRequest("/forecast/ai-estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(aiRequestBody()),
    });
    if (requestRevision !== locationRevision) return;
    aiForecast = result.forecast;
    aiDisclaimer.textContent = result.disclaimer;
    aiDisclaimer.hidden = false;
    aiForecastLegendLabel.textContent =
      result.source === "OpenAI experimental estimate"
        ? "AI experimental projection (unvalidated)"
        : "Baseline estimate (unvalidated)";
    renderChart(lastForecast, aiForecast);
    explanation.hidden = true;
  } catch (error) {
    if (requestRevision === locationRevision) {
      explanation.textContent = error.message || "Could not generate the AI estimate.";
    }
  } finally {
    if (requestRevision === locationRevision) {
      aiEstimateButton.disabled = false;
      aiEstimateButton.textContent = aiForecast.length
        ? "Refresh AI estimate"
        : "Add AI estimate";
    }
  }
});

explainButton.addEventListener("click", async () => {
  const requestRevision = locationRevision;
  explainButton.disabled = true;
  explainButton.textContent = "Preparing explanation…";
  explanation.hidden = false;
  explanation.textContent = "";
  try {
    const result = await apiRequest("/forecast/explanation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(aiRequestBody()),
    });
    if (requestRevision === locationRevision) {
      explanation.textContent = result.explanation;
    }
  } catch (error) {
    if (requestRevision === locationRevision) {
      explanation.textContent = error.message || "Could not generate an explanation.";
    }
  } finally {
    if (requestRevision === locationRevision) {
      explainButton.disabled = false;
      explainButton.textContent = "Explain forecast";
    }
  }
});

weatherChatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = weatherQuestionInput.value.trim();
  if (!lastForecast.length || !question) return;

  const requestRevision = locationRevision;
  weatherChatButton.disabled = true;
  weatherChatButton.textContent = "Thinking…";
  weatherChatAnswer.hidden = false;
  weatherChatAnswer.textContent = "Preparing a detailed answer…";
  try {
    const result = await apiRequest("/forecast/question", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...aiRequestBody(), question }),
    });
    if (requestRevision === locationRevision) {
      weatherChatAnswer.textContent = result.answer;
    }
  } catch (error) {
    if (requestRevision === locationRevision) {
      weatherChatAnswer.textContent = error.message || "Could not answer the question.";
    }
  } finally {
    if (requestRevision === locationRevision) {
      weatherChatButton.disabled = false;
      weatherChatButton.textContent = "Ask AI";
    }
  }
});

function showApiKeyDialog() {
  openAiKeyInput.value = "";
  if (!apiKeyDialog.open) {
    apiKeyDialog.showModal();
  }
}

saveApiKeyButton.addEventListener("click", () => {
  openAiApiKey = openAiKeyInput.value.trim();
  openAiKeyInput.value = "";
  aiForecast = [];
  if (lastForecast.length) renderChart(lastForecast);
  apiKeyDialog.close();
});

skipApiKeyButton.addEventListener("click", () => {
  openAiApiKey = "";
  openAiKeyInput.value = "";
  apiKeyDialog.close();
});

apiKeyDialog.addEventListener("close", () => {
  openAiKeyInput.value = "";
});

apiKeySettingsButton.addEventListener("click", showApiKeyDialog);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  connectAndForecast(Number(daysInput.value));
});

daysInput.addEventListener("input", () => {
  updateDaysValue();
  if (lastForecast.length && Number(daysInput.value) !== lastRequestedDays) {
    clearForecastResults();
    setNotice(
      "Forecast horizon changed. Generate a new forecast before adding the AI estimate.",
    );
  }
});
initializeLocationMap();

temperatureUnitInput.addEventListener("change", () => {
  temperatureUnit = temperatureUnitInput.value;
  chartUnitLabel.textContent = `°${temperatureUnit}`;
  if (lastForecast.length) {
    renderChart(lastForecast, aiForecast);
    renderForecastInfo(lastForecastInfo);
  }
});

tokenInput.addEventListener("input", () => {
  if (apiToken && tokenInput.value.trim() !== apiToken) {
    apiToken = "";
    openAiApiKey = "";
    clearForecastResults();
    setStatus("auth-status", "Not connected", "pending");
    setStatus("model-status", "Waiting for API", "pending");
  }
});

window.addEventListener("resize", () => {
  if (lastForecast.length) {
    renderChart(lastForecast, aiForecast);
  }
});

async function checkApiHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("API unavailable");
    const result = await response.json();
    setStatus("api-status", result.status === "healthy" ? "Healthy" : "Unavailable",
      result.status === "healthy" ? "good" : "bad");
  } catch {
    setStatus("api-status", "Unavailable", "bad");
  }
}

checkApiHealth();
updateDaysValue();
showApiKeyDialog();
