const symbolSelect = document.querySelector("#symbolSelect");
const refreshButton = document.querySelector("#refreshButton");
const statusText = document.querySelector("#statusText");
const priceCanvas = document.querySelector("#priceChart");
const priceContext = priceCanvas.getContext("2d");
let activeDashboard = null;

const formatCurrency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2
});

const formatNumber = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 4
});

function setStatus(message) {
  statusText.textContent = message;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function valueOrDash(value, formatter = formatNumber) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return formatter.format(value);
}

function drawPriceChart(points) {
  const rect = priceCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  priceCanvas.width = Math.max(320, Math.floor(rect.width * dpr));
  priceCanvas.height = Math.floor(280 * dpr);
  priceContext.scale(dpr, dpr);

  const width = priceCanvas.width / dpr;
  const height = priceCanvas.height / dpr;
  const padding = { top: 18, right: 22, bottom: 34, left: 54 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = points.flatMap((point) => [point.actual, point.predicted]).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  priceContext.clearRect(0, 0, width, height);
  priceContext.fillStyle = "#ffffff";
  priceContext.fillRect(0, 0, width, height);

  priceContext.strokeStyle = "#dce3dc";
  priceContext.lineWidth = 1;
  priceContext.fillStyle = "#65716c";
  priceContext.font = "12px system-ui";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight * i) / 4;
    const label = max - (range * i) / 4;
    priceContext.beginPath();
    priceContext.moveTo(padding.left, y);
    priceContext.lineTo(width - padding.right, y);
    priceContext.stroke();
    priceContext.fillText(formatCurrency.format(label), 8, y + 4);
  }

  function pointToXY(point, index) {
    const x = padding.left + (plotWidth * index) / Math.max(1, points.length - 1);
    const y = padding.top + plotHeight - ((point - min) / range) * plotHeight;
    return [x, y];
  }

  function drawLine(key, color) {
    priceContext.strokeStyle = color;
    priceContext.lineWidth = 2.5;
    priceContext.beginPath();
    points.forEach((point, index) => {
      const [x, y] = pointToXY(point[key], index);
      if (index === 0) {
        priceContext.moveTo(x, y);
      } else {
        priceContext.lineTo(x, y);
      }
    });
    priceContext.stroke();
  }

  drawLine("actual", "#0f766e");
  drawLine("predicted", "#b45309");

  const first = points[0]?.date || "";
  const last = points[points.length - 1]?.date || "";
  priceContext.fillStyle = "#65716c";
  priceContext.fillText(first, padding.left, height - 10);
  priceContext.textAlign = "right";
  priceContext.fillText(last, width - padding.right, height - 10);
  priceContext.textAlign = "left";

  priceContext.fillStyle = "#0f766e";
  priceContext.fillRect(width - 175, 18, 12, 12);
  priceContext.fillStyle = "#17211d";
  priceContext.fillText("Actual", width - 156, 28);
  priceContext.fillStyle = "#b45309";
  priceContext.fillRect(width - 94, 18, 12, 12);
  priceContext.fillStyle = "#17211d";
  priceContext.fillText("Predicted", width - 75, 28);
}

function renderDashboard(data) {
  activeDashboard = data;
  document.querySelector("#pageTitle").textContent = `${data.symbol} model dashboard`;
  document.querySelector("#sourceFile").textContent = data.sourceFile;
  document.querySelector("#dateRange").textContent = `${data.dateRange.start} to ${data.dateRange.end} • ${data.rows} rows`;
  document.querySelector("#latestClose").textContent = valueOrDash(data.summary.latestClose, formatCurrency);
  document.querySelector("#totalReturn").textContent = `${valueOrDash(data.summary.totalReturn)}%`;
  document.querySelector("#bestModel").textContent = data.summary.bestModel;
  document.querySelector("#bestAccuracy").textContent = `${valueOrDash(data.summary.bestAccuracy)}%`;
  document.querySelector("#seriesLabel").textContent = `${data.summary.bestModel} prediction`;

  document.querySelector("#metricsBody").innerHTML = data.metrics
    .map((metric, index) => `
      <tr>
        <td class="${index === 0 ? "rank" : ""}">${metric.model}</td>
        <td>${valueOrDash(metric.accuracy)}%</td>
        <td>${valueOrDash(metric.mse)}</td>
        <td class="${metric.r2 < 0 ? "negative" : ""}">${valueOrDash(metric.r2)}</td>
      </tr>
    `)
    .join("");

  document.querySelector("#recentBody").innerHTML = data.recentRows
    .map((row) => `
      <tr>
        <td>${row.date}</td>
        <td>${valueOrDash(row.open, formatCurrency)}</td>
        <td>${valueOrDash(row.high, formatCurrency)}</td>
        <td>${valueOrDash(row.low, formatCurrency)}</td>
        <td>${valueOrDash(row.close, formatCurrency)}</td>
      </tr>
    `)
    .join("");

  drawPriceChart(data.priceSeries);
}

async function loadDashboard() {
  const symbol = symbolSelect.value;
  if (!symbol) {
    return;
  }

  refreshButton.disabled = true;
  setStatus(`Evaluating ${symbol} models...`);

  try {
    const data = await fetchJson(`/api/dashboard/${symbol}`);
    renderDashboard(data);
    setStatus(`Loaded ${data.summary.modelCount} models from ${data.sourceFile}.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    refreshButton.disabled = false;
  }
}

async function init() {
  try {
    const { symbols } = await fetchJson("/api/symbols");
    symbolSelect.innerHTML = symbols.map((symbol) => `<option value="${symbol}">${symbol}</option>`).join("");
    await loadDashboard();
  } catch (error) {
    setStatus(error.message);
  }
}

refreshButton.addEventListener("click", loadDashboard);
symbolSelect.addEventListener("change", loadDashboard);
window.addEventListener("resize", () => {
  if (activeDashboard) {
    drawPriceChart(activeDashboard.priceSeries);
  }
});

init();
