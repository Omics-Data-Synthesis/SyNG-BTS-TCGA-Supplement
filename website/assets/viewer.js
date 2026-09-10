/* Shared learning-curve viewer.
 *
 * Loaded inside an <iframe> by the main page. The full figure configuration is
 * passed via URL query parameters:
 *
 *   assets/viewer.html?data=plots/<...>/data.json&classifier=XGB&metric=f1_score
 *
 * It fetches the per-configuration data.json (all classifiers x metrics) and
 * draws exactly one figure: real data on the left (red), augmented data on the
 * right (blue), with observed points (+/- 1 SD error bars), the fitted curve,
 * and its 95% confidence band.
 */
(function () {
  "use strict";

  var COLORS = { real: "#d62728", generated: "#1f77b4" };
  var PANEL_LABEL = { real: "Real", generated: "Augmented" };
  var X_DOMAINS = { real: [0.0, 0.46], generated: [0.54, 1.0] };

  var container = document.getElementById("plot");

  function fail(message) {
    container.textContent = message;
    container.classList.add("error");
  }

  var params = new URLSearchParams(window.location.search);
  var dataPath = params.get("data");
  var classifier = params.get("classifier");
  var metric = params.get("metric");
  if (!dataPath || !classifier || !metric) {
    fail("Missing query parameters: expected data, classifier and metric.");
    return;
  }

  function bandTrace(s, color, axisRefs) {
    return {
      type: "scatter",
      x: s.fit_n.concat(s.fit_n.slice().reverse()),
      y: s.fit_hi.concat(s.fit_lo.slice().reverse()),
      fill: "toself",
      fillcolor: color,
      opacity: 0.18,
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
      xaxis: axisRefs.x,
      yaxis: axisRefs.y,
    };
  }

  function fitTrace(s, color, axisRefs, label) {
    return {
      type: "scatter",
      mode: "lines",
      x: s.fit_n,
      y: s.fit_y,
      line: { color: color, dash: "dash" },
      name: "Fitted (" + label + ")",
      hovertemplate: "n=%{x:.0f}<br>fit=%{y:.3f}<extra></extra>",
      xaxis: axisRefs.x,
      yaxis: axisRefs.y,
    };
  }

  function observedTrace(s, color, axisRefs, label) {
    return {
      type: "scatter",
      mode: "markers",
      x: s.obs_n,
      y: s.obs_y,
      marker: { color: color, size: 7 },
      error_y: {
        type: "data",
        array: s.obs_sd,
        color: color,
        thickness: 1,
        visible: true,
      },
      customdata: s.obs_sd,
      name: "Observed (" + label + ")",
      hovertemplate: "n=%{x:.0f}<br>mean=%{y:.3f}<br>sd=%{customdata:.3f}<extra></extra>",
      xaxis: axisRefs.x,
      yaxis: axisRefs.y,
    };
  }

  function render(payload) {
    var clfSeries = payload.series && payload.series[classifier];
    var series = clfSeries && clfSeries[metric];
    if (!series || !series.real || !series.generated) {
      fail("No data for classifier=" + classifier + ", metric=" + metric + ".");
      return;
    }

    var axisRefs = { real: { x: "x", y: "y" }, generated: { x: "x2", y: "y2" } };
    var traces = [];
    ["real", "generated"].forEach(function (source) {
      var s = series[source];
      var color = COLORS[source];
      var refs = axisRefs[source];
      var label = PANEL_LABEL[source];
      // The confidence band is only present when the fit covariance was usable.
      if (s.fit_n && s.fit_lo) traces.push(bandTrace(s, color, refs));
      if (s.fit_n) traces.push(fitTrace(s, color, refs, label));
      traces.push(observedTrace(s, color, refs, label));
    });

    // Fixed y range shared by both panels (same limits as the reference PNGs).
    var yRange = [0.4, 1.0];

    var layout = {
      margin: { l: 58, r: 14, t: 40, b: 96 },
      showlegend: true,
      legend: { orientation: "h", yanchor: "top", y: -0.19, xanchor: "center", x: 0.5 },
      xaxis: {
        domain: X_DOMAINS.real,
        title: { text: "Sample size" },
        zeroline: false,
      },
      xaxis2: {
        domain: X_DOMAINS.generated,
        title: { text: "Sample size" },
        zeroline: false,
      },
      yaxis: { title: { text: metric }, range: yRange, zeroline: false },
      yaxis2: { range: yRange, zeroline: false },
      annotations: [
        {
          text: classifier + ": " + PANEL_LABEL.real,
          xref: "paper",
          yref: "paper",
          x: (X_DOMAINS.real[0] + X_DOMAINS.real[1]) / 2,
          y: 1.06,
          showarrow: false,
          font: { size: 14 },
        },
        {
          text: classifier + ": " + PANEL_LABEL.generated,
          xref: "paper",
          yref: "paper",
          x: (X_DOMAINS.generated[0] + X_DOMAINS.generated[1]) / 2,
          y: 1.06,
          showarrow: false,
          font: { size: 14 },
        },
      ],
    };

    Plotly.newPlot(container, traces, layout, {
      responsive: true,
      displaylogo: false,
    });
  }

  // viewer.html lives under assets/, while data paths are relative to the site root.
  fetch("../" + dataPath)
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(render)
    .catch(function (err) {
      fail("Failed to load " + dataPath + " (" + err.message + ").");
    });
})();
