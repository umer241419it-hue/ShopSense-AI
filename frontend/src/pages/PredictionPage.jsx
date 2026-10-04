import { useState } from "react";
import {
  RotateCcw,
  Sparkles,
  CheckCircle2,
  XCircle,
  Users,
  Layers,
  Calendar,
  ShieldCheck,
  AlertCircle,
  Clock,
  BookOpen,
} from "lucide-react";
import api from "../api";

const initial = {
  visitorType: "Returning_Visitor",
  month: "May",
  weekend: false,
  productPages: 10,
  productMinutes: 8,
  informationPages: 1,
  informationMinutes: 1,
  adminPages: 2,
  adminMinutes: 1,
  specialDay: "none",
};

const months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * Production Model Payload Formatter
 *
 * Emits strictly the 10 real-time observable session features expected by the
 * production Random Forest pipeline. Retrospective attribution metrics (PageValues,
 * BounceRates, ExitRates) and synthetic category IDs (OperatingSystems, Browser,
 * Region, TrafficType) are completely excluded from the production prediction path.
 */
function toModelPayload(form) {
  const adminPages = Math.max(0, Number(form.adminPages) || 0);
  const adminMin = Math.max(0, Number(form.adminMinutes) || 0);
  const infoPages = Math.max(0, Number(form.informationPages) || 0);
  const infoMin = Math.max(0, Number(form.informationMinutes) || 0);
  const prodPages = Math.max(0, Number(form.productPages) || 0);
  const prodMin = Math.max(0, Number(form.productMinutes) || 0);

  const specialDayMap = {
    none: 0.0,
    slight: 0.2,
    moderate: 0.4,
    close: 0.8,
    veryClose: 1.0,
  };

  return {
    Administrative: adminPages,
    Administrative_Duration: adminMin * 60,
    Informational: infoPages,
    Informational_Duration: infoMin * 60,
    ProductRelated: prodPages,
    ProductRelated_Duration: prodMin * 60,
    SpecialDay: specialDayMap[form.specialDay] ?? 0.0,
    Month: form.month,
    VisitorType: form.visitorType,
    Weekend: Boolean(form.weekend),
  };
}

export function PredictionPage() {
  const [form, setForm] = useState(initial);
  const [r, setR] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const set = (key, value) => setForm((current) => ({ ...current, [key]: value }));

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      const payload = toModelPayload(form);
      const res = await api.post("/predictions", payload);
      setR(res.data);
    } catch (e) {
      setErr(e.response?.data?.message || "Prediction request failed");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setForm(initial);
    setR(null);
    setErr("");
  }

  const totalPages =
    (Number(form.adminPages) || 0) +
    (Number(form.informationPages) || 0) +
    (Number(form.productPages) || 0);

  const totalMinutes =
    (Number(form.adminMinutes) || 0) +
    (Number(form.informationMinutes) || 0) +
    (Number(form.productMinutes) || 0);

  const intentClass =
    r?.intent_level === "High" || r?.intent_level === "Very High"
      ? "high"
      : r?.intent_level === "Medium" || r?.intent_level === "Moderate"
      ? "medium"
      : "low";

  return (
    <section className="page prediction-container">
      <div className="prediction-page-header">
        <div>
          <span className="eyebrow">AI INFERENCE ENGINE</span>
          <h1>Predict purchase intention</h1>
          <p>
            Enter real-time browsing observations to evaluate live purchase propensity using the
            ShopSense AI production model.
          </p>
        </div>
        <button className="secondary" type="button" onClick={reset}>
          <RotateCcw size={15} /> Reset Form
        </button>
      </div>

      <div className="prediction-layout">
        {/* Left Column: Behavioral Input Form */}
        <form className="prediction-form-panel" onSubmit={submit}>
          {/* Section 1: Visitor & Session */}
          <div className="form-card-section">
            <div className="section-header">
              <div className="section-icon-badge">
                <Users size={17} />
              </div>
              <h3>Visitor &amp; Session</h3>
            </div>
            <p className="section-desc">Store relationship and temporal session timing.</p>

            <div className="grid-3-col">
              <div className="form-field">
                <label htmlFor="visitorType">Visitor Type</label>
                <select
                  id="visitorType"
                  className="form-control"
                  value={form.visitorType}
                  onChange={(e) => set("visitorType", e.target.value)}
                >
                  <option value="Returning_Visitor">Returning visitor</option>
                  <option value="New_Visitor">New visitor</option>
                  <option value="Other">Other visitor</option>
                </select>
                <span className="field-hint">Customer relationship</span>
              </div>

              <div className="form-field">
                <label htmlFor="month">Session Month</label>
                <select
                  id="month"
                  className="form-control"
                  value={form.month}
                  onChange={(e) => set("month", e.target.value)}
                >
                  {months.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <span className="field-hint">Calendar seasonality</span>
              </div>

              <div className="form-field">
                <label htmlFor="weekend">Weekend Visit?</label>
                <select
                  id="weekend"
                  className="form-control"
                  value={String(form.weekend)}
                  onChange={(e) => set("weekend", e.target.value === "true")}
                >
                  <option value="false">No (Weekday)</option>
                  <option value="true">Yes (Weekend)</option>
                </select>
                <span className="field-hint">Day of the week</span>
              </div>
            </div>
          </div>

          {/* Section 2: Browsing Activity */}
          <div className="form-card-section">
            <div className="section-header">
              <div className="section-icon-badge">
                <Layers size={17} />
              </div>
              <h3>Browsing Activity</h3>
            </div>
            <p className="section-desc">Page views and dwell time recorded across key store areas.</p>

            {/* Product Pages */}
            <div className="activity-card">
              <div className="activity-card-header">
                <span className="activity-card-title">Product Catalog &amp; Items</span>
                <span className="activity-card-subtitle">Browsing products and item collections</span>
              </div>
              <div className="grid-2-col">
                <div className="form-field">
                  <label htmlFor="productPages">Product Pages Viewed</label>
                  <div className="input-addon-group">
                    <input
                      id="productPages"
                      type="number"
                      min="0"
                      max="500"
                      step="1"
                      className="form-control"
                      value={form.productPages}
                      onChange={(e) => set("productPages", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">pages</span>
                  </div>
                  <span className="field-hint">Opened product detail pages</span>
                </div>

                <div className="form-field">
                  <label htmlFor="productMinutes">Time on Product Pages</label>
                  <div className="input-addon-group">
                    <input
                      id="productMinutes"
                      type="number"
                      min="0"
                      max="833"
                      step="1"
                      className="form-control"
                      value={form.productMinutes}
                      onChange={(e) => set("productMinutes", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">min</span>
                  </div>
                  <span className="field-hint">Dwell time on product catalog</span>
                </div>
              </div>
            </div>

            {/* Information Pages */}
            <div className="activity-card">
              <div className="activity-card-header">
                <span className="activity-card-title">Information &amp; Support</span>
                <span className="activity-card-subtitle">Policies, delivery info, and customer guides</span>
              </div>
              <div className="grid-2-col">
                <div className="form-field">
                  <label htmlFor="informationPages">Info Pages Viewed</label>
                  <div className="input-addon-group">
                    <input
                      id="informationPages"
                      type="number"
                      min="0"
                      max="30"
                      step="1"
                      className="form-control"
                      value={form.informationPages}
                      onChange={(e) => set("informationPages", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">pages</span>
                  </div>
                  <span className="field-hint">FAQ &amp; service pages</span>
                </div>

                <div className="form-field">
                  <label htmlFor="informationMinutes">Time on Info Pages</label>
                  <div className="input-addon-group">
                    <input
                      id="informationMinutes"
                      type="number"
                      min="0"
                      max="50"
                      step="1"
                      className="form-control"
                      value={form.informationMinutes}
                      onChange={(e) => set("informationMinutes", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">min</span>
                  </div>
                  <span className="field-hint">Dwell time on help pages</span>
                </div>
              </div>
            </div>

            {/* Administrative Pages */}
            <div className="activity-card">
              <div className="activity-card-header">
                <span className="activity-card-title">Administrative &amp; Account</span>
                <span className="activity-card-subtitle">Customer account, profile, and login pages</span>
              </div>
              <div className="grid-2-col">
                <div className="form-field">
                  <label htmlFor="adminPages">Admin Pages Viewed</label>
                  <div className="input-addon-group">
                    <input
                      id="adminPages"
                      type="number"
                      min="0"
                      max="50"
                      step="1"
                      className="form-control"
                      value={form.adminPages}
                      onChange={(e) => set("adminPages", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">pages</span>
                  </div>
                  <span className="field-hint">Account &amp; settings views</span>
                </div>

                <div className="form-field">
                  <label htmlFor="adminMinutes">Time on Admin Pages</label>
                  <div className="input-addon-group">
                    <input
                      id="adminMinutes"
                      type="number"
                      min="0"
                      max="84"
                      step="1"
                      className="form-control"
                      value={form.adminMinutes}
                      onChange={(e) => set("adminMinutes", e.target.value)}
                      required
                    />
                    <span className="input-addon-suffix">min</span>
                  </div>
                  <span className="field-hint">Dwell time on account views</span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Special Shopping Context */}
          <div className="form-card-section">
            <div className="section-header">
              <div className="section-icon-badge">
                <Calendar size={17} />
              </div>
              <h3>Special Shopping Context</h3>
            </div>
            <p className="section-desc">Proximity of the session to major shopping holidays or sales events.</p>

            <div className="form-field">
              <label htmlFor="specialDay">Proximity to Special Day</label>
              <select
                id="specialDay"
                className="form-control"
                value={form.specialDay}
                onChange={(e) => set("specialDay", e.target.value)}
              >
                <option value="none">Not close (Standard browsing day)</option>
                <option value="slight">Slightly close (Within 4–5 days of event)</option>
                <option value="moderate">Moderately close (Within 2–3 days of event)</option>
                <option value="close">Very close (Eve of holiday / promotion)</option>
                <option value="veryClose">Holiday peak (On the special shopping day)</option>
              </select>
              <span className="field-hint">Models shopping holiday surges and promotional buying peaks</span>
            </div>
          </div>

          {/* Real-Time Session Telemetry Summary */}
          <div className="derived-indicators-card">
            <span className="derived-indicators-title">
              <Clock size={13} />
              Session Telemetry Summary
            </span>
            <div className="derived-indicators-grid">
              <div className="derived-indicator-item">
                <span className="derived-indicator-label">Total Navigation Depth</span>
                <span className="derived-indicator-value">{totalPages} pages</span>
              </div>
              <div className="derived-indicator-item">
                <span className="derived-indicator-label">Total Session Duration</span>
                <span className="derived-indicator-value">{totalMinutes} min</span>
              </div>
              <div className="derived-indicator-item">
                <span className="derived-indicator-label">Observable Features</span>
                <span className="derived-indicator-value">10 Real-Time Signals</span>
              </div>
            </div>
          </div>

          {/* Academic Transparency Note */}
          <div className="technical-callout">
            <ShieldCheck size={18} />
            <span>
              <strong>Real-Time Observable Inference:</strong> This prediction is evaluated exclusively on real-time
              behavioral signals (browsing dwell time, navigation depth, and calendar seasonality). Retrospective
              attribution metrics (PageValues, BounceRates, ExitRates) and anonymized codes are deliberately excluded.
            </span>
          </div>

          {err && (
            <div className="error-alert">
              <AlertCircle size={18} />
              <span>{err}</span>
            </div>
          )}

          <button className="btn-predict" type="submit" disabled={busy}>
            <Sparkles size={18} />
            <span>{busy ? "Analyzing Session..." : "Predict Purchase Intention"}</span>
          </button>
        </form>

        {/* Right Column: Prediction Result Panel */}
        <aside className="prediction-result-panel">
          <div className="result-panel-header">
            <h3>Prediction Outcome</h3>
            <span className="live-indicator">
              <span className="pulse-dot" />
              Production Model
            </span>
          </div>

          {!r ? (
            <div className="empty-result-state">
              <div className="empty-icon-wrapper">
                <Sparkles size={28} />
              </div>
              <h4>Ready for Assessment</h4>
              <p>
                Configure the visitor browsing attributes on the left and submit to generate real-time purchase propensity analytics.
              </p>

              <div className="empty-feature-list">
                <div className="empty-feature-item">
                  <CheckCircle2 size={16} />
                  <span>Binary Purchase vs. Non-Purchase classification</span>
                </div>
                <div className="empty-feature-item">
                  <CheckCircle2 size={16} />
                  <span>Calibrated purchase probability percentage</span>
                </div>
                <div className="empty-feature-item">
                  <CheckCircle2 size={16} />
                  <span>Intent level categorization (High / Moderate / Low)</span>
                </div>
                <div className="empty-feature-item">
                  <CheckCircle2 size={16} />
                  <span>Real-time observable session signals only</span>
                </div>
              </div>
            </div>
          ) : (
            <div>
              {/* Decision Hero Card */}
              <div className={`result-hero-card ${r.prediction === 1 ? "success" : "neutral"}`}>
                <div className="result-hero-icon">
                  {r.prediction === 1 ? <CheckCircle2 size={26} /> : <XCircle size={26} />}
                </div>
                <div className="result-hero-meta">
                  <span className="result-hero-label">
                    {r.prediction === 1 ? "Positive Intent" : "Standard Session"}
                  </span>
                  <strong className="result-hero-title">
                    {r.prediction_label || (r.prediction === 1 ? "Purchase" : "No Purchase")}
                  </strong>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="result-metrics-grid">
                <div className="result-metric-card">
                  <span className="result-metric-title">Purchase Probability</span>
                  <strong className="result-metric-val">{(r.purchase_probability * 100).toFixed(1)}%</strong>
                  <span className="result-metric-sub">Likelihood to complete order</span>
                </div>

                <div className="result-metric-card">
                  <span className="result-metric-title">Model Confidence</span>
                  <strong className="result-metric-val">{(r.confidence * 100).toFixed(1)}%</strong>
                  <span className="result-metric-sub">Classifier certainty</span>
                </div>
              </div>

              {/* Intent Level Card */}
              <div className="intent-card">
                <div className="intent-header">
                  <span className="intent-header-label">Assessed Intent Level</span>
                  <span className={`intent-badge ${intentClass}`}>
                    {r.intent_level || (r.prediction === 1 ? "High Intent" : "Low Intent")}
                  </span>
                </div>

                <div className="probability-bar-track">
                  <div
                    className={`probability-bar-fill ${intentClass}`}
                    style={{ width: `${Math.min(100, Math.max(0, r.purchase_probability * 100))}%` }}
                  />
                </div>

                <div className="prob-split-labels">
                  <span>Purchase: {(r.purchase_probability * 100).toFixed(1)}%</span>
                  <span>
                    No Purchase:{" "}
                    {(
                      (r.no_purchase_probability != null
                        ? r.no_purchase_probability
                        : 1 - r.purchase_probability) * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>

              <div className="result-explanation-note">
                Statistical propensity estimate produced by the ShopSense production model based solely on real-time navigation depth and dwell time.
              </div>

              <button
                type="button"
                className="secondary"
                style={{ width: "100%", justifyContent: "center" }}
                onClick={reset}
              >
                <RotateCcw size={15} /> Reset &amp; Test Another
              </button>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
