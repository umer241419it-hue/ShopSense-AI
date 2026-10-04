import { useState } from "react";
import { RotateCcw, Sparkles, CheckCircle2, Info } from "lucide-react";
import api from "../api";

const initial = {
  visitorType: "Returning_Visitor",
  productPages: 10,
  productMinutes: 8,
  informationPages: 1,
  informationMinutes: 1,
  adminPages: 2,
  adminMinutes: 1,
  bounceRate: 5,
  exitRate: 8,
  pageValue: 0,
  specialDay: "none",
  month: "May",
  weekend: false,
};

const months = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function toModelPayload(form) {
  const specialDayMap = {
    none: 0,
    slight: 0.2,
    moderate: 0.4,
    close: 0.8,
    veryClose: 1,
  };

  return {
    Administrative: Number(form.adminPages),
    Administrative_Duration: Number(form.adminMinutes) * 60,
    Informational: Number(form.informationPages),
    Informational_Duration: Number(form.informationMinutes) * 60,
    ProductRelated: Number(form.productPages),
    ProductRelated_Duration: Number(form.productMinutes) * 60,
    BounceRates: Number(form.bounceRate) / 100,
    ExitRates: Number(form.exitRate) / 100,
    PageValues: Number(form.pageValue),
    SpecialDay: specialDayMap[form.specialDay],
    Month: form.month,
    // These dataset-coded technical attributes are intentionally handled
    // behind the scenes so users never need to understand arbitrary codes.
    OperatingSystems: 2,
    Browser: 2,
    Region: 1,
    TrafficType: 2,
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
      setR((await api.post("/predictions", payload)).data);
    } catch (e) {
      setErr(e.response?.data?.message || "Prediction failed");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setForm(initial);
    setR(null);
    setErr("");
  }

  return (
    <section className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">PURCHASE INTENTION</span>
          <h1>Predict purchase intention</h1>
          <p>Describe the visitor's browsing session and let ShopSense AI estimate purchase intent.</p>
        </div>
        <button className="secondary" type="button" onClick={reset}>
          <RotateCcw size={16} /> Reset
        </button>
      </div>

      <div className="grid-2">
        <form className="panel friendly-form" onSubmit={submit}>
          <div className="form-section">
            <h3>Visitor</h3>
            <p className="section-help">Start with a few details about this visit.</p>

            <div className="friendly-grid">
              <label>
                <span>Visitor type</span>
                <select value={form.visitorType} onChange={(e) => set("visitorType", e.target.value)}>
                  <option value="Returning_Visitor">Returning visitor</option>
                  <option value="New_Visitor">New visitor</option>
                  <option value="Other">Other</option>
                </select>
              </label>

              <label>
                <span>Visit month</span>
                <select value={form.month} onChange={(e) => set("month", e.target.value)}>
                  {months.map((month) => <option key={month}>{month}</option>)}
                </select>
              </label>
            </div>

            <label className="toggle-field">
              <span>Was this visit on a weekend?</span>
              <select value={String(form.weekend)} onChange={(e) => set("weekend", e.target.value === "true")}>
                <option value="false">No</option>
                <option value="true">Yes</option>
              </select>
            </label>
          </div>

          <div className="form-section">
            <h3>Browsing activity</h3>
            <p className="section-help">Tell us how much the visitor explored the website.</p>

            <div className="friendly-grid">
              <label>
                <span>Product pages viewed</span>
                <input type="number" min="0" max="500" step="1" value={form.productPages} onChange={(e) => set("productPages", e.target.value)} />
                <small>Number of product pages opened.</small>
              </label>

              <label>
                <span>Time spent on product pages</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="833" step="1" value={form.productMinutes} onChange={(e) => set("productMinutes", e.target.value)} />
                  <em>minutes</em>
                </div>
              </label>

              <label>
                <span>Information pages viewed</span>
                <input type="number" min="0" max="30" step="1" value={form.informationPages} onChange={(e) => set("informationPages", e.target.value)} />
                <small>Help, information or guide pages.</small>
              </label>

              <label>
                <span>Time on information pages</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="50" step="1" value={form.informationMinutes} onChange={(e) => set("informationMinutes", e.target.value)} />
                  <em>minutes</em>
                </div>
              </label>

              <label>
                <span>Account / administrative pages viewed</span>
                <input type="number" min="0" max="50" step="1" value={form.adminPages} onChange={(e) => set("adminPages", e.target.value)} />
                <small>Login, account or other administrative pages.</small>
              </label>

              <label>
                <span>Time on account / administrative pages</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="84" step="1" value={form.adminMinutes} onChange={(e) => set("adminMinutes", e.target.value)} />
                  <em>minutes</em>
                </div>
              </label>
            </div>
          </div>

          <div className="form-section">
            <h3>Engagement</h3>
            <p className="section-help">These describe how deeply the visitor interacted with the site.</p>

            <div className="friendly-grid">
              <label>
                <span>Bounce rate</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="100" step="1" value={form.bounceRate} onChange={(e) => set("bounceRate", e.target.value)} />
                  <em>%</em>
                </div>
                <small>Approximate share of visits that leave after one page.</small>
              </label>

              <label>
                <span>Exit rate</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="100" step="1" value={form.exitRate} onChange={(e) => set("exitRate", e.target.value)} />
                  <em>%</em>
                </div>
                <small>Approximate share of page views that end the session.</small>
              </label>

              <label>
                <span>Purchase-related page value</span>
                <div className="input-suffix">
                  <input type="number" min="0" max="400" step="0.1" value={form.pageValue} onChange={(e) => set("pageValue", e.target.value)} />
                  <em>value</em>
                </div>
                <small>Optional analytics value associated with pages visited. Use 0 if unknown.</small>
              </label>

              <label>
                <span>How close is the visit to a special shopping day?</span>
                <select value={form.specialDay} onChange={(e) => set("specialDay", e.target.value)}>
                  <option value="none">Not close</option>
                  <option value="slight">A little close</option>
                  <option value="moderate">Moderately close</option>
                  <option value="close">Very close</option>
                  <option value="veryClose">On the special day</option>
                </select>
              </label>
            </div>
          </div>

          <div className="technical-note">
            <Info size={17} />
            <span>Technical session details are handled automatically. You do not need to know any dataset codes or machine identifiers.</span>
          </div>

          {err && <div className="error full">{err}</div>}

          <button className="primary full" disabled={busy}>
            <Sparkles size={17} /> {busy ? "Scoring..." : "Predict purchase intention"}
          </button>
        </form>

        <div className="panel result-panel">
          {!r ? (
            <div className="empty">
              <Sparkles size={34} />
              <h3>Prediction result</h3>
              <p>Complete the session details and run the prediction to see the estimated purchase intent.</p>
            </div>
          ) : (
            <>
              <div className={r.prediction === 1 ? "result success" : "result neutral"}>
                <CheckCircle2 size={30} />
                <div>
                  <span>Predicted outcome</span>
                  <strong>{r.prediction_label}</strong>
                </div>
              </div>

              <div className="prob-grid">
                <div>
                  <span>Purchase probability</span>
                  <strong>{(r.purchase_probability * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{(r.confidence * 100).toFixed(1)}%</strong>
                </div>
              </div>

              <div className="intent">
                <span>Intent level</span>
                <strong>{r.intent_level}</strong>
                <div className="bar"><i style={{ width: `${r.purchase_probability * 100}%` }} /></div>
              </div>

              <p className="result-note">
                This is a model estimate based on the session information provided. It is not a guarantee that the visitor will purchase.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
