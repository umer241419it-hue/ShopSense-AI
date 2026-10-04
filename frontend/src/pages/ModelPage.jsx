import { useEffect, useState } from "react";
import { BarChart3, Info, CheckCircle2 } from "lucide-react";
import api from "../api";

export function ModelPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  const [activeTab, setActiveTab] = useState("production");

  useEffect(() => {
    api
      .get("/model/metrics")
      .then((x) => setD(x.data))
      .catch((e) => setErr(e.response?.data?.message || "Unable to load model metrics"));
  }, []);

  const displayRows =
    activeTab === "production"
      ? d?.productionRows?.length
        ? d.productionRows
        : d?.rows || []
      : d?.academicRows?.length
      ? d.academicRows
      : [];

  const currentBest =
    activeTab === "production"
      ? d?.bestModel || "Production Random Forest (Tuned)"
      : d?.academicBestModel || "Random Forest (Tuned)";

  return (
    <section className="page">
      <div className="page-title">
        <div>
          <span className="eyebrow">DUAL-MODEL ARCHITECTURE</span>
          <h1>Model performance</h1>
          <p>
            Hold-out validation metrics for both the live production model and the historical academic benchmark.
          </p>
        </div>
      </div>

      <div className="panel" style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
        {err ? (
          <div className="error">{err}</div>
        ) : !d ? (
          <div className="loading">Loading model metrics...</div>
        ) : (
          <>
            {/* Model Architecture Toggle */}
            <div style={{ display: "flex", gap: "10px", borderBottom: "1px solid #e2e8f0", paddingBottom: "12px" }}>
              <button
                type="button"
                className={activeTab === "production" ? "primary" : "secondary"}
                onClick={() => setActiveTab("production")}
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <CheckCircle2 size={16} />
                Production Model (Live Inference)
              </button>
              <button
                type="button"
                className={activeTab === "academic" ? "primary" : "secondary"}
                onClick={() => setActiveTab("academic")}
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <BarChart3 size={16} />
                Academic Benchmark (UCI 17-Features)
              </button>
            </div>

            {/* Model Banner */}
            <div className="model-banner">
              <BarChart3 />
              <div>
                <strong>
                  {currentBest} — {activeTab === "production" ? "Production Model" : "Academic Benchmark"}
                </strong>
                <span>
                  {activeTab === "production"
                    ? "Trained strictly on 10 real-time observable session signals. Retrospective PageValues and synthetic proxies are excluded."
                    : "Trained on the complete historical 17-feature UCI dataset including retrospective Google Analytics PageValues."}
                </span>
              </div>
            </div>

            {/* Metrics Table */}
            <div className="table header metrics">
              <span>Model</span>
              <span>Accuracy</span>
              <span>Precision</span>
              <span>Recall</span>
              <span>F1</span>
              <span>ROC-AUC</span>
            </div>

            {displayRows.map((r, i) => (
              <div className="row metrics" key={i}>
                <span style={{ fontWeight: 600 }}>{r.Model}</span>
                <span>{(Number(r.Accuracy) * 100).toFixed(2)}%</span>
                <span>{(Number(r.Precision) * 100).toFixed(2)}%</span>
                <span>{(Number(r.Recall) * 100).toFixed(2)}%</span>
                <span style={{ fontWeight: 700, color: "#0f766e" }}>{Number(r["F1-Score"]).toFixed(3)}</span>
                <span>{Number(r["ROC-AUC"]).toFixed(3)}</span>
              </div>
            ))}

            {/* Academic Technical Note */}
            <div
              style={{
                marginTop: "12px",
                padding: "14px 16px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
                borderRadius: "8px",
                display: "flex",
                gap: "10px",
                alignItems: "flex-start",
                fontSize: "13px",
                color: "#475467",
                lineHeight: "1.5",
              }}
            >
              <Info size={18} style={{ color: "#0f766e", flexShrink: 0, marginTop: "2px" }} />
              <div>
                <strong style={{ color: "#0f172a", display: "block", marginBottom: "4px" }}>
                  Dual-Model Architecture Context
                </strong>
                The project maintains two model contexts. The academic benchmark model uses the complete historical UCI feature set for reproducible dataset benchmarking. The production inference model uses only features that are genuinely observable during an active browsing session. This separation prevents retrospective attribution variables and anonymized dataset identifiers from being fabricated as user inputs.
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
