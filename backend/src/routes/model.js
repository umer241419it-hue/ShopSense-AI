import { Router } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const router = Router();
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const PROD_METRICS_PATH = path.join(PROJECT_ROOT, "reports", "production_model_comparison.csv");
const ACADEMIC_METRICS_PATH = path.join(PROJECT_ROOT, "reports", "model_comparison.csv");

function parseCsv(csv) {
  const lines = csv.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((header, index) => {
        const value = values[index] ?? "";
        const numeric = Number(value);
        return [header, value !== "" && Number.isFinite(numeric) ? numeric : value];
      })
    );
  });
}

router.get("/metrics", async (_req, res, next) => {
  try {
    let prodRows = [];
    let academicRows = [];

    try {
      const prodCsv = await fs.readFile(PROD_METRICS_PATH, "utf8");
      prodRows = parseCsv(prodCsv);
    } catch {
      // Fallback if not yet created
    }

    try {
      const acadCsv = await fs.readFile(ACADEMIC_METRICS_PATH, "utf8");
      academicRows = parseCsv(acadCsv);
    } catch {
      // Fallback
    }

    const rows = prodRows.length > 0 ? prodRows : academicRows;
    const bestModel = [...rows].sort((a, b) => (b["F1-Score"] ?? 0) - (a["F1-Score"] ?? 0))[0] ?? null;
    const bestAcademic = [...academicRows].sort((a, b) => (b["F1-Score"] ?? 0) - (a["F1-Score"] ?? 0))[0] ?? null;

    return res.json({
      activeModel: "production_model.joblib",
      modelType: "Production (Real-Time Observable Features)",
      bestModel: bestModel?.Model ?? "Production Random Forest (Tuned)",
      rows,
      productionRows: prodRows,
      academicRows,
      academicBestModel: bestAcademic?.Model ?? "Random Forest (Tuned)",
      features: [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "SpecialDay",
        "Month",
        "VisitorType",
        "Weekend",
      ],
      dataset: {
        name: "UCI Online Shoppers Purchasing Intention",
        instances: 12330,
        target: "Revenue",
      },
      technicalNote:
        "The project maintains two model contexts. The academic benchmark model uses the complete historical UCI feature set for reproducible dataset benchmarking. The production inference model uses only features that are genuinely observable during an active browsing session. This separation prevents retrospective attribution variables and anonymized dataset identifiers from being fabricated as user inputs.",
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
