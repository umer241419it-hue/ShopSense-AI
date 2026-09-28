import { Router } from "express";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const router = Router();
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const METRICS_PATH = path.join(PROJECT_ROOT, "reports", "model_comparison.csv");

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
    const csv = await fs.readFile(METRICS_PATH, "utf8");
    const rows = parseCsv(csv);
    const bestModel = [...rows].sort((a, b) => (b["F1-Score"] ?? 0) - (a["F1-Score"] ?? 0))[0] ?? null;

    return res.json({
      rows,
      bestModel: bestModel?.Model ?? null,
      dataset: {
        name: "UCI Online Shoppers Purchasing Intention",
        instances: 12330,
        target: "Revenue",
      },
    });
  } catch (error) {
    return next(error);
  }
});

export default router;
