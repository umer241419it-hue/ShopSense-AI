import { Router } from "express";
import axios from "axios";
import { Prediction } from "../models/Prediction.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();
router.use(requireAuth);

router.post("/", async (req, res, next) => {
  try {
    if (!req.body || typeof req.body !== "object" || Array.isArray(req.body)) {
      return res.status(400).json({ message: "Prediction payload must be a JSON object." });
    }

    const {
      Administrative,
      Administrative_Duration,
      Informational,
      Informational_Duration,
      ProductRelated,
      ProductRelated_Duration,
      SpecialDay,
      Month,
      VisitorType,
      Weekend,
    } = req.body;

    // Strict validation of production numeric attributes
    const numericFields = [
      { name: "Administrative", val: Administrative },
      { name: "Administrative_Duration", val: Administrative_Duration },
      { name: "Informational", val: Informational },
      { name: "Informational_Duration", val: Informational_Duration },
      { name: "ProductRelated", val: ProductRelated },
      { name: "ProductRelated_Duration", val: ProductRelated_Duration },
      { name: "SpecialDay", val: SpecialDay },
    ];

    for (const f of numericFields) {
      if (f.val !== undefined) {
        const num = Number(f.val);
        if (!Number.isFinite(num) || num < 0) {
          return res.status(400).json({
            message: "One or more session attributes are invalid.",
            detail: `Field '${f.name}' must be a non-negative number.`,
          });
        }
      }
    }

    if (SpecialDay !== undefined) {
      const sd = Number(SpecialDay);
      if (sd < 0 || sd > 1) {
        return res.status(400).json({
          message: "One or more session attributes are invalid.",
          detail: "Field 'SpecialDay' must be between 0.0 and 1.0.",
        });
      }
    }

    // Construct strictly the 10 real-time observable production features
    const productionPayload = {
      Administrative: Math.max(0, Number(Administrative) || 0),
      Administrative_Duration: Math.max(0, Number(Administrative_Duration) || 0),
      Informational: Math.max(0, Number(Informational) || 0),
      Informational_Duration: Math.max(0, Number(Informational_Duration) || 0),
      ProductRelated: Math.max(0, Number(ProductRelated) || 0),
      ProductRelated_Duration: Math.max(0, Number(ProductRelated_Duration) || 0),
      SpecialDay: Math.min(1, Math.max(0, Number(SpecialDay) || 0)),
      Month: typeof Month === "string" && Month.trim() ? Month.trim() : "May",
      VisitorType: typeof VisitorType === "string" && VisitorType.trim() ? VisitorType.trim() : "Returning_Visitor",
      Weekend: Boolean(Weekend),
    };

    const mlUrl = process.env.ML_SERVICE_URL;
    if (!mlUrl) {
      return res.status(503).json({ message: "ML service is not configured." });
    }

    const response = await axios.post(`${mlUrl}/predict`, productionPayload, { timeout: 15000 });
    const result = response.data;

    const saved = await Prediction.create({
      user: req.userId,
      input: productionPayload,
      prediction: result.prediction,
      predictionLabel: result.prediction_label,
      purchaseProbability: result.purchase_probability,
      noPurchaseProbability: result.no_purchase_probability,
      confidence: result.confidence,
      intentLevel: result.intent_level,
    });

    return res.status(201).json({
      ...result,
      id: saved._id,
      createdAt: saved.createdAt,
    });
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 422) {
        return res.status(400).json({
          message: "One or more session attributes are invalid.",
          detail: error.response.data?.detail,
        });
      }
      if (error.code === "ECONNABORTED") {
        return res.status(504).json({ message: "ML service request timed out." });
      }
      if (error.code === "ECONNREFUSED" || !error.response) {
        return res.status(503).json({ message: "ML service is unavailable." });
      }
      return res.status(502).json({ message: "ML service error." });
    }
    return next(error);
  }
});

router.get("/", async (req, res, next) => {
  try {
    const limit = Math.min(Math.max(Number(req.query.limit) || 20, 1), 100);
    const predictions = await Prediction.find({ user: req.userId })
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();
    return res.json(predictions);
  } catch (error) {
    return next(error);
  }
});

export default router;
