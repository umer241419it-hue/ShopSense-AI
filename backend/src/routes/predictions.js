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

    const mlUrl = process.env.ML_SERVICE_URL;
    if (!mlUrl) {
      return res.status(503).json({ message: "ML service is not configured." });
    }

    const response = await axios.post(`${mlUrl}/predict`, req.body, { timeout: 15000 });
    const result = response.data;

    const saved = await Prediction.create({
      user: req.userId,
      input: req.body,
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
