import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __serverDir = path.dirname(fileURLToPath(import.meta.url));
dotenv.config();
if (!process.env.JWT_SECRET) {
  dotenv.config({ path: path.resolve(__serverDir, "../.env") });
}
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import { connectDatabase } from "./config/db.js";
import authRoutes from "./routes/auth.js";
import predictionRoutes from "./routes/predictions.js";
import analyticsRoutes from "./routes/analytics.js";
import modelRoutes from "./routes/model.js";

const app = express();
const port = Number(process.env.PORT || 5000);

if (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32) {
  throw new Error("JWT_SECRET must be configured with at least 32 characters.");
}

const allowedOrigins = (process.env.CLIENT_URL || "http://localhost:5173")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error("Origin not allowed by CORS"));
  },
}));
app.use(express.json({ limit: "1mb" }));

app.get("/api/health", (_req, res) => {
  res.json({
    status: mongoose.connection.readyState === 1 ? "ok" : "degraded",
    service: "shopsense-backend",
    database: mongoose.connection.readyState === 1 ? "connected" : "disconnected",
  });
});

app.use("/api/auth", authRoutes);
app.use("/api/predictions", predictionRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api/model", modelRoutes);

app.use((_req, res) => {
  res.status(404).json({ message: "API route not found" });
});

app.use((error, _req, res, _next) => {
  console.error(error);
  const status = error.status || (error.message === "Origin not allowed by CORS" ? 403 : 500);
  res.status(status).json({
    message: status === 500 ? "Internal server error" : error.message,
  });
});

connectDatabase()
  .then(() => app.listen(port, () => console.log(`ShopSense backend listening on ${port}`)))
  .catch((error) => {
    console.error("Database connection failed:", error.message);
    process.exit(1);
  });
