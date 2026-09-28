"""
Master Orchestrator Script for Online Shoppers Purchasing Intention Prediction.
Runs the complete academic ML pipeline from data acquisition to inference testing.
"""

import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, validate_dataset
from src.eda import run_full_eda
from src.train import run_training_pipeline
from src.predict import ShopperPurchasePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MasterPipeline")


def main():
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING MASTER PIPELINE: Online Shoppers Purchasing Intention Prediction")
    logger.info("=" * 80)
    
    # Step 1: Data Acquisition & Validation
    logger.info("\n>>> STEP 1: Acquiring & Validating Dataset...")
    df_raw = load_raw_data()
    val_report = validate_dataset(df_raw)
    logger.info(f"Dataset ready with shape {val_report['shape']}.")
    
    # Step 2: Exploratory Data Analysis & Visualizations
    logger.info("\n>>> STEP 2: Running Exploratory Data Analysis Suite...")
    run_full_eda()
    
    # Step 3: Model Training, Cross-Validation, Tuning & Evaluation
    logger.info("\n>>> STEP 3: Executing ML Training, Tuning, and Evaluation Pipeline...")
    train_results = run_training_pipeline()
    
    # Step 4: Verification of Saved Artifacts
    logger.info("\n>>> STEP 4: Verifying Saved Artifacts & Models...")
    best_model_path = train_results["best_model_path"]
    assert best_model_path.exists(), f"Best model not found at {best_model_path}!"
    logger.info(f"Best model artifact verified: {best_model_path} ({best_model_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # Step 5: Test Inference Engine
    logger.info("\n>>> STEP 5: Testing Inference on Sample Scenarios...")
    predictor = ShopperPurchasePredictor(best_model_path)
    
    test_samples = [
        {
            "name": "High Intent Buyer",
            "data": {
                "Administrative": 4, "Administrative_Duration": 120.0,
                "Informational": 2, "Informational_Duration": 60.0,
                "ProductRelated": 35, "ProductRelated_Duration": 1400.0,
                "BounceRates": 0.002, "ExitRates": 0.012, "PageValues": 45.0,
                "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
                "Browser": 2, "Region": 1, "TrafficType": 2,
                "VisitorType": "Returning_Visitor", "Weekend": False
            }
        },
        {
            "name": "Quick Bouncer",
            "data": {
                "Administrative": 0, "Administrative_Duration": 0.0,
                "Informational": 0, "Informational_Duration": 0.0,
                "ProductRelated": 1, "ProductRelated_Duration": 0.0,
                "BounceRates": 0.20, "ExitRates": 0.20, "PageValues": 0.0,
                "SpecialDay": 0.0, "Month": "Feb", "OperatingSystems": 1,
                "Browser": 1, "Region": 1, "TrafficType": 1,
                "VisitorType": "Returning_Visitor", "Weekend": False
            }
        }
    ]
    
    for sample in test_samples:
        pred_out = predictor.predict(sample["data"])
        logger.info(
            f"Scenario: '{sample['name']}' -> Prediction: {pred_out['prediction_label']} | "
            f"Purchase Probability: {pred_out['purchase_probability']:.2%} | "
            f"Intent Level: {pred_out['intent_level']}"
        )
        
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info(f"Best Model Selected: {train_results['best_model_name']}")
    logger.info("All reports, figures, datasets, and models are generated and ready.")
    logger.info("Launch interactive dashboard using: streamlit run app.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
