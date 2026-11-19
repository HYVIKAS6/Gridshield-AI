"""
Standalone Model Training Script for GridShield AI Smart Grid Intrusion Detection.
Trains, evaluates, and persists all five machine learning & deep learning models:
1. Isolation Forest (Unsupervised)
2. Random Forest (Supervised Bagging)
3. XGBoost (Supervised Gradient Boosting)
4. MLP Neural Network (Deep Feedforward)
5. PyTorch LSTM (Deep Recurrent Temporal)

Usage:
    python train.py --dataset msu_benchmark --epochs 15
    python train.py --dataset synthetic_sample --epochs 10
"""

import argparse
import json
import os
import sys
import time

# Ensure project root in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing import LeakFreeGridPipeline, chronological_train_test_split, validate_grid_dataframe
from data.download_datasets import load_dataset
from ml_models.random_forest import GridRandomForestModel
from ml_models.xgboost_model import GridXGBoostModel, HAS_XGBOOST
from ml_models.isolation_forest import GridIsolationForestModel
from ml_models.mlp_model import GridMLPModel
from ml_models.lstm_model import GridLSTMModel, HAS_TORCH
from model_training import calibrate_contamination, compare_all_models


def parse_args():
    parser = argparse.ArgumentParser(description="GridShield AI Model Training Engine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="msu_benchmark",
        choices=["msu_benchmark", "synthetic_sample"],
        help="Dataset to train models on (default: msu_benchmark)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.25,
        help="Chronological train-test split ratio (default: 0.25)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for LSTM and MLP (default: 15)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "saved"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible random seed (default: 42)",
    )
    return parser.parse_args()


def run_training():
    args = parse_args()
    print("=" * 80)
    print(" 🛡️  GRIDSHIELD AI — STANDALONE MODEL TRAINING SUITE")
    print("=" * 80)
    print(f"Dataset Target:        {args.dataset}")
    print(f"Test Ratio:            {args.test_ratio * 100:.1f}%")
    print(f"Random Seed:           {args.seed}")
    print(f"Deep Learning Epochs:  {args.epochs}")
    print(f"Artifact Save Dir:     {args.save_dir}")
    print("-" * 80)

    # 1. Load Dataset
    print(f"\n[1/5] Ingesting dataset: '{args.dataset}'...")
    df, desc = load_dataset(args.dataset)
    is_valid, msg, df_clean = validate_grid_dataframe(df)
    if not is_valid:
        print(f"❌ Error validating dataset: {msg}")
        sys.exit(1)

    print(f"✅ Loaded {len(df_clean)} records. {desc}")
    attack_count = int(df_clean["target"].sum())
    normal_count = len(df_clean) - attack_count
    print(f"   Distribution: {normal_count} Normal ({normal_count/len(df_clean)*100:.1f}%), {attack_count} Attack ({attack_count/len(df_clean)*100:.1f}%)")

    # 2. Chronological Split & Leak-Free Pipeline
    print("\n[2/5] Executing chronological train-test split & leak-free feature pipeline...")
    train_df, test_df = chronological_train_test_split(df_clean, test_ratio=args.test_ratio)
    pipeline = LeakFreeGridPipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    y_train = train_df["target"].values
    y_test = test_df["target"].values

    print(f"   Training partition: {len(X_train)} samples, Feature dimensionality: {X_train.shape[1]}")
    print(f"   Testing partition:  {len(X_test)} samples (strictly unseen chronological holdout)")

    # 3. Model Training
    print("\n[3/5] Training 5 Core Anomaly Detection Architectures...")
    models = {}
    evaluations = {}
    os.makedirs(args.save_dir, exist_ok=True)
    meta_dir = os.path.join(PROJECT_ROOT, "models", "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    # (A) Isolation Forest
    print("   -> Training Isolation Forest (Unsupervised)...", end="", flush=True)
    contam = calibrate_contamination(y_train)
    iso_model = GridIsolationForestModel(contamination=contam, random_state=args.seed)
    iso_model.fit(X_train)
    models["Isolation Forest"] = iso_model
    evaluations["Isolation Forest"] = iso_model.evaluate(X_test, y_test)
    iso_path = os.path.join(args.save_dir, "isolation_forest.joblib")
    iso_model.save(iso_path)
    print(f" Done! Acc: {evaluations['Isolation Forest']['accuracy']:.3f}, F1: {evaluations['Isolation Forest']['f1']:.3f}")

    # (B) Random Forest
    print("   -> Training Random Forest Classifier (Bagging)...", end="", flush=True)
    rf_model = GridRandomForestModel(n_estimators=100, random_state=args.seed)
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = rf_model
    evaluations["Random Forest"] = rf_model.evaluate(X_test, y_test)
    rf_path = os.path.join(args.save_dir, "random_forest.joblib")
    rf_model.save(rf_path)
    print(f" Done! Acc: {evaluations['Random Forest']['accuracy']:.3f}, F1: {evaluations['Random Forest']['f1']:.3f}")

    # (C) XGBoost
    if HAS_XGBOOST:
        print("   -> Training XGBoost Classifier (Boosting)...", end="", flush=True)
        xgb_model = GridXGBoostModel(n_estimators=100, random_state=args.seed)
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model
        evaluations["XGBoost"] = xgb_model.evaluate(X_test, y_test)
        xgb_path = os.path.join(args.save_dir, "xgboost_model.joblib")
        xgb_model.save(xgb_path)
        print(f" Done! Acc: {evaluations['XGBoost']['accuracy']:.3f}, F1: {evaluations['XGBoost']['f1']:.3f}")

    # (D) MLP Neural Network
    print("   -> Training MLP Neural Network (Feedforward)...", end="", flush=True)
    mlp_model = GridMLPModel(max_iter=args.epochs * 10, random_state=args.seed)
    mlp_model.fit(X_train, y_train)
    models["MLP Neural Network"] = mlp_model
    evaluations["MLP Neural Network"] = mlp_model.evaluate(X_test, y_test)
    mlp_path = os.path.join(args.save_dir, "mlp_model.joblib")
    mlp_model.save(mlp_path)
    print(f" Done! Acc: {evaluations['MLP Neural Network']['accuracy']:.3f}, F1: {evaluations['MLP Neural Network']['f1']:.3f}")

    # (E) PyTorch LSTM
    if HAS_TORCH:
        print(f"   -> Training PyTorch LSTM ({args.epochs} Epochs)...", end="", flush=True)
        lstm_model = GridLSTMModel(epochs=args.epochs, random_state=args.seed)
        lstm_model.fit(X_train, y_train)
        models["PyTorch LSTM"] = lstm_model
        evaluations["PyTorch LSTM"] = lstm_model.evaluate(X_test, y_test)
        lstm_path = os.path.join(args.save_dir, "lstm_model.pt")
        lstm_model.save(lstm_path)
        print(f" Done! Acc: {evaluations['PyTorch LSTM']['accuracy']:.3f}, F1: {evaluations['PyTorch LSTM']['f1']:.3f}")

    # 4. Save Pipeline & Metadata
    import joblib
    pipeline_path = os.path.join(args.save_dir, "feature_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": timestamp_str,
        "dataset_name": args.dataset,
        "n_samples": len(df_clean),
        "test_ratio": args.test_ratio,
        "models_evaluated": list(evaluations.keys()),
        "evaluations": {
            k: {
                "accuracy": v["accuracy"],
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1"],
                "roc_auc": v["roc_auc"],
                "false_positive_rate": v["false_positive_rate"],
                "training_time_sec": v["training_time_sec"],
                "inference_time_sec": v["inference_time_sec"],
            }
            for k, v in evaluations.items()
        },
    }
    meta_path = os.path.join(meta_dir, f"training_meta_{timestamp_str}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 5. Model Comparison Summary
    print("\n[4/5] Multi-Model Performance Comparison Table (Holdout Test Evaluation):")
    print("-" * 85)
    comp_df = compare_all_models(evaluations)
    print(comp_df.to_string(index=False))
    print("-" * 85)

    print("\n[5/5] Artifact Persistence Status:")
    print(f"   ✅ Saved feature pipeline to: {pipeline_path}")
    print(f"   ✅ Saved model checkpoints to: {args.save_dir}")
    print(f"   ✅ Saved training metadata to: {meta_path}")
    print("\n🎉 Training run completed successfully!\n")


if __name__ == "__main__":
    run_training()
