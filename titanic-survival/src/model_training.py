"""Entraînement du modèle Titanic."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE


def load_processed_data():
    """Charge X_train, X_valid, y_train, y_valid depuis data/processed."""
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_valid = pd.read_csv(PROCESSED_DATA_DIR / "X_valid.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_valid = pd.read_csv(PROCESSED_DATA_DIR / "y_valid.csv").squeeze()
    return X_train, X_valid, y_train, y_valid


def train_model(
    X_train, y_train, n_estimators: int = 100, max_depth: int | None = None, random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """Entraîne un RandomForest simple sur les données Titanic."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path: Path | None = None) -> Path:
    """Sauvegarde le modèle sur disque."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        model_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(model, model_path)
    return model_path


def run_training() -> Path:
    """Pipeline complet d'entraînement."""
    X_train, _, y_train, _ = load_processed_data()
    model = train_model(X_train, y_train)
    return save_model(model)


if __name__ == "__main__":
    run_training()
