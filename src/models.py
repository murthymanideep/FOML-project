"""Dependency parsing models: EdgeScorer and RelationClassifier."""
import joblib
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
import lightgbm as lgb
from .config import LIGHTGBM_HEAD_PARAMS, LIGHTGBM_REL_PARAMS

class EdgeScorer:
    """Predicts attachment scores for directed edges h -> m."""
    def __init__(self, model_type: str = "lgb"):
        self.model_type = model_type
        self.vectorizer = DictVectorizer(sparse=True)
        if model_type == "lgb":
            self.model = lgb.LGBMClassifier(**LIGHTGBM_HEAD_PARAMS)
        elif model_type == "linear":
            self.model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        elif model_type == "sgd":
            self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=100)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, edge_features: List[Dict[str, Any]], labels: List[int]):
        print(f"Fitting EdgeScorer ({self.model_type}) on {len(edge_features)} samples...")
        X = self.vectorizer.fit_transform(edge_features)
        y = np.array(labels)
        if self.model_type == "lgb":
            # For LightGBM, convert sparse matrix or use CSR
            self.model.fit(X.astype(np.float32), y)
        else:
            self.model.fit(X, y)
        print("EdgeScorer fitting completed.")

    def predict_scores(self, edge_features: List[Dict[str, Any]]) -> np.ndarray:
        """Returns probability / margin score for edge existence."""
        if not edge_features:
            return np.array([])
        X = self.vectorizer.transform(edge_features)
        if self.model_type == "lgb":
            # Probability of positive class
            probs = self.model.predict_proba(X.astype(np.float32))[:, 1]
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.decision_function(X)
        return probs

    def save(self, filepath: str):
        joblib.dump({"model": self.model, "vectorizer": self.vectorizer, "type": self.model_type}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "EdgeScorer":
        data = joblib.load(filepath)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.vectorizer = data["vectorizer"]
        return instance

class RelationClassifier:
    """Classifies dependency relation given edge h -> m."""
    def __init__(self, model_type: str = "lgb"):
        self.model_type = model_type
        self.vectorizer = DictVectorizer(sparse=True)
        self.label_to_idx = {}
        self.idx_to_label = {}
        if model_type == "lgb":
            self.model = lgb.LGBMClassifier(**LIGHTGBM_REL_PARAMS)
        elif model_type == "linear":
            self.model = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        elif model_type == "sgd":
            self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, max_iter=100)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, rel_features: List[Dict[str, Any]], labels: List[str]):
        print(f"Fitting RelationClassifier ({self.model_type}) on {len(rel_features)} samples...")
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        self.idx_to_label = {i: l for i, l in enumerate(unique_labels)}

        X = self.vectorizer.fit_transform(rel_features)
        y = np.array([self.label_to_idx[l] for l in labels])

        if self.model_type == "lgb":
            self.model.fit(X.astype(np.float32), y)
        else:
            self.model.fit(X, y)
        print(f"RelationClassifier fitting completed across {len(unique_labels)} classes.")

    def predict(self, rel_features: List[Dict[str, Any]]) -> List[str]:
        if not rel_features:
            return []
        X = self.vectorizer.transform(rel_features)
        if self.model_type == "lgb":
            preds = self.model.predict(X.astype(np.float32))
        else:
            preds = self.model.predict(X)
        return [self.idx_to_label.get(int(p), "dep") for p in preds]

    def save(self, filepath: str):
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "type": self.model_type
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "RelationClassifier":
        data = joblib.load(filepath)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.vectorizer = data["vectorizer"]
        instance.label_to_idx = data["label_to_idx"]
        instance.idx_to_label = data["idx_to_label"]
        return instance
