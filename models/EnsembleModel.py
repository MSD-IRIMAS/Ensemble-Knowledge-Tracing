import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# Voting strategies

def soft_voting(preds_list: list) -> np.ndarray:
    return np.mean(preds_list, axis=0)


def weighted_voting(preds_list: list, weights: list) -> np.ndarray:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    stacked = np.array(preds_list)            # (n_models, n_samples)
    return (stacked * w[:, np.newaxis]).sum(axis=0)



# Stacking meta-learner


class StackingMeta:

    def __init__(self, C: float = 1.0):
        self.meta = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        self.fitted = False

    def fit(self, val_preds_list: list, val_labels: np.ndarray):
        X = np.column_stack(val_preds_list)          # (n_val, n_models)
        y = (val_labels >= 0.5).astype(int)
        self.meta.fit(X, y)
        self.fitted = True

    def predict(self, test_preds_list: list) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StackingMeta must be fitted before calling predict().")
        X = np.column_stack(test_preds_list)         # (n_test, n_models)
        return self.meta.predict_proba(X)[:, 1]      # probability of class 1



# Shared evaluation


def evaluate_predictions(preds: np.ndarray, labels: np.ndarray) -> tuple:
    binary_preds  = (preds  >= 0.5).astype(int)
    binary_labels = (labels >= 0.5).astype(int)

    auc = roc_auc_score(binary_labels, preds)
    f1  = f1_score(binary_labels, binary_preds, zero_division=0)
    acc = accuracy_score(binary_labels, binary_preds)

    return round(auc, 4), round(f1, 4), round(acc, 4)


