import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
import os 
import pickle
import pandas as pd

def precision_at_k(model, X, y, k=10, batch_size=4096):
    """Multi-label ranking precision: P@k = (# true positives in top-k) / k."""
    model.eval()
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    scores = np.concatenate(scores)

    topk = np.argsort(-scores, axis=1)[:, :k]
    hits = np.take_along_axis(y, topk, axis=1).sum(axis=1)
    return float((hits / k).mean())


def per_sample_loss(model, X, y, batch_size=4096):
    """Mean BCE loss per sample, the signal the attacker exploits."""
    model.eval()
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(y[i:i + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            l = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none').mean(dim=1)
            losses.append(l.cpu().numpy())
    return np.concatenate(losses)


def mia_auc(model, X_forget, y_forget, X_retain, y_retain, seed=42, max_n=5000):
    """Loss-based MIA: can an attacker tell Df from Dr? Ideal AUC is 0.5."""
    rng = np.random.default_rng(seed)
    n = min(len(X_forget), len(X_retain), max_n)
    fi = rng.choice(len(X_forget), n, replace=False)
    ri = rng.choice(len(X_retain), n, replace=False)

    loss_f = per_sample_loss(model, X_forget[fi], y_forget[fi])
    loss_r = per_sample_loss(model, X_retain[ri], y_retain[ri])

    features = np.concatenate([loss_f, loss_r]).reshape(-1, 1)
    labels = np.concatenate([np.ones(n), np.zeros(n)])

    attacker = LogisticRegression(max_iter=1000)
    probs = cross_val_predict(attacker, features, labels, cv=5, method='predict_proba')[:, 1]
    return float(roc_auc_score(labels, probs))


def final_score(precision_val, auc, execution_time, time_threshold=300.0):
    """Weighted competition score: 45% utility, 45% MIA resistance, 10% speed."""
    mia_resistance = 1.0 - 2.0 * abs(auc - 0.5)
    time_score = 1.0 if execution_time <= time_threshold else time_threshold / execution_time
    return 0.45 * precision_val + 0.45 * mia_resistance + 0.10 * time_score


class Timer:
    """Times the unlearning phase only."""
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.t0




def save_submission(model, payload, val_ids, execution_time,
                    out_dir, id_col='user_id'):
    """Writes the three files required by the submission spec.

    File names are case-sensitive and no extra files are allowed.
    """
    os.makedirs(out_dir, exist_ok=True)

    new_payload = {
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'architecture': payload['architecture'],
        'best_hyperparameters': payload['best_hyperparameters'],
        'model_class_source': payload['model_class_source'],
    }
    with open(os.path.join(out_dir, 'model_artifact'), 'wb') as f:
        pickle.dump(new_payload, f)

    with open(os.path.join(out_dir, 'execution_time.txt'), 'w') as f:
        f.write(str(max(1, int(round(execution_time)))))

    pd.DataFrame({id_col: val_ids}).to_csv(
        os.path.join(out_dir, 'validation_ids.csv'), index=False)

    print(f"Submission written to {out_dir}: {sorted(os.listdir(out_dir))}")