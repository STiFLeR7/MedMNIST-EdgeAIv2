import numpy as np

def accuracy(preds, labels): return (preds == labels).mean()

def brier_score(probs, labels):
    one_hot = np.eye(probs.shape[1])[labels]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))

def ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            acc = (predictions[mask] == labels[mask]).mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
    return ece
