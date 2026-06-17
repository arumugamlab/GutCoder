import pandas as pd
import numpy as np

def oversampling_gaussian_noise(X: np.ndarray, y: np.ndarray, sigma: float = 0.1, random_state: int = 42):
    """Noise only on non-zero entries, clip(0) removes negatives."""
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() == counts.max():
        return X, y

    minority_class = classes[np.argmin(counts)]
    n_to_generate  = counts.max() - counts.min()
    X_minority     = X[y == minority_class]
    idx            = rng.choice(len(X_minority), size=n_to_generate, replace=True)
    X_to_aug       = X_minority[idx]

    augmented = []
    for sample in X_to_aug:
        noise  = rng.normal(0, sigma, size=sample.shape)
        noise *= sample != 0
        aug    = np.clip(sample + noise, 0, None)
        augmented.append(aug)

    X_res = np.vstack([X, np.array(augmented, dtype=np.float32)])
    y_res = np.concatenate([y, np.full(n_to_generate, minority_class)])
    print(f"  Augmentation: {counts.min()}, {counts.max()} minority samples (σ={sigma})")
    return X_res, y_res
