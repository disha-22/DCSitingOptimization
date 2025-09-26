import numpy as np

def theil_index(S: np.ndarray, eps=1e-12) -> float:
    S = np.asarray(S, dtype=float).reshape(-1)
    S = np.clip(S, 0.0, None)
    m = S.mean()
    if m <= eps:
        return 0.0
    r = S / m
    return float((r * np.log(r + eps)).mean())

def atkinson_index(S: np.ndarray, epsilon: float = 1.0, eps=1e-12) -> float:
    S = np.asarray(S, dtype=float).reshape(-1)
    S = np.clip(S, 0.0, None)
    m = S.mean()

    if m <= eps:
        return 0.0
    if abs(epsilon - 1.0) < 1e-9:
        g = np.exp(np.log(S + eps).mean())  # geometric mean
        return float(1.0 - g / m)
    p = 1.0 - epsilon
    mean_power = np.mean((S + eps)**p)
    eq = mean_power**(1.0/p)
    return float(1.0 - eq / m)

def evaluate_equity_metrics(results, data):
    """Evaluate various equity metrics post-optimization"""
    
    S = (data['S_g'] @ np.sum(results['g'], axis=1) +
         data['S_s'] @ results['s'].flatten() +
         data['S_w'] @ results['w'].flatten() +
         data['S_dc'] @ np.sum(results['a'], axis=1))
    
    metrics = {
        'max_water_scarcity': np.max(S),
        'mad_water_scarcity': np.mean(np.abs(S[:, np.newaxis] - S)),
        'theil_index': theil_index(S),
        'atkinson_0.5': atkinson_index(S, epsilon=0.5),
        'atkinson_1.0': atkinson_index(S, epsilon=1.0),
        'atkinson_2.0': atkinson_index(S, epsilon=2.0),
        'gini_coefficient': gini_coefficient(S) if you want to add this too
    }
    
    return metrics

