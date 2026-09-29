"""Bandit (M4, Rev 2 §9)."""

from __future__ import annotations

import numpy as np

class LinearThompsonSampler:
    """Bayesian linear regression for contextual Thompson Sampling.
    
    Fits: r ~ N(w^T x, sigma^2)
    Prior on w: N(0, lambda_reg^-1 I)
    """
    def __init__(self, d: int, lambda_reg: float = 1.0, sigma: float = 1.0):
        self.d = d
        self.sigma2 = sigma ** 2
        # Precision matrix B = X^T X / sigma^2 + lambda I
        self.B = np.eye(d) * lambda_reg
        # B_inv is maintained to avoid inverting at sample time
        self.B_inv = np.eye(d) / lambda_reg
        # f = X^T Y / sigma^2
        self.f = np.zeros(d)
        
    def update(self, x: np.ndarray, r: float):
        """Rank-1 update of the posterior."""
        x = x.reshape(-1, 1)
        # Sherman-Morrison update for B_inv
        # (B + u v^T)^-1 = B^-1 - B^-1 u v^T B^-1 / (1 + v^T B^-1 u)
        # Here u = x / sigma, v = x / sigma
        B_inv_x = self.B_inv @ x
        denom = self.sigma2 + float(x.T @ B_inv_x)
        
        self.B += (x @ x.T) / self.sigma2
        self.B_inv -= (B_inv_x @ B_inv_x.T) / denom
        
        self.f += (x.flatten() * r) / self.sigma2
        
    def sample_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Sample weights from the posterior N(mu, B^-1)."""
        mu = self.B_inv @ self.f
        # B_inv is symmetric positive definite
        try:
            L = np.linalg.cholesky(self.B_inv)
            z = rng.standard_normal(self.d)
            return mu + L @ z
        except np.linalg.LinAlgError:
            # Fallback if numerical issues
            return mu

    def inflate_variance(self, factor: float = 2.0):
        """Forgetting mechanism: inflate posterior variance (shrink precision)."""
        self.B /= factor
        self.B_inv *= factor
        # Note: f is not scaled to keep the mean mu = B^-1 f constant
        self.f /= factor

class PolicyBandit:
    """Manages the TS models for the 8 arms (R0-R7)."""
    
    def __init__(self, d: int, lambda_reg: float = 1.0, sigma: float = 1.0, seed: int = 42):
        self.arms = [f"R{i}" for i in range(8)]
        self.models = {arm: LinearThompsonSampler(d, lambda_reg, sigma) for arm in self.arms}
        self.rng = np.random.default_rng(seed)
        
    def select_arm(self, x: np.ndarray, restrict_to: list[str] | None = None) -> tuple[str, bool]:
        """Select an arm using Thompson sampling."""
        best_arm = None
        best_val = -np.inf
        
        # Staggered exploration (epsilon-greedy style overlay or just TS)
        # TS natively explores, so we just sample.
        
        candidates = restrict_to if restrict_to is not None else self.arms
        
        for arm in candidates:
            w_sample = self.models[arm].sample_weights(self.rng)
            val = float(w_sample @ x)
            if val > best_val:
                best_val = val
                best_arm = arm
                
        # Return selected arm, and a boolean indicating if it was exploratory
        # In pure TS, everything is a sample. 
        # (Could define exploratory if it differs from the argmax of mu).
        return best_arm, False
        
    def update(self, arm: str, x: np.ndarray, r: float):
        """Update the posterior for the pulled arm."""
        self.models[arm].update(x, r)
        
    def forget(self, factor: float = 2.0):
        """Apply forgetting to all arms (e.g. on a CUSUM alarm)."""
        for m in self.models.values():
            m.inflate_variance(factor)
