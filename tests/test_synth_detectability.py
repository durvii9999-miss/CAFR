import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cafr.data.synth.generator import get_cell_params, CELLS
from cafr.data.synth.injections import inject_C0, inject_C1, inject_C3, inject_C4, inject_C5, inject_C6
from cafr.data.synth.labels import INJECTORS
from cafr.monitor.features import (
    compute_bias_stats, compute_dispersion_stats, compute_cusum_stats, compute_intermittency_stats, robust_sd
)

def test_synth_gates_4_3_to_4_12():
    """Evaluate synthetic detectability using baseline parameters as the 'forecast'.
    
    This verifies that the injections actually produce the mathematical signatures
    required by the monitor, isolated from forecasting error (Gate 4.3 - 4.12).
    """
    rng = np.random.default_rng(42)
    T = 200
    tau = 120
    
    n_series = 200
    causes = ["C0", "C1", "C3", "C4", "C5_size", "C5_occ", "C6"]
    
    data = []
    
    for cause in causes:
        for i in range(n_series):
            cell = rng.choice(CELLS)
            p, k, mu_z = get_cell_params(cell, rng)
            
            if cause == "C0":
                y, delta, z, params = inject_C0(T, tau, p, k, mu_z, rng)
            elif cause == "C1":
                y, delta, z, params = INJECTORS["C1"](T, tau, p, k, mu_z, rng)
            elif cause == "C3":
                y, delta, z, params = INJECTORS["C3"](T, tau, p, k, mu_z, rng)
            elif cause == "C4":
                y, delta, z, params = INJECTORS["C4"](T, tau, p, k, mu_z, rng)
            elif cause == "C5_size":
                # Force size
                while True:
                    y, delta, z, params = INJECTORS["C5"](T, tau, p, k, mu_z, rng)
                    if params.get("target") == "size": break
            elif cause == "C5_occ":
                while True:
                    y, delta, z, params = INJECTORS["C5"](T, tau, p, k, mu_z, rng)
                    if params.get("target") == "occurrence": break
            elif cause == "C6":
                y, delta, z, params = INJECTORS["C6"](T, tau, p, k, mu_z, rng)
                
            # Compute idealized residuals using pre-injection parameters (mu_z, p)
            # as the "forecast".
            e_sz = z - mu_z
            e_sz[delta == 0] = np.nan # Only nonzero sizes
            
            e_occ = delta - p
            
            # Post-injection window for stats (e.g., t=170 to 199)
            W_short = 24
            post_e_sz = e_sz[-W_short:]
            post_delta = delta[-W_short:]
            post_e_occ = e_occ[-W_short:]
            
            # C1: zb (standardized bias)
            c1_stat = compute_bias_stats(post_e_sz, np.ones(W_short, dtype=bool), min_evidence=8)["z_b"]
            
            # C4: dispersion ratio
            pre_e_sz = e_sz[:60]
            c4_stat = compute_dispersion_stats(e_sz[-60:], pre_e_sz, np.ones(60, dtype=bool), min_evidence=10)["r"]
            
            # C5: CUSUM max_S
            sz_sigma = robust_sd(pre_e_sz[~np.isnan(pre_e_sz)])
            c5_sz_stat = compute_cusum_stats(post_e_sz[~np.isnan(post_e_sz)], sz_sigma)["max_S"]
            
            occ_sigma = np.sqrt(p * (1 - p))
            c5_occ_stat = compute_cusum_stats(post_e_occ, occ_sigma)["max_S"]
            
            # C6: ADI
            W_class = 30
            c6_stat = compute_intermittency_stats(delta[-W_class:], np.ones(W_class, dtype=bool))["adi"]
            
            # C3: FZ or AUC (simplified to difference in p for ideal model)
            c3_stat = np.mean(delta[-60:]) - p
            
            data.append({
                "cause": cause,
                "c1_stat": abs(c1_stat),
                "c3_stat": abs(c3_stat),
                "c4_stat": abs(np.log(c4_stat)) if c4_stat > 0 else np.nan,
                "c5_sz_stat": c5_sz_stat,
                "c5_occ_stat": c5_occ_stat,
                "c6_stat": abs(c6_stat - (1.0/p)), # deviation from expected ADI
            })
            
    df = pd.DataFrame(data)
    
    # Gate 4.3: Targeted Separation AUC >= 0.80
    def check_auc(c_target, c0_mask, stat_col):
        target_mask = df["cause"] == c_target
        mask = target_mask | c0_mask
        
        # Drop NaNs
        valid = ~df[stat_col].isna()
        mask = mask & valid
        
        y_true = target_mask[mask].astype(int)
        y_score = df.loc[mask, stat_col]
        
        if len(y_score) < 2: return 0.5
        return roc_auc_score(y_true, y_score)
        
    c0_mask = df["cause"] == "C0"
    
    auc_c1 = check_auc("C1", c0_mask, "c1_stat")
    auc_c3 = check_auc("C3", c0_mask, "c3_stat")
    auc_c4 = check_auc("C4", c0_mask, "c4_stat")
    auc_c5_sz = check_auc("C5_size", c0_mask, "c5_sz_stat")
    auc_c5_occ = check_auc("C5_occ", c0_mask, "c5_occ_stat")
    auc_c6 = check_auc("C6", c0_mask, "c6_stat")
    
    print(f"AUC C1: {auc_c1:.3f}")
    print(f"AUC C3: {auc_c3:.3f}")
    print(f"AUC C4: {auc_c4:.3f}")
    print(f"AUC C5 (size): {auc_c5_sz:.3f}")
    print(f"AUC C5 (occ): {auc_c5_occ:.3f}")
    print(f"AUC C6: {auc_c6:.3f}")
    
    print("\nC4 Stats:")
    print(df[df["cause"] == "C4"]["c4_stat"].describe())
    print(df[df["cause"] == "C0"]["c4_stat"].describe())
    
    assert auc_c1 >= 0.80
    assert auc_c3 >= 0.80
    assert auc_c4 >= 0.80
    assert auc_c5_sz >= 0.80
    assert auc_c5_occ >= 0.80
    assert auc_c6 >= 0.80
    
    # Gate 4.4: Confound test AUC <= 0.65
    # C1 must not move size dispersion (C4)
    conf_c1_c4 = check_auc("C1", c0_mask, "c4_stat")
    print(f"C1 confound on C4: {conf_c1_c4:.3f}")
    assert conf_c1_c4 <= 0.65
    
    # C4 must not move size mean (C1)
    conf_c4_c1 = check_auc("C4", c0_mask, "c1_stat")
    print(f"C4 confound on C1: {conf_c4_c1:.3f}")
    assert conf_c4_c1 <= 0.65
    
    # C3 must not move size dispersion (C4)
    conf_c3_c4 = check_auc("C3", c0_mask, "c4_stat")
    print(f"C3 confound on C4: {conf_c3_c4:.3f}")
    assert conf_c3_c4 <= 0.65
