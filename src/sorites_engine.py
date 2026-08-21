import numpy as np
import pandas as pd

def compute_elkan_threshold(c_fp: float, c_fn: float) -> float:
    """
    Computes the optimal expected-cost-minimizing threshold tau* according to Elkan (2001):
    tau* = C(FP) / (C(FP) + C(FN))
    """
    if c_fp <= 0 or c_fn <= 0:
        raise ValueError("Costs must be strictly positive numbers.")
    return c_fp / (c_fp + c_fn)

def evaluate_threshold_cost(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    threshold: float,
    c_fp: float,
    c_fn: float,
    amounts: np.ndarray = None
) -> dict:
    """
    Evaluates classification confusion matrix and total financial loss at a given threshold tau.
    """
    y_pred = (y_probas >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total_cost = fp * c_fp + fn * c_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        'threshold': threshold,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'total_cost': total_cost,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    if amounts is not None:
        # Amount lost to undetected frauds (FNs) and cost of reviewing FPs
        fn_amount_loss = float(np.sum(amounts[(y_pred == 0) & (y_true == 1)]))
        result['fn_amount_loss'] = fn_amount_loss

    return result

def analyze_contested_zone(
    df_test: pd.DataFrame,
    y_probas: np.ndarray,
    c_fp_range: tuple,
    c_fn: float
) -> dict:
    """
    Identifies transactions in the 'Contested Zone'—those whose classification flips
    depending on which reasonable cost ratio (C(FP) min vs max) is chosen.
    
    Parameters:
    - c_fp_range: Tuple of (c_fp_min, c_fp_max)
    - c_fn: False Negative cost baseline
    """
    c_fp_min, c_fp_max = c_fp_range
    tau_low = compute_elkan_threshold(c_fp_min, c_fn)
    tau_high = compute_elkan_threshold(c_fp_max, c_fn)

    # Make sure tau_min is lower boundary, tau_max is upper boundary
    tau_min, tau_max = min(tau_low, tau_high), max(tau_low, tau_high)

    # Contested condition: tau_min <= proba < tau_max
    is_contested = (y_probas >= tau_min) & (y_probas < tau_max)
    contested_count = int(np.sum(is_contested))
    total_count = len(y_probas)
    proportion = contested_count / total_count if total_count > 0 else 0.0

    contested_df = df_test[is_contested].copy()
    contested_df['predicted_proba'] = y_probas[is_contested]

    contested_amount_total = float(contested_df['Amount'].sum()) if 'Amount' in contested_df.columns else 0.0
    contested_frauds = int(np.sum(contested_df['Class'] == 1)) if 'Class' in contested_df.columns else 0

    return {
        'tau_min': tau_min,
        'tau_max': tau_max,
        'contested_count': contested_count,
        'total_count': total_count,
        'proportion': proportion,
        'contested_amount_total': contested_amount_total,
        'contested_frauds': contested_frauds,
        'contested_df': contested_df
    }

def find_sorites_twin_transactions(
    df_test: pd.DataFrame,
    y_probas: np.ndarray,
    tau_star: float,
    epsilon: float = 0.015
) -> dict:
    """
    Finds two real or near-identical transactions straddling tau* to demonstrate Eubulides' Sorites Paradox.
    Transaction A has p(x) = tau* - epsilon -> Genuine label
    Transaction B has p(x) = tau* + epsilon -> Fraud label
    """
    below_idx = np.where((y_probas >= tau_star - epsilon) & (y_probas < tau_star))[0]
    above_idx = np.where((y_probas >= tau_star) & (y_probas <= tau_star + epsilon))[0]

    if len(below_idx) > 0 and len(above_idx) > 0:
        idx_a = below_idx[0]
        idx_b = above_idx[0]

        tx_a = df_test.iloc[idx_a].to_dict()
        tx_b = df_test.iloc[idx_b].to_dict()

        proba_a = float(y_probas[idx_a])
        proba_b = float(y_probas[idx_b])
    else:
        # Fallback synthetic demonstration pair if exact range has sparse observations
        sample_row = df_test.iloc[0].to_dict()
        tx_a = sample_row.copy()
        tx_b = sample_row.copy()

        proba_a = max(0.001, tau_star - 0.005)
        proba_b = min(0.999, tau_star + 0.005)
        tx_a['Amount'] = round(float(tx_a.get('Amount', 100.0)), 2)
        tx_b['Amount'] = round(float(tx_b.get('Amount', 100.0)) + 0.50, 2)

    return {
        'tx_a': tx_a,
        'tx_b': tx_b,
        'proba_a': proba_a,
        'proba_b': proba_b,
        'diff': abs(proba_b - proba_a),
        'label_a': 'GENUINE (PASS)' if proba_a < tau_star else 'FRAUD (FLAGGED)',
        'label_b': 'GENUINE (PASS)' if proba_b < tau_star else 'FRAUD (FLAGGED)',
        'tau_star': tau_star
    }

def run_cost_ratio_sweep(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    c_fn: float = 1000.0,
    n_points: int = 50
) -> pd.DataFrame:
    """
    Sweeps C(FP) from small to large to demonstrate how tau* and total expected cost evolve.
    """
    c_fp_values = np.linspace(1.0, 500.0, n_points)
    records = []

    for c_fp in c_fp_values:
        tau = compute_elkan_threshold(c_fp, c_fn)
        cost_info = evaluate_threshold_cost(y_true, y_probas, tau, c_fp, c_fn)
        records.append({
            'C_FP': c_fp,
            'C_FN': c_fn,
            'Cost_Ratio': c_fp / c_fn,
            'Tau_Star': tau,
            'Total_Cost': cost_info['total_cost'],
            'FP': cost_info['FP'],
            'FN': cost_info['FN'],
            'Precision': cost_info['precision'],
            'Recall': cost_info['recall'],
            'F1': cost_info['f1_score']
        })

    return pd.DataFrame(records)

def run_threshold_cost_curve(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    c_fp: float,
    c_fn: float,
    n_thresholds: int = 100
) -> pd.DataFrame:
    """
    Evaluates Total Expected Cost across decision thresholds tau in [0.001, 0.999]
    to show the convex minimum at Elkan's tau*.
    """
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    records = []

    for tau in thresholds:
        info = evaluate_threshold_cost(y_true, y_probas, tau, c_fp, c_fn)
        records.append({
            'Threshold': tau,
            'Total_Cost': info['total_cost'],
            'FP': info['FP'],
            'FN': info['FN'],
            'FP_Cost': info['FP'] * c_fp,
            'FN_Cost': info['FN'] * c_fn,
            'Precision': info['precision'],
            'Recall': info['recall']
        })

    return pd.DataFrame(records)

