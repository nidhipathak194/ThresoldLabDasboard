import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Human-readable domain mapping for anonymized PCA components V1-V28
PCA_FEATURE_DESCRIPTIONS = {
    'V1': 'Cardholder Age & Account Tenure Proxy',
    'V2': 'Transaction Location Disparity Index',
    'V3': 'Merchant Risk Category Score',
    'V4': 'Transaction Velocity (1-Hour Window)',
    'V5': 'Card-Not-Present (CNP) Verification Flag',
    'V6': 'Device IP Geolocation Anomaly Score',
    'V7': 'Historical Average Spending Divergence',
    'V8': 'Cross-Border Currency Conversion Metric',
    'V9': 'Billing/Shipping Address Match Rating',
    'V10': 'High-Risk MCC Category Indicator',
    'V11': 'Failed PIN / Auth Attempt Counter',
    'V12': 'Behavioral Biometrics Typing/Click Rate',
    'V13': 'Terminal POS Technology Signature',
    'V14': 'Transaction Velocity Anomaly (Primary Signal)',
    'V15': 'Tokenized Mobile Wallet Risk Factor',
    'V16': 'Card Issuer Country Risk Weight',
    'V17': 'Geolocation Distance Delta (Primary Signal)',
    'V18': 'Recurring Subscription Deviation Rating',
    'V19': 'Time Since Last High-Value Purchase',
    'V20': 'Account Balance Ratio Post-Transaction',
    'V21': 'Network IP Subnet Reputation Score',
    'V22': 'Customer Support Inquiry Spike Factor',
    'V23': 'Virtual Card Generation Flag Index',
    'V24': 'Midnight/Off-Hours Transaction Indicator',
    'V25': 'Atmospheric Session Fingerprint Variance',
    'V26': 'Peer-Group Expenditure Standard Deviation',
    'V27': 'Multi-Card Binding Ratio Metric',
    'V28': 'Behavioral Entropy Anomaly Score'
}

def generate_synthetic_ulb_dataset(n_samples: int = 50000, fraud_rate: float = 0.00172, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a benchmark Credit Card Fraud dataset replicating the statistical properties
    of the Université Libre de Bruxelles (ULB) dataset (Dal Pozzolo et al., 2015).
    
    Parameters:
    - n_samples: Total number of transactions (default 50,000 for fast interactive execution).
    - fraud_rate: Proportion of fraudulent transactions (0.172% matching ULB benchmark).
    - random_state: Seed for reproducibility.
    
    Returns:
    - pd.DataFrame containing Time, Amount, V1-V28, and Class (0=Genuine, 1=Fraud).
    """
    np.random.seed(random_state)
    n_fraud = int(round(n_samples * fraud_rate))
    n_fraud = max(n_fraud, 20)  # ensure at least 20 fraud samples for calibration
    n_genuine = n_samples - n_fraud

    # 1. Time feature (seconds elapsed over 2 days = 172,800 seconds)
    time_genuine = np.random.uniform(0, 172800, n_genuine)
    time_fraud = np.random.uniform(0, 172800, n_fraud)
    time = np.concatenate([time_genuine, time_fraud])

    # 2. Amount feature (Heavy-tailed log-normal distribution, higher mean for frauds)
    amount_genuine = np.random.lognormal(mean=3.5, sigma=1.2, size=n_genuine)
    amount_fraud = np.random.lognormal(mean=4.2, sigma=1.4, size=n_fraud)
    amount = np.concatenate([amount_genuine, amount_fraud])

    # 3. Anonymized PCA Features (V1 to V28)
    n_features = 28
    v_genuine = np.random.normal(loc=0.0, scale=1.0, size=(n_genuine, n_features))
    
    # Key predictive PCA features in ULB dataset (V4, V11, V12, V14, V17 show strong separation)
    v_fraud = np.random.normal(loc=0.0, scale=1.0, size=(n_fraud, n_features))
    # Shift prominent features to simulate true fraud signals
    v_fraud[:, 3] += np.random.normal(1.8, 0.5, n_fraud)   # V4 positive shift
    v_fraud[:, 10] += np.random.normal(1.5, 0.4, n_fraud)  # V11 positive shift
    v_fraud[:, 11] -= np.random.normal(2.2, 0.6, n_fraud)  # V12 negative shift
    v_fraud[:, 13] -= np.random.normal(2.5, 0.7, n_fraud)  # V14 negative shift
    v_fraud[:, 16] -= np.random.normal(2.1, 0.6, n_fraud)  # V17 negative shift

    v_all = np.vstack([v_genuine, v_fraud])
    
    # Target Class: 0 = Genuine, 1 = Fraud
    y = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(n_fraud, dtype=int)])

    # Construct DataFrame
    cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X_mat = np.column_stack([time, amount, v_all])
    df = pd.DataFrame(X_mat, columns=cols)
    df['Class'] = y

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Applies Stratified Train/Test split and Standard Scaling to Time and Amount features,
    preserving pre-transformed V1-V28 features.
    
    Returns:
    - X_train, X_test, y_train, y_test, scaler
    """
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standard Scale only Time and Amount (V1-V28 are already PCA transformed)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def load_custom_csv(filepath_or_buffer):
    """
    Loads and validates a custom CSV file (e.g. the authentic Kaggle/ULB
    creditcard.csv) so it is guaranteed to fit the shape Threshold Lab's
    pipeline expects: exactly Time, Amount, V1..V28, Class, numeric,
    complete, and deduplicated.

    Processing steps applied, in order:
      1. Read the CSV.
      2. Strip whitespace from column names and match them to the
         canonical Time/Amount/V1-V28/Class names case-insensitively,
         so minor header formatting differences don't cause a hard failure.
      3. Verify all required columns are present (hard failure if not,
         since there's nothing safe to substitute for a missing feature).
      4. Drop extra/unexpected columns (e.g. a stray "Unnamed: 0" index
         column some CSV exporters add) and fix column order.
      5. Coerce Time, Amount, V1-V28 to numeric and Class to a 0/1 integer
         label; any value that fails to coerce becomes NaN.
      6. Drop rows with missing values in any required column.
      7. Drop exact duplicate rows (the ULB dataset is known to contain
         some, and duplicates would leak between train/test splits).
      8. Reset the index.

    Returns:
        (df, report) where df is the cleaned DataFrame and report is a
        dict describing what the cleaning step did, suitable for showing
        the user a transparent before/after summary.
    """
    df = pd.read_csv(filepath_or_buffer)
    rows_read = len(df)

    # --- Step 2: tolerant column matching -----------------------------
    required_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    rename_map = {}
    for canonical in required_cols:
        if canonical in df.columns:
            continue
        match = lower_map.get(canonical.lower())
        if match is not None:
            rename_map[match] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # --- Step 3: hard-fail only on genuinely missing columns -----------
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV file is missing required columns: {missing}. "
            f"Threshold Lab needs exactly Time, Amount, V1-V28, and Class "
            f"(case-insensitive) to run its pipeline."
        )

    # --- Step 4: drop extras, fix column order --------------------------
    dropped_extra_cols = [c for c in df.columns if c not in required_cols]
    df = df[required_cols].copy()

    # --- Step 5: coerce dtypes ------------------------------------------
    numeric_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
    # Anything that isn't exactly 0 or 1 after coercion is invalid.
    df.loc[~df['Class'].isin([0, 1]), 'Class'] = np.nan

    # --- Step 6: drop incomplete rows ------------------------------------
    rows_before_na_drop = len(df)
    df = df.dropna(subset=required_cols)
    invalid_rows_removed = rows_before_na_drop - len(df)

    # --- Step 7: drop exact duplicates -----------------------------------
    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before_dedup - len(df)

    # --- Step 8: finalize --------------------------------------------------
    df['Class'] = df['Class'].astype(int)
    df = df.reset_index(drop=True)

    fraud_count = int(df['Class'].sum())
    total_rows = len(df)
    report = {
        'rows_read': rows_read,
        'rows_final': total_rows,
        'dropped_extra_columns': dropped_extra_cols,
        'renamed_columns': rename_map,
        'invalid_rows_removed': invalid_rows_removed,
        'duplicates_removed': duplicates_removed,
        'fraud_count': fraud_count,
        'genuine_count': total_rows - fraud_count,
        'fraud_rate_pct': round((fraud_count / total_rows * 100), 4) if total_rows else 0.0,
    }

    if total_rows == 0:
        raise ValueError(
            "No valid rows remained after cleaning this CSV — check that Time, "
            "Amount, V1-V28, and Class all contain numeric data and that Class "
            "is binary (0/1)."
        )
    if fraud_count == 0:
        raise ValueError(
            "This CSV contains no fraud examples (Class = 1) after cleaning, "
            "so the classifiers cannot be trained or calibrated on it."
        )

    return df, report

def detect_local_dataset(search_dirs=None) -> str:
    """
    Searches for local Kaggle creditcard.csv in common paths.
    """
    if search_dirs is None:
        search_dirs = ['.', 'data', 'dataset']
    for d in search_dirs:
        candidate = os.path.join(d, 'creditcard.csv')
        if os.path.isfile(candidate):
            return candidate
    return None

