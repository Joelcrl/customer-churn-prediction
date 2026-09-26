"""
Preprocessing pipeline for raw Telco customer data.

Replicates notebooks/02_preprocessing.ipynb exactly, so raw input
(the original Kaggle format) can be scored by the trained model.
Order matters: encoding -> feature engineering (on raw values) -> scaling.
"""
import pandas as pd

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

BINARY_YES_NO = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

ADDON_SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']

# Allowed values per categorical column (exact spelling, case-sensitive)
RAW_SCHEMA = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    **{col: ['Yes', 'No', 'No internet service'] for col in ADDON_SERVICES},
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)'],
}

# Categories kept after pd.get_dummies(drop_first=True) in the notebook.
# The alphabetically first category of each column was dropped.
ONE_HOT_KEEP = {
    'MultipleLines': ['No phone service', 'Yes'],
    'InternetService': ['Fiber optic', 'No'],
    **{col: ['No internet service', 'Yes'] for col in ADDON_SERVICES},
    'Contract': ['One year', 'Two year'],
    'PaymentMethod': ['Credit card (automatic)', 'Electronic check', 'Mailed check'],
}

REQUIRED_COLUMNS = list(RAW_SCHEMA.keys()) + NUMERIC_FEATURES


def validate_raw(df_raw):
    """Return a list of human-readable error messages. Empty list means valid."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]

    if len(df_raw) == 0:
        return ["The file contains no rows."]

    errors = []
    for col, allowed in RAW_SCHEMA.items():
        if col == 'SeniorCitizen':
            values = pd.to_numeric(df_raw[col], errors='coerce')
            bad_mask = ~values.isin(allowed)
        else:
            values = df_raw[col].astype(str).str.strip()
            bad_mask = ~values.isin(allowed)
        if bad_mask.any():
            bad_values = sorted(set(df_raw.loc[bad_mask, col].astype(str)))[:5]
            errors.append(
                f"'{col}' has invalid values {bad_values} in {bad_mask.sum()} row(s). "
                f"Allowed: {allowed}"
            )

    for col in ['tenure', 'MonthlyCharges']:
        values = pd.to_numeric(df_raw[col], errors='coerce')
        bad_mask = values.isna() | (values < 0)
        if bad_mask.any():
            errors.append(f"'{col}' must be a non-negative number ({bad_mask.sum()} invalid row(s)).")

    return errors


def preprocess_raw(df_raw, scaler, feature_columns):
    """
    Convert raw Telco data into the model's feature format.

    feature_columns: the exact column order the model was trained on
    (pass X_test.columns). Selecting by this list guarantees correct order
    and raises a KeyError if any engineered column is missing.
    """
    df = df_raw.copy()
    for col in RAW_SCHEMA:
        if col != 'SeniorCitizen':
            df[col] = df[col].astype(str).str.strip()

    out = pd.DataFrame(index=df.index)

    # Binary encoding (same mapping as notebook)
    out['gender'] = (df['gender'] == 'Male').astype(int)
    out['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen']).astype(int)
    for col in BINARY_YES_NO:
        out[col] = (df[col] == 'Yes').astype(int)

    # Numeric (raw values). Blank TotalCharges -> 0, as in the notebook
    out['tenure'] = pd.to_numeric(df['tenure'])
    out['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'])
    out['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # One-hot encoding with an explicit category list, so a small upload
    # that lacks some categories still produces every training column
    for col, categories in ONE_HOT_KEEP.items():
        for cat in categories:
            out[f'{col}_{cat}'] = (df[col] == cat).astype(int)

    # Feature engineering on RAW values, before scaling
    out['avg_charge_per_tenure'] = out['TotalCharges'] / (out['tenure'] + 1)
    out['total_services'] = out[[f'{s}_Yes' for s in ADDON_SERVICES]].sum(axis=1)
    out['high_risk_combo'] = (
        (out['InternetService_Fiber optic'] == 1) &
        (out['PaymentMethod_Electronic check'] == 1)
    ).astype(int)

    out = out[list(feature_columns)]
    out[NUMERIC_FEATURES] = scaler.transform(out[NUMERIC_FEATURES])
    return out


def enforce_consistency(record):
    """Apply Telco business rules to a single-customer input dict."""
    record = dict(record)
    if record['InternetService'] == 'No':
        for col in ADDON_SERVICES:
            record[col] = 'No internet service'
    if record['PhoneService'] == 'No':
        record['MultipleLines'] = 'No phone service'
    return record


# Example rows (real rows from the original dataset) for the downloadable template
TEMPLATE_ROWS = pd.DataFrame([
    ['7590-VHVEG', 'Female', 0, 'Yes', 'No', 1, 'No', 'No phone service', 'DSL',
     'No', 'Yes', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes',
     'Electronic check', 29.85, 29.85],
    ['5575-GNVDE', 'Male', 0, 'No', 'No', 34, 'Yes', 'No', 'DSL',
     'Yes', 'No', 'Yes', 'No', 'No', 'No', 'One year', 'No',
     'Mailed check', 56.95, 1889.50],
    ['9237-HQITU', 'Female', 0, 'No', 'No', 2, 'Yes', 'No', 'Fiber optic',
     'No', 'No', 'No', 'No', 'No', 'No', 'Month-to-month', 'Yes',
     'Electronic check', 70.70, 151.65],
], columns=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'])
