import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

MODELS_DIR = Path(__file__).parent.parent / 'models'
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / 'final' / 'final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / 'final' / 'model_card.json', 'r') as f:
        model_card = json.load(f)
    with open(MODELS_DIR / 'final' / 'xgb_explainer_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    splits_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'splits'
    with open(splits_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(splits_dir / 'y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return model, scaler, model_card, xgb_model, X_test, y_test

model, scaler, model_card, xgb_model, X_test, y_test = load_artifacts()

@st.cache_resource
def get_shap_explainer(_xgb_model):
    return shap.TreeExplainer(_xgb_model)

explainer = get_shap_explainer(xgb_model)

def get_contract_type(row):
    if row.get('Contract_One year', 0) == 1:
        return 'One year'
    elif row.get('Contract_Two year', 0) == 1:
        return 'Two year'
    else:
        return 'Month-to-month'

def risk_badge(risk_label):
    color = "#ff4b4b" if risk_label == "High Risk" else "#09ab3b"
    return f'<span style="background-color:{color}; color:white; padding:3px 10px; border-radius:12px; font-size:14px;">{risk_label}</span>'

@st.cache_data
def build_display_dataset(_X_test):
    display_df = _X_test.copy()
    display_df[NUMERIC_FEATURES] = scaler.inverse_transform(_X_test[NUMERIC_FEATURES])
    display_df['contract_type'] = _X_test.apply(get_contract_type, axis=1)
    return display_df

X_test_display = build_display_dataset(X_test)

@st.cache_data
def compute_shap(_customer_data, _idx):
    return explainer.shap_values(_customer_data)

# ===== Sidebar =====
st.sidebar.title("Settings")
threshold = st.sidebar.slider(
    "Churn risk threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Customers with predicted churn probability above this value are classified as high-risk."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
st.sidebar.caption("Applied to Summary and Individual Analysis tabs.")

contract_options = ['Month-to-month', 'One year', 'Two year']
selected_contracts = st.sidebar.multiselect(
    "Contract type", options=contract_options, default=contract_options
)

tenure_min = int(X_test_display['tenure'].min())
tenure_max = int(X_test_display['tenure'].max())
tenure_range = st.sidebar.slider(
    "Tenure range (months)",
    min_value=tenure_min, max_value=tenure_max,
    value=(tenure_min, tenure_max)
)

filter_mask = (
    X_test_display['contract_type'].isin(selected_contracts) &
    X_test_display['tenure'].between(tenure_range[0], tenure_range[1])
)
X_filtered = X_test[filter_mask]
X_filtered_display = X_test_display[filter_mask]

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(f"""
**Model:** {model_card['model_type']}

**Test ROC-AUC:** {model_card['performance']['roc_auc_test']}

**Trained on:** {model_card['created_date']}
""")

# ===== Main =====
st.title("Customer Churn Prediction Dashboard")
st.markdown("Identify at-risk customers and understand the drivers behind churn predictions.")

if len(X_filtered) == 0:
    st.warning("No customers match the current filters. Adjust the sidebar filters to see results.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Summary", "📁 Batch Prediction", "🔍 Individual Analysis"])

# ===== Tab 1: Summary =====
with tab1:
    proba_filtered = model.predict_proba(X_filtered)[:, 1]
    high_risk_mask = proba_filtered >= threshold
    n_high_risk = high_risk_mask.sum()

    monthly_charges = X_filtered_display['MonthlyCharges'].values
    revenue_at_risk = monthly_charges[high_risk_mask].sum()

    high_risk_pct = n_high_risk / len(X_filtered) * 100
    delta_color = "inverse" if high_risk_pct > 20 else "normal"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers (Filtered)", f"{len(X_filtered):,}")
    col2.metric(
        "High-Risk Customers", f"{n_high_risk:,}",
        f"{high_risk_pct:.1f}% of filtered",
        delta_color=delta_color
    )
    col3.metric("Estimated Revenue at Risk", f"${revenue_at_risk:,.0f}/month")
    col4.metric("Model ROC-AUC", f"{model_card['performance']['roc_auc_test']:.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(proba_filtered, bins=30, color='steelblue', edgecolor='white')
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        ax.set_xlabel('Predicted Churn Probability')
        ax.set_ylabel('Number of Customers')
        ax.legend()
        st.pyplot(fig)

    with col_right:
        st.subheader("Churn Rate by Tenure")
        tenure_buckets = pd.cut(
            X_filtered_display['tenure'],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=['0-6mo', '6-12mo', '12-24mo', '24-48mo', '48-72mo']
        )
        churn_pred = (proba_filtered >= threshold).astype(int)
        bucket_churn = pd.DataFrame({'bucket': tenure_buckets, 'predicted_churn': churn_pred}) \
            .groupby('bucket', observed=True)['predicted_churn'].mean() * 100
        bucket_churn = bucket_churn.dropna()

        if len(bucket_churn) == 0:
            st.info("Not enough tenure variation in the filtered data to show this chart.")
        else:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            bucket_churn.plot(kind='bar', ax=ax2, color='crimson')
            ax2.set_ylabel('Predicted Churn Rate (%)')
            ax2.set_xlabel('Tenure Bucket')
            plt.xticks(rotation=0)
            st.pyplot(fig2)

    st.divider()
    st.subheader("Retention Campaign Simulator")
    retention_rate = st.slider(
        "If a retention campaign successfully prevents this % of high-risk customers from churning:",
        min_value=5, max_value=50, value=15, step=5
    )
    revenue_saved = revenue_at_risk * (retention_rate / 100)
    st.metric(
        f"Estimated Monthly Revenue Saved (at {retention_rate}% retention success)",
        f"${revenue_saved:,.0f}/month"
    )
    st.caption("Calculated as: Revenue at Risk x assumed retention success rate. This is a simplified estimate for illustration, not a guaranteed outcome.")

    st.divider()
    st.subheader("Top Churn Drivers (SHAP)")
    top_features_df = pd.DataFrame({'feature': model_card['top_features_shap']})
    st.dataframe(top_features_df, hide_index=True, use_container_width=True)
    st.caption("Based on mean absolute SHAP values from the underlying XGBoost model.")

    st.divider()
    st.subheader("Model Performance Metrics")
    perf = model_card['performance']
    metrics_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [perf['roc_auc_test'], perf['accuracy'], perf['precision'], perf['recall'], perf['f1']]
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with st.expander("About this model"):
        st.markdown(f"""
        **Model type:** {model_card['model_type']}

        **Dataset:** {model_card['dataset']}

        **Train size:** {model_card['train_size']:,} customers | **Test size:** {model_card['test_size']:,} customers

        **Number of features:** {model_card['num_features']}

        **Known limitations:**
        """)
        for limitation in model_card['known_limitations']:
            st.markdown(f"- {limitation}")

# ===== Tab 2: Batch Prediction =====
with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV with customer data to predict churn risk for multiple customers at once.")
    st.caption("Note: sidebar filters do not apply to this tab, as it works on independently uploaded or sampled data.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        use_sample = st.button("Load sample data")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if use_sample:
        st.session_state['batch_df'] = X_test.sample(50, random_state=42).reset_index(drop=True)
        st.success(f"Loaded {len(st.session_state['batch_df'])} sample customers from the test set.")
    elif uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            missing_cols = set(X_test.columns) - set(temp_df.columns)
            if missing_cols:
                st.error(f"Uploaded file is missing required columns: {missing_cols}")
            else:
                st.session_state['batch_df'] = temp_df[X_test.columns]
                st.success(f"Loaded {len(st.session_state['batch_df'])} customers from uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    batch_df = st.session_state.get('batch_df', None)

    if batch_df is not None:
        try:
            with st.spinner("Generating predictions..."):
                batch_proba = model.predict_proba(batch_df)[:, 1]
        except Exception as e:
            st.error(f"Could not generate predictions from this file. Make sure it matches the expected preprocessed format. Details: {e}")
            batch_proba = None

        if batch_proba is not None:
            batch_display = batch_df.copy()
            batch_display[NUMERIC_FEATURES] = scaler.inverse_transform(batch_df[NUMERIC_FEATURES])

            batch_results = batch_display.copy()
            batch_results.insert(0, 'churn_probability', batch_proba.round(3))
            batch_results.insert(1, 'risk_level', np.where(
                batch_proba >= threshold, 'High Risk', 'Low Risk'
            ))
            batch_results = batch_results.sort_values('churn_probability', ascending=False)

            col1, col2, col3 = st.columns(3)
            n_high = (batch_proba >= threshold).sum()
            col1.metric("Total Customers", len(batch_results))
            col2.metric("High-Risk Count", n_high)
            col3.metric("Estimated Revenue at Risk",
                         f"${batch_results.loc[batch_results['risk_level']=='High Risk', 'MonthlyCharges'].sum():,.0f}/mo")

            st.divider()

            risk_filter = st.selectbox("Filter by risk level", ["All", "High Risk", "Low Risk"])
            display_df = batch_results if risk_filter == "All" else batch_results[batch_results['risk_level'] == risk_filter]

            st.dataframe(
                display_df[['churn_probability', 'risk_level', 'tenure', 'MonthlyCharges', 'TotalCharges']],
                use_container_width=True,
                hide_index=True
            )

            csv_output = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download results as CSV",
                data=csv_output,
                file_name='churn_predictions.csv',
                mime='text/csv'
            )
    else:
        st.info("Upload a CSV or click 'Load sample data' to see predictions.")

# ===== Tab 3: Individual Analysis =====
def get_recommendations(shap_values_row, feature_names, top_n=3):
    feature_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_row
    })
    risk_increasing = feature_shap[feature_shap['shap_value'] > 0].sort_values('shap_value', ascending=False)

    recommendation_map = {
        'PaymentMethod_Electronic check': 'Encourage switch to automatic payment (credit card or bank transfer), which shows notably lower churn.',
        'Contract_One year': 'Offer incentive to upgrade to a Two-year contract for additional discount and lock-in.',
        'InternetService_Fiber optic': 'Review pricing or service quality complaints specific to Fiber optic customers.',
        'MonthlyCharges': 'Consider a loyalty discount or bundle offer to offset high monthly charges.',
        'TotalCharges': 'Review long-term value perception; consider a re-engagement offer.',
        'PaperlessBilling': 'Confirm customer is comfortable with digital billing; offer support if needed.',
        'tenure': 'Early-tenure customer: prioritize onboarding check-ins and early engagement.',
        'avg_charge_per_tenure': 'High spend relative to tenure: monitor for early dissatisfaction signals.',
    }

    recommendations = []
    for _, row in risk_increasing.head(top_n).iterrows():
        feature = row['feature']
        if feature in recommendation_map:
            rec = recommendation_map[feature]
        elif feature.startswith('OnlineSecurity') or feature.startswith('TechSupport'):
            rec = 'Offer a free trial of Online Security / Tech Support add-ons to increase engagement.'
        else:
            continue

        if rec not in recommendations:
            recommendations.append(rec)

    if not recommendations:
        recommendations.append('No strong risk-increasing factors identified; continue standard engagement.')

    return recommendations

with tab3:
    st.subheader("Individual Customer Analysis")
    st.markdown("Select a customer to see their churn risk and the factors driving the prediction.")
    st.caption("Respects the contract type and tenure filters set in the sidebar.")

    proba_filtered_tab3 = model.predict_proba(X_filtered)[:, 1]
    options_df = pd.DataFrame({'index': X_filtered.index, 'proba': proba_filtered_tab3})

    risk_category = st.radio(
        "Show customers:",
        ["Highest risk", "Lowest risk", "All"],
        horizontal=True
    )

    if risk_category == "Highest risk":
        filtered_options = options_df.nlargest(min(20, len(options_df)), 'proba')
    elif risk_category == "Lowest risk":
        filtered_options = options_df.nsmallest(min(20, len(options_df)), 'proba')
    else:
        filtered_options = options_df

    filtered_options = filtered_options.copy()
    filtered_options['label'] = filtered_options.apply(
        lambda row: f"Customer #{row['index']} — {row['proba']:.0%} churn risk", axis=1
    )

    selected_label = st.selectbox(
        "Select customer",
        options=filtered_options['label'].tolist(),
        index=0
    )
    customer_idx = filtered_options.loc[filtered_options['label'] == selected_label, 'index'].values[0]

    customer_data = X_test.loc[[customer_idx]]
    customer_display = customer_data.copy()
    customer_display[NUMERIC_FEATURES] = scaler.inverse_transform(customer_data[NUMERIC_FEATURES])

    customer_proba = model.predict_proba(customer_data)[0, 1]
    risk_label = "High Risk" if customer_proba >= threshold else "Low Risk"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Churn Probability", f"{customer_proba:.1%}")
        st.metric("Risk Level", risk_label)
        st.metric("Tenure", f"{customer_display['tenure'].values[0]:.0f} months")
        st.metric("Monthly Charges", f"${customer_display['MonthlyCharges'].values[0]:.2f}")

    with col2:
        st.markdown("**Why this prediction?**")
        xgb_proba = xgb_model.predict_proba(customer_data)[0, 1]
        customer_shap_values = compute_shap(customer_data, customer_idx)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(
            shap.Explanation(
                values=customer_shap_values[0],
                base_values=explainer.expected_value,
                data=customer_display.iloc[0],
                feature_names=customer_data.columns.tolist()
            ),
            max_display=8,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(f"SHAP explanation is based on the XGBoost base model (proxy for the ensemble). XGBoost's own churn probability for this customer: {xgb_proba:.1%}")

    st.divider()
    st.subheader("Recommended Actions")
    recommendations = get_recommendations(
        customer_shap_values[0], customer_data.columns.tolist()
    )
    for rec in recommendations:
        st.markdown(f"- {rec}")
    st.caption("Recommendations are derived from the top risk-increasing SHAP factors for this customer and are illustrative, not a substitute for retention team judgment.")

    st.divider()
    with st.expander("View raw customer data"):
        st.dataframe(customer_display.T, use_container_width=True)

st.divider()
st.caption("Built by Joel Chriscendo Rahardjo Liem | [GitHub Repository](https://github.com/Joelcrl/customer-churn-prediction)")