import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD TRAINED MODEL
# =========================
model = joblib.load("random_forest.pkl")
THRESHOLD = 0.2

# =========================
# PREDEFINED PCA PROFILES
# (Derived from real clusters)
# =========================
PCA_PROFILES = {
    "Normal Purchase": np.array([
        0.02, -0.01, 0.03, -0.04, 0.01, 0.02, -0.02, 0.01,
        0.00, 0.01, -0.01, 0.02, 0.03, 0.01, 0.00, -0.01,
        0.01, 0.00, 0.02, -0.01, 0.01, 0.02, 0.01, 0.01,
        0.02, 0.01, 0.01, 0.02
    ]),
    "Suspicious Pattern": np.array([
        -0.9, 1.2, -0.7, 1.5, -0.4, -0.9, -1.1, 0.8,
        -1.3, -1.0, 1.2, -1.1, -0.5, -1.6, 0.2, -0.8,
        -1.2, 0.0, 0.4, 0.6, 0.5, -0.2, -0.6, 0.4,
        0.1, 0.3, 0.4, -0.2
    ]),
    "Highly Fraudulent Pattern": np.array([
        -2.3, 1.9, -1.6, 4.0, -0.5, -1.4, -2.5, 1.4,
        -2.8, -2.7, 3.2, -2.9, -0.6, -4.3, 0.4, -1.1,
        -2.8, 0.0, 0.2, 0.4, 0.6, -0.1, -0.5, 0.3,
        0.0, 0.2, 0.3, -0.1
    ])
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# UI HEADER
# =========================
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("**Random Forest–based fraud analysis using trained ML model**")

st.info(
    f"**Model:** Random Forest  \n"
    f"**Threshold:** {THRESHOLD}  \n"
    f"**Features:** 29 (V1–V28 + Amount)"
)

# =========================
# USER INPUT (ABSTRACTED UI)
# =========================
st.markdown("### 🔍 Transaction Risk Simulator")

profile = st.radio(
    "Transaction Profile",
    ["Normal Purchase", "Suspicious Pattern", "Highly Fraudulent Pattern"],
    horizontal=True
)

amount = st.number_input(
    "Transaction Amount ($)",
    min_value=1.0,
    max_value=100000.0,
    value=200.0,
    step=50.0
)

intensity = st.slider(
    "Risk Intensity",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    help="Higher values simulate more extreme behavior"
)

# =========================
# BUILD MODEL INPUT
# =========================
base_pca = PCA_PROFILES[profile]
pca_values = base_pca * intensity

# Same amount scaling logic used earlier
amount_scaled = np.log1p(amount) / 10

X = np.concatenate([pca_values, [amount_scaled]]).reshape(1, -1)

# =========================
# PREDICTION
# =========================
if st.button("🔎 Analyze Transaction", use_container_width=True):
    prob = model.predict_proba(X)[0][1]

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fraud Probability", f"{prob:.1%}")

    with col2:
        risk_level = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
        st.metric("Risk Level", risk_level)

    with col3:
        confidence = abs(prob - 0.5) * 2
        st.metric("Confidence", f"{confidence:.1%}")

    st.progress(float(prob))

    # =========================
    # FINAL DECISION
    # =========================
    if prob >= THRESHOLD:
        st.error(
            f"🚨 **FRAUDULENT TRANSACTION DETECTED**\n\n"
            f"Fraud Probability: **{prob:.1%}**\n\n"
            f"**Recommended Action:** Block transaction and verify user."
        )
    else:
        st.success(
            f"✅ **LEGITIMATE TRANSACTION**\n\n"
            f"Legitimacy Probability: **{(1 - prob):.1%}**\n\n"
            f"**Recommended Action:** Approve transaction."
        )

    with st.expander("📘 Decision Logic"):
        st.markdown(
            f"""
            **Threshold Used:** {THRESHOLD}

            - Probability < 0.3 → Low risk  
            - Probability 0.3–0.7 → Medium risk (review)  
            - Probability > 0.7 → High risk (block)

            This abstraction preserves model integrity while avoiding direct PCA exposure.
            """
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🛡️ Built with Streamlit | Uses trained Random Forest model only")
