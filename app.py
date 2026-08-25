import os
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Retention Support",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "final_churn_model.pkl"

# These are the exact feature columns used by the notebook after pd.get_dummies(drop_first=True).
EXPECTED_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

RAW_REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def prepare_input(raw_df):
    """
    Convert raw customer fields into the same dummy-variable structure
    used by the training notebook.
    """
    data = raw_df.copy()

    # Standardise numeric fields.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Use fixed category levels so pd.get_dummies creates the same baseline
    # categories as the notebook.
    for col, levels in CATEGORY_LEVELS.items():
        data[col] = pd.Categorical(data[col], categories=levels)

    encoded = pd.get_dummies(
        data[RAW_REQUIRED_COLUMNS],
        columns=list(CATEGORY_LEVELS.keys()),
        drop_first=True,
        dtype=int,
    )

    encoded = encoded.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    return encoded


def risk_level(score):
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def priority_label(level):
    if level == "High":
        return "Priority 1"
    if level == "Medium":
        return "Priority 2"
    return "Priority 3"


def recommended_action(level):
    if level == "High":
        return (
            "Contact the customer promptly, review service concerns and current "
            "package fit, and consider a targeted retention incentive."
        )
    if level == "Medium":
        return (
            "Use proactive engagement, review the customer's service package and "
            "monitor for additional signs of churn risk."
        )
    return (
        "No immediate retention intervention is required. Maintain normal service "
        "quality and loyalty engagement."
    )


def build_profile_flags(customer):
    """
    Simple business profile flags.
    These are not model explanations and should not be treated as causal factors.
    """
    flags = []

    if customer["Contract"] == "Month-to-month":
        flags.append("Month-to-month contract")

    if float(customer["tenure"]) < 12:
        flags.append("Short customer tenure")

    if float(customer["MonthlyCharges"]) >= 80:
        flags.append("Relatively high monthly charges")

    if customer["PaymentMethod"] == "Electronic check":
        flags.append("Electronic check payment method")

    if customer["TechSupport"] == "No":
        flags.append("No technical support service")

    if customer["OnlineSecurity"] == "No":
        flags.append("No online security service")

    if not flags:
        flags.append("No major profile flag triggered by the current business rules")

    return flags


def validate_batch_data(df):
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, ""


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Customer Retention Management System")
st.caption(
    "Use churn risk scores to identify customers who may need retention attention "
    "and prioritise follow-up actions."
)

if model is None:
    st.error(
        "Model file not found. Place `final_churn_model.pkl` in the same GitHub "
        "folder as `app.py`, then restart the app."
    )
    st.stop()


# ------------------------------------------------------------
# Main navigation
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Customer Risk Assessment",
        "📊 Retention Management Dashboard",
        "ℹ️ Model Decision Explanation",
    ]
)


# ------------------------------------------------------------
# Tab 1: Single customer
# ------------------------------------------------------------
if page == "Customer Assessment":
    st.subheader("Customer Profile Assessment")
    st.write(
        "Enter the customer's current profile. The model will return a churn risk "
        "score and an operational retention priority."
    )

    with st.form("single_customer_form"):
        st.markdown("### Customer Relationship")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", CATEGORY_LEVELS["gender"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = "No"
            dependents = "No"
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12, step=1
            )
            contract = st.selectbox("Contract", CATEGORY_LEVELS["Contract"])

        with col2:
            phone = "Yes"
            multiple_lines = "No"
            internet = st.selectbox(
                "Internet Service", CATEGORY_LEVELS["InternetService"]
            )
            online_security = st.selectbox(
                "Online Security", CATEGORY_LEVELS["OnlineSecurity"]
            )
            online_backup = st.selectbox(
                "Online Backup", CATEGORY_LEVELS["OnlineBackup"]
            )
            tech_support = st.selectbox(
                "Tech Support", CATEGORY_LEVELS["TechSupport"]
            )

        with col3:
            device_protection = st.selectbox(
                "Device Protection", CATEGORY_LEVELS["DeviceProtection"]
            )
            streaming_tv = "No"
            streaming_movies = "No"
            paperless = "No"
            payment = st.selectbox(
                "Payment Method", CATEGORY_LEVELS["PaymentMethod"]
            )
            monthly = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0, step=1.0
            )
            total = st.number_input(
                "Total Charges", min_value=0.0, value=840.0, step=10.0
            )

        submitted = st.form_submit_button("Analyse Customer")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        customer_df = pd.DataFrame([customer])
        encoded_customer = prepare_input(customer_df)

        score = float(model.predict_proba(encoded_customer)[0, 1])
        level = risk_level(score)
        priority = priority_label(level)

        st.divider()
        st.subheader("Customer Risk Assessment")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Risk Score", f"{score:.1%}")
        metric2.metric("Risk Level", level)
        metric3.metric("Retention Priority", priority)

        st.progress(min(max(score, 0.0), 1.0))

        st.markdown("### Retention Recommendation")
        st.write(recommended_action(level))

        st.markdown("### Customer Risk Indicators")
        st.caption(
            "These profile flags are simple business rules for interpretation. "
            "They are not model feature explanations and do not prove causality."
        )

        for item in build_profile_flags(customer):
            st.write(f"• {item}")

        st.info(
            "The churn risk score is used for prioritisation. Because the model was "
            "trained with class-balancing methods, the score should not automatically "
            "be interpreted as a perfectly calibrated real-world probability."
        )


# ------------------------------------------------------------
# Tab 2: Batch customer prioritisation
# ------------------------------------------------------------
elif page == "Retention Dashboard":
    st.subheader("Retention Priority Dashboard")
    st.write(
        "Upload a CSV containing multiple customers. The app will score all records "
        "and create a prioritised retention list."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Use the same raw column names shown in the provided batch template.",
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        valid, message = validate_batch_data(batch_df)

        if not valid:
            st.error(message)
        else:
            encoded_batch = prepare_input(batch_df)
            scores = model.predict_proba(encoded_batch)[:, 1]

            results = batch_df.copy()
            results["ChurnRiskScore"] = scores
            results["RiskLevel"] = results["ChurnRiskScore"].apply(risk_level)
            results["RetentionPriority"] = results["RiskLevel"].apply(priority_label)
            results["RecommendedAction"] = results["RiskLevel"].apply(
                recommended_action
            )

            level_order = {"High": 0, "Medium": 1, "Low": 2}
            results["_risk_order"] = results["RiskLevel"].map(level_order)
            results = results.sort_values(
                ["_risk_order", "ChurnRiskScore"],
                ascending=[True, False],
            ).drop(columns=["_risk_order"])

            total_customers = len(results)
            high_count = int((results["RiskLevel"] == "High").sum())
            medium_count = int((results["RiskLevel"] == "Medium").sum())
            low_count = int((results["RiskLevel"] == "Low").sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Customers Analysed", total_customers)
            m2.metric("High Risk", high_count)
            m3.metric("Medium Risk", medium_count)
            m4.metric("Low Risk", low_count)

            st.markdown("### Customer Risk Overview")
            distribution = (
                results["RiskLevel"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
            )
            st.bar_chart(distribution)

            st.markdown("### Prioritised Customer List")

            display_columns = []
            if "customerID" in results.columns:
                display_columns.append("customerID")

            display_columns += [
                "ChurnRiskScore",
                "RiskLevel",
                "RetentionPriority",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "RecommendedAction",
            ]

            st.dataframe(
                results[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prioritised Customer List",
                data=csv_data,
                file_name="customer_retention_priority.csv",
                mime="text/csv",
            )


# ------------------------------------------------------------
# Tab 3: Decision logic
# ------------------------------------------------------------
elif page == "Decision Logic":
    st.subheader("How the Application Supports Business Decisions")

    st.markdown(
        """
        **1. Customer data**  
        Customer profile and service information are entered or uploaded.

        **2. Churn model**  
        The saved final model generates a churn risk score.

        **3. Risk level**  
        The score is converted into an operational risk band:
        - High: score ≥ 0.70
        - Medium: 0.40 ≤ score < 0.70
        - Low: score < 0.40

        **4. Retention priority**  
        High-risk customers are placed at the top of the follow-up queue.

        **5. Recommended action**  
        The application suggests the type of retention response that employees can
        consider for each risk level.

        **User-friendly design principle:** reduce unnecessary input requirements while maintaining the model workflow.

        **Business objective:** identify customers who may leave early enough for the
        company to prioritise retention resources more efficiently.
        """
    )

    st.warning(
        "The 0.40 and 0.70 risk bands are operational rules for this prototype. "
        "They should be reviewed against real retention costs and outcomes before "
        "production deployment."
    )
