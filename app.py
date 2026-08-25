import os
import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Retention Management System",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "final_churn_model.pkl"
PORTFOLIO_PATH = "demo_customer_portfolio.csv"


# ------------------------------------------------------------
# Feature configuration
# Same structure as notebook preprocessing
# ------------------------------------------------------------

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
# Load resources
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        return pd.read_csv(PORTFOLIO_PATH)
    return None


model = load_model()
portfolio = load_portfolio()


# ------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------

def prepare_input(raw_df):

    data = raw_df.copy()

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col, levels in CATEGORY_LEVELS.items():
        data[col] = pd.Categorical(
            data[col],
            categories=levels
        )

    encoded = pd.get_dummies(
        data[RAW_REQUIRED_COLUMNS],
        columns=list(CATEGORY_LEVELS.keys()),
        drop_first=True,
        dtype=int
    )

    encoded = encoded.reindex(
        columns=EXPECTED_FEATURES,
        fill_value=0
    )

    return encoded


def risk_level(score):
    if score >= 0.70:
        return "High"
    elif score >= 0.40:
        return "Medium"
    return "Low"


def priority(level):
    return {
        "High": "Priority 1",
        "Medium": "Priority 2",
        "Low": "Priority 3"
    }[level]


def recommendation(level):
    if level == "High":
        return (
            "Contact customer promptly, review service concerns, "
            "and consider a targeted retention offer."
        )

    if level == "Medium":
        return (
            "Maintain engagement, review service satisfaction, "
            "and monitor future churn signals."
        )

    return (
        "Maintain normal service quality and continue loyalty engagement."
    )


def risk_indicators(customer):

    indicators = []

    if customer["Contract"] == "Month-to-month":
        indicators.append("Month-to-month contract")

    if float(customer["tenure"]) < 12:
        indicators.append("Short customer tenure")

    if float(customer["MonthlyCharges"]) >= 80:
        indicators.append("Relatively high monthly charges")

    if customer["PaymentMethod"] == "Electronic check":
        indicators.append("Electronic check payment")

    if customer["TechSupport"] == "No":
        indicators.append("No technical support service")

    if not indicators:
        indicators.append(
            "No major profile indicator identified"
        )

    return indicators


def predict_customer(customer):

    df = pd.DataFrame([customer])

    encoded = prepare_input(df)

    score = float(
        model.predict_proba(encoded)[0,1]
    )

    level = risk_level(score)

    return score, level


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("Customer Retention Management System")

st.caption(
    "Identify customers with higher churn risk and support retention decisions."
)


# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Customer Risk Assessment",
        "📊 Retention Management Dashboard",
        "ℹ️ Model Explanation"
    ]
)


# ------------------------------------------------------------
# Page 1
# ------------------------------------------------------------

if page == "🔍 Customer Risk Assessment":

    st.subheader("Customer Risk Assessment")

    mode = st.radio(
        "Assessment Mode",
        [
            "Existing Customer Lookup",
            "What-if Scenario Analysis"
        ],
        horizontal=True
    )


    if mode == "Existing Customer Lookup":

        if portfolio is None:
            st.error(
                "demo_customer_portfolio.csv not found."
            )
            st.stop()


        customer_id = st.selectbox(
            "Select Customer ID",
            portfolio["customerID"]
        )


        if st.button("Analyse Customer"):

            customer = portfolio[
                portfolio["customerID"] == customer_id
            ].iloc[0].to_dict()


            score, level = predict_customer(customer)


            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Churn Risk Score",
                f"{score:.1%}"
            )

            c2.metric(
                "Risk Level",
                level
            )

            c3.metric(
                "Retention Priority",
                priority(level)
            )


            st.progress(score)


            st.markdown("### Customer Profile")

            display = pd.DataFrame(
                {
                    "Attribute": list(customer.keys()),
                    "Value": list(customer.values())
                }
            )

            st.dataframe(
                display,
                hide_index=True
            )


            st.markdown("### Customer Risk Indicators")

            for item in risk_indicators(customer):
                st.write("• " + item)


            st.markdown("### Retention Recommendation")

            st.info(
                recommendation(level)
            )


    else:

        st.info(
            "Use this mode to test different customer scenarios."
        )

        customer = {}

        col1, col2 = st.columns(2)

        with col1:

            customer["gender"] = st.selectbox(
                "Gender",
                CATEGORY_LEVELS["gender"]
            )

            customer["SeniorCitizen"] = (
                1 if st.selectbox(
                    "Senior Citizen",
                    ["No","Yes"]
                ) == "Yes"
                else 0
            )

            customer["tenure"] = st.number_input(
                "Tenure",
                0,
                100,
                12
            )

            customer["Contract"] = st.selectbox(
                "Contract",
                CATEGORY_LEVELS["Contract"]
            )

            customer["InternetService"] = st.selectbox(
                "Internet Service",
                CATEGORY_LEVELS["InternetService"]
            )


        with col2:

            customer["PaymentMethod"] = st.selectbox(
                "Payment Method",
                CATEGORY_LEVELS["PaymentMethod"]
            )

            customer["MonthlyCharges"] = st.number_input(
                "Monthly Charges",
                0.0,
                200.0,
                70.0
            )

            customer["TotalCharges"] = st.number_input(
                "Total Charges",
                0.0,
                10000.0,
                840.0
            )

            customer["OnlineSecurity"] = st.selectbox(
                "Online Security",
                CATEGORY_LEVELS["OnlineSecurity"]
            )

            customer["TechSupport"] = st.selectbox(
                "Tech Support",
                CATEGORY_LEVELS["TechSupport"]
            )


        # hidden default attributes
        customer.update(
            {
                "Partner":"No",
                "Dependents":"No",
                "PhoneService":"Yes",
                "MultipleLines":"No",
                "OnlineBackup":"No",
                "DeviceProtection":"No",
                "StreamingTV":"No",
                "StreamingMovies":"No",
                "PaperlessBilling":"No"
            }
        )


        if st.button("Calculate Scenario Risk"):

            score, level = predict_customer(customer)

            st.metric(
                "Estimated Churn Risk",
                f"{score:.1%}"
            )

            st.write(
                "Risk Level:",
                level
            )


# ------------------------------------------------------------
# Page 2
# ------------------------------------------------------------

elif page == "📊 Retention Management Dashboard":

    st.subheader(
        "Retention Management Dashboard"
    )


    if portfolio is None:

        st.error(
            "demo_customer_portfolio.csv not found."
        )

    else:

        results = portfolio.copy()

        encoded = prepare_input(results)

        scores = model.predict_proba(encoded)[:,1]

        results["ChurnRiskScore"] = scores

        results["RiskLevel"] = (
            results["ChurnRiskScore"]
            .apply(risk_level)
        )

        results["RetentionPriority"] = (
            results["RiskLevel"]
            .apply(priority)
        )


        a,b,c,d = st.columns(4)

        a.metric(
            "Customers Analysed",
            len(results)
        )

        b.metric(
            "High Risk",
            sum(results["RiskLevel"]=="High")
        )

        c.metric(
            "Medium Risk",
            sum(results["RiskLevel"]=="Medium")
        )

        d.metric(
            "Low Risk",
            sum(results["RiskLevel"]=="Low")
        )


        st.bar_chart(
            results["RiskLevel"]
            .value_counts()
        )


        st.markdown(
            "### Priority Customer List"
        )


        st.dataframe(
            results[
                [
                    "customerID",
                    "ChurnRiskScore",
                    "RiskLevel",
                    "RetentionPriority",
                    "Contract",
                    "tenure",
                    "MonthlyCharges"
                ]
            ]
            .sort_values(
                "ChurnRiskScore",
                ascending=False
            ),
            hide_index=True
        )


# ------------------------------------------------------------
# Page 3
# ------------------------------------------------------------

else:

    st.subheader(
        "Model Explanation"
    )


    st.write(
        """
        Final model selected:

        **Gradient Boosting**

        Selection reason:

        Highest F1-score among evaluated models.

        Test F1-score:

        **0.6306**

        F1-score was selected because churn prediction requires
        balancing the identification of churn customers and
        avoiding unnecessary retention costs.

        The churn risk score supports business prioritisation
        and should not be interpreted as a guaranteed outcome.
        """
    )
