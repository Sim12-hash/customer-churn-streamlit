import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Retention Management System",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "final_churn_model.pkl"
PORTFOLIO_PATH = BASE_DIR / "demo_customer_portfolio.csv"


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
    if PORTFOLIO_PATH.exists():
        return pd.read_csv(PORTFOLIO_PATH)
    return None


model = load_model()
portfolio = load_portfolio()


# ------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------

def prepare_input(raw_df):
    data = raw_df.copy()

    # 1. Enforce numeric data types
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 2. Strict alignment with training data preprocessing (Data Quality Assurance)
    # In the training environment, missing 'TotalCharges' were imputed with 0.
    # Maintaining exact consistency here prevents runtime errors during single-customer predictions.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        data[col] = data[col].fillna(0)

    # 3. Align categorical levels to prevent unexpected encoding shifts
    for col, levels in CATEGORY_LEVELS.items():
        data[col] = pd.Categorical(
            data[col],
            categories=levels
        )

    # 4. Apply One-Hot Encoding matching the baseline pipeline
    encoded = pd.get_dummies(
        data[RAW_REQUIRED_COLUMNS],
        columns=list(CATEGORY_LEVELS.keys()),
        drop_first=True,
        dtype=int
    )

    # 5. Guarantee feature dimensions strictly match the trained model's expectations
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

    if customer.get("Contract") == "Month-to-month":
        indicators.append("Month-to-month contract")

    try:
        tenure_val = float(customer.get("tenure", 0))
        if tenure_val < 12:
            indicators.append("Short customer tenure") 
    except ValueError:
        pass
    try:
        monthly_val = float(customer.get("MonthlyCharges", 0))
        if monthly_val >= 80:
            indicators.append("Higher monthly charges")
    except ValueError:
        pass

    if customer.get("PaymentMethod") == "Electronic check":
        indicators.append("Electronic check payment")

    if customer.get("TechSupport") == "No":
        indicators.append("No technical support service")

    if not indicators:
        indicators.append("No major profile indicator identified")

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
# Navigation & System Sidebar (Premium UI)
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌐 NextGen Telco CRM")
    st.caption("Intelligent Customer Retention Platform")
    st.divider()
    
    st.markdown("### 🧭 Main Menu")
    page = st.radio(
        "Navigation Menu",
        [
            "🔍 Customer Risk Assessment",
            "📊 Retention Management Dashboard",
            "🔬 Model Evaluation & Analytics"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("**System Status:** 🟢 Active")
    st.caption("Deployment: Streamlit Prototype\n\nModel Engine: Gradient Boosting")
# ------------------------------------------------------------
# Page 1: Customer Risk Assessment
# ------------------------------------------------------------

if page == "🔍 Customer Risk Assessment":

    st.subheader("Customer Risk Assessment")

    st.write(
        "Enter the customer's information to estimate churn risk "
        "and support retention decisions."
    )

    st.divider()

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "assessment_result" not in st.session_state:
        st.session_state["assessment_result"] = None

    if "scenario_result" not in st.session_state:
        st.session_state["scenario_result"] = None


    # ========================================================
    # CUSTOMER INPUT
    # ========================================================

    with st.form("customer_assessment_form"):

        # ====================================================
        # CORE CUSTOMER INFORMATION
        # ====================================================

        st.markdown("### Core Customer Information")

        st.caption(
            "These fields are prioritised based on the main churn "
            "patterns identified during exploratory analysis. "
            "Other model-required attributes remain available "
            "under Additional Information."
        )

        # ----------------------------------------------------
        # CUSTOMER RELATIONSHIP
        # ----------------------------------------------------

        st.markdown("**Customer Relationship**")

        col1, col2 = st.columns(2)

        with col1:

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
                help=(
                    "Number of months the customer has remained "
                    "with the company."
                )
            )

        with col2:

            contract = st.selectbox(
                "Contract",
                CATEGORY_LEVELS["Contract"],
                help=(
                    "The customer's current contract type. "
                    "Contract type showed a clear relationship "
                    "with churn during exploratory analysis."
                )
            )


        # ----------------------------------------------------
        # SERVICE & SUPPORT
        # ----------------------------------------------------

        st.markdown("**Service & Support**")

        col1, col2, col3 = st.columns(3)

        with col1:

            internet = st.selectbox(
                "Internet Service",
                CATEGORY_LEVELS["InternetService"],
                help=(
                    "Type of internet service used by the customer."
                )
            )

        with col2:

            tech_support = st.selectbox(
                "Tech Support",
                CATEGORY_LEVELS["TechSupport"],
                help=(
                    "Whether the customer subscribes to "
                    "technical support."
                )
            )

        with col3:

            online_security = st.selectbox(
                "Online Security",
                CATEGORY_LEVELS["OnlineSecurity"],
                help=(
                    "Whether the customer subscribes to "
                    "online security."
                )
            )


        # ----------------------------------------------------
        # BILLING
        # ----------------------------------------------------

        st.markdown("**Billing Information**")

        col1, col2 = st.columns(2)

        with col1:

            monthly = st.number_input(
                "Monthly Charges (RM)",
                min_value=0.0,
                value=70.0,
                step=1.0,
                help=(
                    "The customer's current monthly charges."
                )
            )

        with col2:

            payment = st.selectbox(
                "Payment Method",
                CATEGORY_LEVELS["PaymentMethod"],
                help=(
                    "The customer's current payment method."
                )
            )


        # ====================================================
        # ADDITIONAL INFORMATION
        # ====================================================

        st.markdown("### Additional Customer Information")

        st.caption(
            "These attributes are still required by the trained model "
            "but are placed here to keep the main assessment focused "
            "and easier to complete."
        )

        with st.expander("Show Additional Information"):

            # ------------------------------------------------
            # CUSTOMER BACKGROUND
            # ------------------------------------------------

            st.markdown("**Customer Background**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                gender = st.selectbox(
                    "Gender",
                    CATEGORY_LEVELS["gender"],
                    help=(
                        "Customer gender recorded in the "
                        "customer profile."
                    )
                )

            with col2:

                senior = st.selectbox(
                    "Senior Citizen",
                    ["No", "Yes"],
                    help=(
                        "Indicates whether the customer is "
                        "classified as a senior citizen."
                    )
                )

            with col3:

                partner = st.selectbox(
                    "Partner",
                    CATEGORY_LEVELS["Partner"],
                    help=(
                        "Indicates whether the customer "
                        "has a partner."
                    )
                )

            with col4:

                dependents = st.selectbox(
                    "Dependents",
                    CATEGORY_LEVELS["Dependents"],
                    help=(
                        "Indicates whether the customer "
                        "has dependents."
                    )
                )


            # ------------------------------------------------
            # PHONE SERVICES
            # ------------------------------------------------

            st.markdown("**Phone Services**")

            col1, col2 = st.columns(2)

            with col1:

                phone = st.selectbox(
                    "Phone Service",
                    CATEGORY_LEVELS["PhoneService"],
                    help=(
                        "Indicates whether the customer "
                        "has phone service."
                    )
                )

            with col2:

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    CATEGORY_LEVELS["MultipleLines"],
                    help=(
                        "Indicates whether the customer "
                        "has multiple phone lines."
                    )
                )


            # ------------------------------------------------
            # ADDITIONAL SERVICES
            # ------------------------------------------------

            st.markdown("**Additional Services**")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                online_backup = st.selectbox(
                    "Online Backup",
                    CATEGORY_LEVELS["OnlineBackup"],
                    help=(
                        "Indicates whether the customer "
                        "subscribes to online backup."
                    )
                )

            with col2:

                device_protection = st.selectbox(
                    "Device Protection",
                    CATEGORY_LEVELS["DeviceProtection"],
                    help=(
                        "Indicates whether the customer "
                        "subscribes to device protection."
                    )
                )

            with col3:

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    CATEGORY_LEVELS["StreamingTV"],
                    help=(
                        "Indicates whether the customer "
                        "subscribes to streaming TV."
                    )
                )

            with col4:

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    CATEGORY_LEVELS["StreamingMovies"],
                    help=(
                        "Indicates whether the customer "
                        "subscribes to streaming movies."
                    )
                )


            # ------------------------------------------------
            # ACCOUNT & BILLING
            # ------------------------------------------------

            st.markdown("**Account & Billing**")

            col1, col2 = st.columns(2)

            with col1:

                paperless = st.selectbox(
                    "Paperless Billing",
                    CATEGORY_LEVELS["PaperlessBilling"],
                    help=(
                        "Indicates whether the customer "
                        "uses paperless billing."
                    )
                )

            with col2:

                total = st.number_input(
                    "Total Charges (RM)",
                    min_value=0.0,
                    value=840.0,
                    step=10.0,
                    help=(
                        "The customer's accumulated charges."
                    )
                )


        # ====================================================
        # SUBMIT
        # ====================================================

        st.divider()

        submitted = st.form_submit_button(
            "Analyse Customer",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # RUN CUSTOMER ANALYSIS
    # ========================================================

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
            "TotalCharges": total
        }

        encoded_customer = prepare_input(
            pd.DataFrame([customer])
        )

        score = float(
            model.predict_proba(encoded_customer)[0, 1]
        )

        level = risk_level(score)

        retention_priority = priority(level)


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        st.session_state["assessment_result"] = {
            "customer": customer.copy(),
            "score": score,
            "level": level,
            "priority": retention_priority
        }

        # Clear previous What-if result
        st.session_state["scenario_result"] = None


    # ========================================================
    # RESTORE SAVED CUSTOMER ANALYSIS
    # ========================================================

    assessment = st.session_state["assessment_result"]


    if assessment is not None:

        customer = assessment["customer"]

        score = assessment["score"]

        level = assessment["level"]

        retention_priority = assessment["priority"]


        # ====================================================
        # ANALYSIS RESULT
        # ====================================================

        st.divider()

        st.subheader("Customer Churn Assessment")

        result1, result2, result3 = st.columns(3)

        with result1:

            st.metric(
                "Churn Risk Score",
                f"{score:.1%}",
                help=(
                    "Estimated churn probability generated "
                    "by the deployed Gradient Boosting model."
                )
            )

        with result2:

            st.metric(
                "Risk Level",
                level,
                help=(
                    "Business-defined risk category based "
                    "on the model's estimated churn probability."
                )
            )

        with result3:

            st.metric(
                "Retention Priority",
                retention_priority,
                help=(
                    "Operational priority used to support "
                    "retention planning."
                )
            )


        st.progress(
            min(max(score, 0.0), 1.0)
        )


        # ====================================================
        # RECOMMENDED ACTION
        # ====================================================

        st.markdown("### Recommended Retention Action")

        if level == "High":

            st.error(
                recommendation(level)
            )

        elif level == "Medium":

            st.warning(
                recommendation(level)
            )

        else:

            st.success(
                recommendation(level)
            )


        # ====================================================
        # CUSTOMER RISK INDICATORS
        # ====================================================

        st.markdown("### Customer Risk Indicators")

        st.caption(
            "These indicators provide additional customer context "
            "based on predefined business rules. They are not "
            "direct explanations of the machine learning model."
        )

        indicators = risk_indicators(customer)

        if indicators:

            indicator_cols = st.columns(
                min(len(indicators), 3)
            )

            for i, item in enumerate(indicators):

                with indicator_cols[
                    i % len(indicator_cols)
                ]:

                    st.info(item)


        # ====================================================
        # WHAT-IF SCENARIO
        # ====================================================

        st.divider()

        st.markdown("### What-if Scenario Analysis")

        st.caption(
            "Explore how the model-estimated churn risk changes "
            "when selected actionable customer attributes are modified."
        )


        scenario_open = (
            st.session_state["scenario_result"] is not None
        )


        with st.expander(
            "Explore What-if Scenario",
            expanded=scenario_open
        ):

            st.markdown("**Scenario Variables**")

            st.caption(
                "These variables were selected as actionable "
                "retention levers based on the churn patterns "
                "identified during the analysis."
            )


            scenario_customer = customer.copy()

            scenario_col1, scenario_col2 = st.columns(2)


            # ------------------------------------------------
            # CONTRACT & TECH SUPPORT
            # ------------------------------------------------

            with scenario_col1:

                scenario_customer["Contract"] = st.selectbox(
                    "Contract",
                    CATEGORY_LEVELS["Contract"],
                    index=CATEGORY_LEVELS["Contract"].index(
                        customer["Contract"]
                    ),
                    key="scenario_contract"
                )

                scenario_customer["TechSupport"] = st.selectbox(
                    "Tech Support",
                    CATEGORY_LEVELS["TechSupport"],
                    index=CATEGORY_LEVELS["TechSupport"].index(
                        customer["TechSupport"]
                    ),
                    key="scenario_tech_support"
                )


            # ------------------------------------------------
            # SECURITY & PAYMENT
            # ------------------------------------------------

            with scenario_col2:

                scenario_customer["OnlineSecurity"] = st.selectbox(
                    "Online Security",
                    CATEGORY_LEVELS["OnlineSecurity"],
                    index=CATEGORY_LEVELS["OnlineSecurity"].index(
                        customer["OnlineSecurity"]
                    ),
                    key="scenario_online_security"
                )

                scenario_customer["PaymentMethod"] = st.selectbox(
                    "Payment Method",
                    CATEGORY_LEVELS["PaymentMethod"],
                    index=CATEGORY_LEVELS["PaymentMethod"].index(
                        customer["PaymentMethod"]
                    ),
                    key="scenario_payment_method"
                )


            st.divider()


            # ------------------------------------------------
            # RUN WHAT-IF
            # ------------------------------------------------

            if st.button(
                "Run What-if Analysis",
                type="secondary",
                use_container_width=True
            ):

                scenario_score, scenario_level = predict_customer(
                    scenario_customer
                )

                st.session_state["scenario_result"] = {
                    "score": scenario_score,
                    "level": scenario_level,
                    "customer": scenario_customer.copy()
                }

                st.rerun()


            # =================================================
            # SCENARIO OUTPUT
            # =================================================

            scenario_result = (
                st.session_state["scenario_result"]
            )


            if scenario_result is not None:

                scenario_score = scenario_result["score"]

                scenario_level = scenario_result["level"]

                scenario_customer_result = (
                    scenario_result["customer"]
                )


                st.divider()

                st.markdown("### What-if Scenario Result")


                # ------------------------------------------------
                # ORIGINAL VS SCENARIO
                # ------------------------------------------------

                original_col, scenario_col = st.columns(2)

                with original_col:

                    st.markdown("#### Original Customer")

                    st.metric(
                        "Churn Risk",
                        f"{score:.1%}"
                    )

                    st.write(
                        f"**Risk Level:** {level}"
                    )


                with scenario_col:

                    st.markdown("#### What-if Scenario")

                    st.metric(
                        "Churn Risk",
                        f"{scenario_score:.1%}"
                    )

                    st.write(
                        f"**Risk Level:** {scenario_level}"
                    )


                # ------------------------------------------------
                # CHANGES MADE
                # ------------------------------------------------

                st.markdown("#### Changes Made")

                changes = []

                scenario_variables = [
                    "Contract",
                    "TechSupport",
                    "OnlineSecurity",
                    "PaymentMethod"
                ]

                display_names = {
                    "Contract": "Contract",
                    "TechSupport": "Tech Support",
                    "OnlineSecurity": "Online Security",
                    "PaymentMethod": "Payment Method"
                }


                for variable in scenario_variables:

                    original_value = customer[variable]

                    scenario_value = (
                        scenario_customer_result[variable]
                    )

                    if original_value != scenario_value:

                        changes.append({
                            "Variable": display_names[variable],
                            "Original": original_value,
                            "What-if": scenario_value
                        })


                if changes:

                    changes_df = pd.DataFrame(changes)

                    st.dataframe(
                        changes_df,
                        hide_index=True,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No changes were made to the selected "
                        "scenario variables."
                    )


                # ------------------------------------------------
                # RISK CHANGE
                # ------------------------------------------------

                st.markdown("#### Risk Change")

                risk_change = scenario_score - score

                change1, change2, change3 = st.columns(3)

                with change1:

                    st.metric(
                        "Original Risk",
                        f"{score:.1%}"
                    )

                with change2:

                    st.metric(
                        "Scenario Risk",
                        f"{scenario_score:.1%}"
                    )

                with change3:

                    st.metric(
                        "Change",
                        f"{risk_change:+.1%}",
                        delta_color="inverse"
                    )


                # ------------------------------------------------
                # INTERPRETATION
                # ------------------------------------------------

                st.markdown("#### Interpretation")

                if risk_change < 0:

                    st.success(
                        f"The scenario reduces the estimated churn "
                        f"risk from {score:.1%} to "
                        f"{scenario_score:.1%}, representing a "
                        f"decrease of {abs(risk_change):.1%}."
                    )

                elif risk_change > 0:

                    st.warning(
                        f"The scenario increases the estimated churn "
                        f"risk from {score:.1%} to "
                        f"{scenario_score:.1%}, representing an "
                        f"increase of {risk_change:.1%}."
                    )

                else:

                    st.info(
                        "The selected scenario does not change the "
                        "estimated churn risk."
                    )


                st.caption(
                    "The what-if result is a model-based scenario "
                    "comparison. It does not prove that changing a "
                    "single customer attribute will directly cause "
                    "churn risk to change."
                )


# ------------------------------------------------------------
# Page 2: Retention Management Dashboard
# ------------------------------------------------------------
elif page == "📊 Retention Management Dashboard":

    st.subheader("Retention Management Dashboard")

    if portfolio is None:
        st.error("⚠️ The customer portfolio dataset is currently unavailable.")
    else:
        results = portfolio.copy()
        encoded = prepare_input(results)
        scores = model.predict_proba(encoded)[:,1]

        results["ChurnRiskScore"] = scores
        results["RiskLevel"] = results["ChurnRiskScore"].apply(risk_level)
        results["RetentionPriority"] = results["RiskLevel"].apply(priority)

        a, b, c, d = st.columns(4)
        a.metric("Total Customers Evaluated", len(results))
        b.metric("High Risk (Priority 1)", sum(results["RiskLevel"]=="High"))
        c.metric("Medium Risk (Priority 2)", sum(results["RiskLevel"]=="Medium"))
        d.metric("Low Risk (Priority 3)", sum(results["RiskLevel"]=="Low"))

        st.divider()
        st.markdown("### 📈 Customer Journey & Business Patterns")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Contract Patterns", "🧾 Billing Patterns", "💰 Financial Patterns", "⏳ Tenure Patterns", "Hello"])

        with tab1:
            st.caption("Helps identify whether customer commitment level is associated with different estimated churn risk groups.")
            fig1 = px.histogram(
                results, x="RiskLevel", color="Contract", barmode="group",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Pattern Insight: Month-to-Month Customers Show Higher Estimated Churn Risk",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_layout(yaxis_title="Customer Count")
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            st.caption("Helps assess whether billing method is associated with different estimated churn risk groups.")
            fig1b = px.histogram(
                results, x="RiskLevel", color="PaymentMethod", barmode="group",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Pattern Insight: Electronic Check Users Show Higher Estimated Churn Risk",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig1b.update_layout(yaxis_title="Customer Count")
            st.plotly_chart(fig1b, use_container_width=True)

        with tab3:
            st.caption("Helps evaluate if financial burden correlates with the model's churn risk predictions.")
            fig2 = px.box(
                results, x="RiskLevel", y="MonthlyCharges", color="RiskLevel",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Pattern Insight: Monthly Charges Distribution Across Predicted Risk Groups",
                color_discrete_map={"High": "#EF553B", "Medium": "#66C2A5", "Low": "#8DA0CB"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            st.caption("Helps identify customer lifecycle stages requiring prioritized retention attention.")
            fig3 = px.scatter(
                results, x="tenure", y="ChurnRiskScore", color="RiskLevel", size="MonthlyCharges",
                hover_data=["customerID", "Contract"],
                title="Pattern Insight: Shorter Tenure is Associated with Higher Estimated Churn Risk",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                color_discrete_map={"High": "#EF553B", "Medium": "#66C2A5", "Low": "#8DA0CB"}
            )
            fig3.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            st.plotly_chart(fig3, use_container_width=True)

        with tab5:
            st.caption("Helps identify whether service type is associated with different estimated churn risk groups.")
            fig3b = px.histogram(
                results, x="RiskLevel", color="InternetService", barmode="group",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Pattern Insight: Fiber Optic Customers Show Higher Estimated Churn Risk",
                color_discrete_sequence=px.colors.qualitative.Pastel1
            )
            fig3b.update_layout(yaxis_title="Customer Count")
            st.plotly_chart(fig3b, use_container_width=True)
            
        st.divider()
        st.markdown("### 🎯 Priority Intervention Roster")
        
        # Added business justification for the selected filters
        with st.expander("ℹ️ Why these filters?"):
            st.markdown("""
            These filters were selected because they represent actionable customer segments that managers can review when planning retention strategies:
            * **Risk Level:** Prioritise customers based on immediate flight risk.
            * **Contract:** Compare commitment levels.
            * **Internet Service:** Review specific core service segments.
            * **Payment Method:** Review billing behaviour and friction points.
            """)

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filter_risk = st.multiselect("Risk Level", ["High", "Medium", "Low"], default=["High"])
        with col_f2:
            filter_contract = st.multiselect("Contract", CATEGORY_LEVELS["Contract"], default=CATEGORY_LEVELS["Contract"])
        with col_f3:
            filter_internet = st.multiselect("Internet Service", CATEGORY_LEVELS["InternetService"], default=CATEGORY_LEVELS["InternetService"])
        with col_f4:
            filter_payment = st.multiselect("Payment Method", CATEGORY_LEVELS["PaymentMethod"], default=CATEGORY_LEVELS["PaymentMethod"])
            
        filtered_results = results[
            (results["RiskLevel"].isin(filter_risk)) & 
            (results["Contract"].isin(filter_contract)) &
            (results["InternetService"].isin(filter_internet)) &
            (results["PaymentMethod"].isin(filter_payment))
        ].copy()
        
        # Adding the recommended action column for the manager
        filtered_results["Recommended Action"] = filtered_results["RiskLevel"].apply(
            lambda x: "Contact & review concerns" if x == "High" else ("Monitor engagement" if x == "Medium" else "Maintain relationship")
        )
        
        st.dataframe(
            filtered_results[["customerID", "ChurnRiskScore", "RiskLevel", "RetentionPriority", "Recommended Action", "Contract", "tenure", "MonthlyCharges"]].sort_values("ChurnRiskScore", ascending=False),
            hide_index=True, use_container_width=True
        )

# ------------------------------------------------------------
# Page 3: Model Evaluation & Analytics
# ------------------------------------------------------------
else:
    st.subheader("Model Evaluation & Advanced Analytics")

    st.markdown("### 🔬 Machine Learning Models Comparison")
    st.write(
        "Multiple machine learning algorithms were evaluated following the CRISP-DM methodology. "
        "All models utilized the same stratified 80/20 train-test split and 5-fold cross-validation."
    )

    # EXACT metrics mapped directly from the Jupyter Notebook output
    model_metrics = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression (Baseline)", 
            "Decision Tree", 
            "Random Forest", 
            "Gradient Boosting (Final)"
        ],
        "Test Accuracy": ["74.10%", "74.10%", "76.79%", "76.72%"],
        "ROC-AUC": ["0.8380", "0.8177", "0.8431", "0.8415"],
        "Test F1-Score": ["0.6121", "0.6162", "0.6237", "0.6306"]
    })
    
    st.info("🖱️ **User Tip:** Hover your mouse over the table headers to view detailed explanations of each evaluation metric.")
    st.dataframe(
        model_metrics, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Algorithm": st.column_config.TextColumn("Algorithm", help="The machine learning methodology evaluated via CRISP-DM."),
            "Test Accuracy": st.column_config.TextColumn("Accuracy", help="The overall percentage of correct predictions. Can be misleading in imbalanced datasets."),
            "ROC-AUC": st.column_config.TextColumn("ROC-AUC", help="Area Under the ROC Curve. Measures the model's ability to distinguish between churners and retained customers."),
            "Test F1-Score": st.column_config.TextColumn("Test F1-Score", help="The harmonic mean of Precision and Recall. Crucial for balancing the cost of false alarms vs. missed churners.")
        }
    )

    st.markdown("### 🏆 Justification for Model Selection")
    st.success("**Gradient Boosting** was selected as the final deployment model.")
    st.write(
        "**Evaluation Rationale:** The **F1-Score** is prioritized over Accuracy to handle the imbalanced nature of churn datasets. "
        "It provides a robust balance between Precision (minimizing false retention costs) and Recall (successfully identifying true churners). "
        "Gradient Boosting achieved the highest Test F1-Score."
    )
    
    st.info(
    """
    ℹ️ **Operational Thresholds**

    Risk categories are defined as:

    - **High Risk:** Churn probability >= 70%
    - **Medium Risk:** Churn probability between 40% and 69%
    - **Low Risk:** Churn probability < 40%

    These thresholds are business-defined prioritisation rules for this prototype,
    rather than model calibration thresholds or absolute statistical boundaries.
    They help customer success teams allocate retention resources effectively and
    can be adjusted according to future business requirements.
    """
)

    st.divider()

    # 🔴 FIXED: Replaced fabricated chart with a highly defensible, unquantified factual list based on EDA.
    st.markdown("### 🔑 Common Characteristics Associated with Churn")
    st.caption("*Note: Based on observed historical data patterns during the exploratory data analysis (EDA) phase.*")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.info("🔹 **Month-to-month Contract**")
        st.info("🔹 **Short Tenure (< 12 months)**")
        st.info("🔹 **Higher Monthly Charges**")
    with col_r2:
        st.info("🔹 **Fiber Optic Internet**")
        st.info("🔹 **Electronic Check Payment**")
