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
# User-input based individual churn assessment
# ------------------------------------------------------------

if page == "🔍 Customer Risk Assessment":

    st.subheader("Customer Risk Assessment")

    st.write(
        "Enter the customer's current information to generate an estimated "
        "churn risk assessment and retention priority."
    )

    st.divider()

    # --------------------------------------------------------
    # Customer Input Form
    # --------------------------------------------------------

    with st.form("customer_assessment_form"):

        # ====================================================
        # CORE CUSTOMER INFORMATION
        # ====================================================

        st.markdown("### Core Customer Information")

        st.caption(
            "These fields provide key customer relationship, service and "
            "billing information required for the assessment."
        )

        col1, col2, col3 = st.columns(3)

        # ----------------------------------------------------
        # Customer Relationship
        # ----------------------------------------------------

        with col1:

            st.markdown("**Customer Relationship**")

            gender = st.selectbox(
                "Gender",
                CATEGORY_LEVELS["gender"],
                help="Customer gender recorded in the customer profile."
            )

            senior = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                help="Whether the customer is classified as a senior citizen."
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
                help="Number of months the customer has been with the company."
            )

            contract = st.selectbox(
                "Contract",
                CATEGORY_LEVELS["Contract"],
                help="Customer's current contract type."
            )

        # ----------------------------------------------------
        # Service & Support
        # ----------------------------------------------------

        with col2:

            st.markdown("**Service & Support**")

            internet = st.selectbox(
                "Internet Service",
                CATEGORY_LEVELS["InternetService"],
                help="Type of internet service used by the customer."
            )

            online_security = st.selectbox(
                "Online Security",
                CATEGORY_LEVELS["OnlineSecurity"],
                help="Whether the customer subscribes to online security."
            )

            online_backup = st.selectbox(
                "Online Backup",
                CATEGORY_LEVELS["OnlineBackup"],
                help="Whether the customer subscribes to online backup."
            )

            tech_support = st.selectbox(
                "Tech Support",
                CATEGORY_LEVELS["TechSupport"],
                help="Whether the customer subscribes to technical support."
            )

        # ----------------------------------------------------
        # Billing
        # ----------------------------------------------------

        with col3:

            st.markdown("**Billing Information**")

            payment = st.selectbox(
                "Payment Method",
                CATEGORY_LEVELS["PaymentMethod"],
                help="Customer's current payment method."
            )

            monthly = st.number_input(
                "Monthly Charges (RM)",
                min_value=0.0,
                value=70.0,
                step=1.0,
                help="Customer's current monthly charges."
            )

            total = st.number_input(
                "Total Charges (RM)",
                min_value=0.0,
                value=840.0,
                step=10.0,
                help="Customer's accumulated charges."
            )

            device_protection = st.selectbox(
                "Device Protection",
                CATEGORY_LEVELS["DeviceProtection"],
                help="Whether the customer subscribes to device protection."
            )


        # ====================================================
        # ADDITIONAL CUSTOMER INFORMATION
        # ====================================================

        st.markdown("### Additional Customer Information")

        st.caption(
            "Additional fields are available to improve prediction completeness "
            "while keeping the main assessment focused."
        )

        with st.expander("Show additional information"):

            add_col1, add_col2, add_col3 = st.columns(3)

            # ------------------------------------------------
            # Customer Background
            # ------------------------------------------------

            with add_col1:

                st.markdown("**Customer Background**")

                partner = st.selectbox(
                    "Partner",
                    CATEGORY_LEVELS["Partner"],
                    help="Whether the customer has a partner recorded in the profile."
                )

                dependents = st.selectbox(
                    "Dependents",
                    CATEGORY_LEVELS["Dependents"],
                    help="Whether the customer has dependents."
                )

                paperless = st.selectbox(
                    "Paperless Billing",
                    CATEGORY_LEVELS["PaperlessBilling"],
                    help="Whether the customer uses paperless billing."
                )

            # ------------------------------------------------
            # Phone & Lines
            # ------------------------------------------------

            with add_col2:

                st.markdown("**Phone Services**")

                phone = st.selectbox(
                    "Phone Service",
                    CATEGORY_LEVELS["PhoneService"],
                    help="Whether the customer has phone service."
                )

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    CATEGORY_LEVELS["MultipleLines"],
                    help="Whether the customer has multiple phone lines."
                )

            # ------------------------------------------------
            # Entertainment Services
            # ------------------------------------------------

            with add_col3:

                st.markdown("**Entertainment Services**")

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    CATEGORY_LEVELS["StreamingTV"],
                    help="Whether the customer subscribes to streaming TV."
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    CATEGORY_LEVELS["StreamingMovies"],
                    help="Whether the customer subscribes to streaming movies."
                )


        st.divider()

        submitted = st.form_submit_button(
            "Analyse Customer",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # ANALYSIS RESULT
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
            "TotalCharges": total,
        }

        customer_df = pd.DataFrame([customer])

        encoded_customer = prepare_input(customer_df)

        score = float(
            model.predict_proba(encoded_customer)[0, 1]
        )

        level = risk_level(score)

        priority = priority_label(level)


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader("Customer Churn Assessment")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Churn Risk Score",
            f"{score:.1%}",
            help="Estimated churn risk score generated by the deployed machine learning model."
        )

        metric2.metric(
            "Risk Level",
            level,
            help="Operational classification based on the selected risk thresholds."
        )

        metric3.metric(
            "Retention Priority",
            priority,
            help="Priority level used to support retention planning."
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
                recommended_action(level)
            )

        elif level == "Medium":

            st.warning(
                recommended_action(level)
            )

        else:

            st.success(
                recommended_action(level)
            )


        # ====================================================
        # CUSTOMER PROFILE INDICATORS
        # ====================================================

        st.markdown("### Customer Risk Profile")

        st.caption(
            "These profile indicators provide additional customer context. "
            "They are based on simple business rules and are not direct "
            "explanations of the machine learning model's prediction."
        )

        indicators = build_profile_flags(customer)

        for item in indicators:

            st.write(f"• {item}")


        # ====================================================
        # MODEL SCORE NOTE
        # ====================================================

        st.info(
            "The churn risk score is intended for customer prioritisation. "
            "Because the model was trained using class-balancing methods, "
            "the score should not automatically be interpreted as a perfectly "
            "calibrated real-world probability."
        )


        # ====================================================
        # OPTIONAL WHAT-IF SCENARIO
        # ====================================================

        st.markdown("### Explore a What-if Scenario")

        st.caption(
            "You can test selected actionable customer attributes to see "
            "how the model-estimated churn risk changes under a different scenario."
        )

        with st.expander("Open What-if Scenario"):

            scenario_customer = customer.copy()

            st.markdown("**Actionable Attributes**")

            st.caption(
                "Only selected attributes that can reasonably be considered "
                "business intervention levers are included in the scenario analysis."
            )

            scenario_col1, scenario_col2 = st.columns(2)

            with scenario_col1:

                scenario_customer["Contract"] = st.selectbox(
                    "Scenario Contract",
                    CATEGORY_LEVELS["Contract"],
                    index=CATEGORY_LEVELS["Contract"].index(
                        customer["Contract"]
                    ),
                    key="scenario_contract"
                )

                scenario_customer["TechSupport"] = st.selectbox(
                    "Scenario Tech Support",
                    CATEGORY_LEVELS["TechSupport"],
                    index=CATEGORY_LEVELS["TechSupport"].index(
                        customer["TechSupport"]
                    ),
                    key="scenario_tech"
                )

            with scenario_col2:

                scenario_customer["OnlineSecurity"] = st.selectbox(
                    "Scenario Online Security",
                    CATEGORY_LEVELS["OnlineSecurity"],
                    index=CATEGORY_LEVELS["OnlineSecurity"].index(
                        customer["OnlineSecurity"]
                    ),
                    key="scenario_security"
                )

                scenario_customer["PaymentMethod"] = st.selectbox(
                    "Scenario Payment Method",
                    CATEGORY_LEVELS["PaymentMethod"],
                    index=CATEGORY_LEVELS["PaymentMethod"].index(
                        customer["PaymentMethod"]
                    ),
                    key="scenario_payment"
                )


            if st.button(
                "Run What-if Scenario",
                type="secondary",
                use_container_width=True
            ):

                scenario_score, scenario_level = predict_customer(
                    scenario_customer
                )

                st.markdown("#### Scenario Results")

                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric(
                    "Original Risk",
                    f"{score:.1%}"
                )

                result_col2.metric(
                    "Scenario Risk",
                    f"{scenario_score:.1%}"
                )

                result_col3.metric(
                    "Risk Change",
                    f"{scenario_score - score:+.1%}",
                    delta_color="inverse"
                )


                level_col1, level_col2 = st.columns(2)

                level_col1.metric(
                    "Original Risk Level",
                    level
                )

                level_col2.metric(
                    "Scenario Risk Level",
                    scenario_level
                )


                st.caption(
                    "The scenario result shows how the model-estimated risk "
                    "changes when the selected customer attributes are modified. "
                    "It should not be interpreted as a guaranteed causal effect."
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
        
        tab1, tab2, tab3 = st.tabs(["📋 Contract Patterns", "💰 Financial Patterns", "⏳ Tenure Patterns"])

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
            st.caption("Helps evaluate if financial burden correlates with the model's churn risk predictions.")
            fig2 = px.box(
                results, x="RiskLevel", y="MonthlyCharges", color="RiskLevel",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Pattern Insight: Monthly Charges Distribution Across Predicted Risk Groups",
                color_discrete_map={"High": "#EF553B", "Medium": "#66C2A5", "Low": "#8DA0CB"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
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
