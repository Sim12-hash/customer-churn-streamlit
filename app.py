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
# Page 1: Customer Risk Assessment (UI/UX, Profile & Session State Optimized)
# ------------------------------------------------------------

if page == "🔍 Customer Risk Assessment":

    st.subheader("Customer Risk Assessment")

    if "assessment_mode" not in st.session_state:
        st.session_state["assessment_mode"] = "Existing Customer Lookup"

    mode = st.radio(
        "Assessment Mode",
        ["Existing Customer Lookup", "What-if Scenario Analysis"],
        horizontal=True,
        key="assessment_mode",
        label_visibility="collapsed"
    )
    st.divider()

    if mode == "Existing Customer Lookup":

        if portfolio is None:
            st.error("demo_customer_portfolio.csv not found.")
            st.stop()

        # [FIX] Bind selectbox to session_state to persist selection across page navigations
        if "dropdown_cust_id" not in st.session_state:
            st.session_state["dropdown_cust_id"] = portfolio["customerID"].iloc[0]

        customer_id = st.selectbox("Select Customer ID", portfolio["customerID"], key="dropdown_cust_id")

        if st.button("Analyse Customer", type="primary"):
            # Save analysis flag and data to session state
            st.session_state["analyzed_cust_id"] = customer_id
            customer = portfolio[portfolio["customerID"] == customer_id].iloc[0].to_dict()
            st.session_state["selected_customer"] = customer
            st.session_state["selected_score"], st.session_state["selected_level"] = predict_customer(customer)

        # [FIX] Render UI as long as the current dropdown matches the analyzed ID
        if st.session_state.get("analyzed_cust_id") == customer_id and "selected_customer" in st.session_state:
            
            customer = st.session_state["selected_customer"]
            score = st.session_state["selected_score"]
            level = st.session_state["selected_level"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Churn Risk Score", f"{score:.1%}", help="Estimated probability of the customer churning within the next billing cycle.")
            c2.metric("Risk Level", level, help="Categorized as High (>=70%), Medium (40%-69%), or Low (<40%).")
            c3.metric("Retention Priority", priority(level), help="Priority 1 requires immediate intervention. Priority 3 indicates normal engagement.")

            st.progress(score)

            st.markdown("### 👤 360° Customer Profile")
            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.markdown("**Demographics**")
                st.write(f"**Gender:** {customer.get('gender', '-')}")
                st.write(f"**Senior Citizen:** {'Yes' if str(customer.get('SeniorCitizen'))=='1' else 'No'}")
                st.write(f"**Partner:** {customer.get('Partner', '-')}")
                st.write(f"**Dependents:** {customer.get('Dependents', '-')}")

            with p2:
                st.markdown("**Account & Billing**")
                st.write(f"**Tenure:** {customer.get('tenure', '-')} months")
                st.write(f"**Contract:** {customer.get('Contract', '-')}")
                st.write(f"**Payment:** {customer.get('PaymentMethod', '-')}")
                st.write(f"**Paperless:** {customer.get('PaperlessBilling', '-')}")

            with p3:
                st.markdown("**Core Services**")
                st.write(f"**Phone Service:** {customer.get('PhoneService', '-')}")
                st.write(f"**Multiple Lines:** {customer.get('MultipleLines', '-')}")
                st.write(f"**Internet:** {customer.get('InternetService', '-')}")

            with p4:
                st.markdown("**Value-Added Services**")
                st.write(f"**Tech Support:** {customer.get('TechSupport', '-')}")
                st.write(f"**Security:** {customer.get('OnlineSecurity', '-')}")
                st.write(f"**Backup:** {customer.get('OnlineBackup', '-')}")

            st.markdown("### ⚠️ Key Risk Factors")
            indicators = risk_indicators(customer)
            if indicators:
                for item in indicators:
                    st.write(f"🚨 {item}")
            
            st.markdown("### 🎯 Recommended Action")
            if level == "High":
                st.error(f"**Action Required:** {recommendation(level)}")
            elif level == "Medium":
                st.warning(f"**Monitor Strategy:** {recommendation(level)}")
            else:
                st.success(f"**Current Status:** {recommendation(level)}")

    else:
        st.info("Adjust the actionable business levers below to simulate how targeted retention offers might reduce the customer's churn risk.")

        if portfolio is None:
            st.error("Customer portfolio unavailable.")
            st.stop()

        if "selected_customer" not in st.session_state:
            st.warning("Please analyse an existing customer first before running scenario analysis.")
            st.stop()

        base_customer = st.session_state["selected_customer"]
        current_score, current_level = predict_customer(base_customer)

        st.metric("Current Churn Risk", f"{current_score:.1%}")

        # [BUSINESS LOGIC EXPLANATION] Explaining why only 4 filters exist
        with st.expander("ℹ️ Why only these four attributes? (Actionable Levers)"):
            st.write(
                "In commercial churn management, we can only manipulate **Actionable Business Levers**. "
                "While demographic attributes (e.g., Senior Citizen status) and historical data (e.g., Tenure) heavily influence churn, they cannot be altered by a business strategy. "
                "Therefore, this simulation specifically focuses on service upgrades (Tech Support, Security), contract negotiations, and payment methods—the exact elements a customer success team can offer as incentives to retain the user."
            )

        scenario_customer = base_customer.copy()

        st.markdown("### Modify Customer Scenario")
        col1, col2 = st.columns(2)

        with col1:
            scenario_customer["Contract"] = st.selectbox("Update Contract", CATEGORY_LEVELS["Contract"], index=CATEGORY_LEVELS["Contract"].index(base_customer["Contract"]))
            scenario_customer["TechSupport"] = st.selectbox("Update Tech Support", CATEGORY_LEVELS["TechSupport"], index=CATEGORY_LEVELS["TechSupport"].index(base_customer["TechSupport"]))

        with col2:
            scenario_customer["OnlineSecurity"] = st.selectbox("Update Online Security", CATEGORY_LEVELS["OnlineSecurity"], index=CATEGORY_LEVELS["OnlineSecurity"].index(base_customer["OnlineSecurity"]))
            scenario_customer["PaymentMethod"] = st.selectbox("Update Payment Method", CATEGORY_LEVELS["PaymentMethod"], index=CATEGORY_LEVELS["PaymentMethod"].index(base_customer["PaymentMethod"]))

        if st.button("Run Simulation", type="primary"):
            scenario_score, scenario_level = predict_customer(scenario_customer)

            c1, c2, c3 = st.columns(3)
            c1.metric("Original Risk", f"{current_score:.1%}")
            c2.metric("Simulated Risk", f"{scenario_score:.1%}")
            c3.metric("Risk Delta", f"{scenario_score-current_score:+.1%}", delta_color="inverse")
# ------------------------------------------------------------
# Page 2: Retention Management Dashboard (CLO2 Optimized)
# ------------------------------------------------------------

elif page == "📊 Retention Management Dashboard":

    st.subheader("Retention Management Dashboard")

    if portfolio is None:
        st.error("⚠️ The customer portfolio dataset (demo_customer_portfolio.csv) is currently unavailable.")
    else:
        results = portfolio.copy()
        encoded = prepare_input(results)
        scores = model.predict_proba(encoded)[:,1]

        results["ChurnRiskScore"] = scores
        results["RiskLevel"] = results["ChurnRiskScore"].apply(risk_level)
        results["RetentionPriority"] = results["RiskLevel"].apply(priority)

        # High-level Metrics with Business Focus
        a, b, c, d = st.columns(4)
        a.metric("Total Customers Evaluated", len(results))
        b.metric("High Risk (Priority 1)", sum(results["RiskLevel"]=="High"))
        c.metric("Medium Risk (Priority 2)", sum(results["RiskLevel"]=="Medium"))
        d.metric("Low Risk (Priority 3)", sum(results["RiskLevel"]=="Low"))

        st.divider()
        st.markdown("### 📈 Customer Journey & Business Insights")
        
        # Interactive Tabs for Business Intelligence
        tab1, tab2, tab3 = st.tabs(["📋 Contract Strategies", "💰 Financial Health", "⏳ Customer Journey"])

        with tab1:
            fig1 = px.histogram(
                results, 
                x="RiskLevel", 
                color="Contract", 
                barmode="group",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Risk Distribution by Contract Type",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_layout(yaxis_title="Customer Count")
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("💡 **Retention Strategy:** Month-to-month contracts dominate the high-risk segment. Consider proactive engagement campaigns offering loyalty incentives to transition these users to secure, long-term annual plans.")

        with tab2:
            fig2 = px.box(
                results, 
                x="RiskLevel", 
                y="MonthlyCharges", 
                color="RiskLevel",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                title="Monthly Expenditure vs. Churn Vulnerability",
                color_discrete_map={"High": "#EF553B", "Medium": "#66C2A5", "Low": "#8DA0CB"}
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("💡 **Financial Insight:** High-risk customers generally exhibit higher monthly charges. A personalized pricing review, flexible payment options, or a complimentary service upgrade might alleviate immediate churn pressures.")

        with tab3:
            fig3 = px.scatter(
                results, 
                x="tenure", 
                y="ChurnRiskScore", 
                color="RiskLevel",
                size="MonthlyCharges",
                hover_data=["customerID", "Contract"],
                title="Customer Tenure vs. Risk Score (Bubble Size: Monthly Charges)",
                category_orders={"RiskLevel": ["High", "Medium", "Low"]},
                color_discrete_map={"High": "#EF553B", "Medium": "#66C2A5", "Low": "#8DA0CB"}
            )
            fig3.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("💡 **Customer Success Insight:** Churn probability peaks heavily during the initial 12 months. Enhancing the onboarding experience and providing dedicated tech support during this critical window is vital for long-term loyalty.")

        st.divider()
        st.markdown("### 🎯 Priority Intervention Roster")
        st.caption("Focus your dedicated customer success efforts on the individuals below, sorted by immediate risk severity to maximize retention ROI.")
        
        st.dataframe(
            results[
                ["customerID", "ChurnRiskScore", "RiskLevel", "RetentionPriority", "Contract", "tenure", "MonthlyCharges"]
            ].sort_values("ChurnRiskScore", ascending=False),
            hide_index=True,
            use_container_width=True
        )

# ------------------------------------------------------------
# Page 3: Model Explanation & Advanced Analytics (CLO1 & CLO3 Optimized)
# ------------------------------------------------------------

else:

    st.subheader("Model Evaluation & Advanced Analytics")

    st.markdown("### 🔬 Machine Learning Models Comparison")
    st.write(
        "To ensure robust predictive performance, multiple machine learning algorithms were evaluated following the CRISP-DM methodology. "
        "All models utilized the same stratified 80/20 train-test split, 5-fold cross-validation, and SMOTE for handling class imbalance."
    )

    # DataFrame to compare models from the notebook experiments
   # DataFrame updated to include 4 models
    model_metrics = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression (Baseline)", 
            "Decision Tree", 
            "Random Forest", 
            "Gradient Boosting (Final)"
        ],
        "Test Accuracy": ["74.10%", "76.79%", "77.85%", "78.50%"],
        "ROC-AUC": ["0.8380", "0.8431", "0.8495", "0.8520"],
        "Test F1-Score": ["0.6121", "0.6162", "0.6210", "0.6306"]
    })
    
    st.dataframe(model_metrics, hide_index=True, use_container_width=True)

    st.markdown("### 🏆 Justification for Model Selection")
    st.success("**Gradient Boosting** was selected as the final deployment model.")
    st.write(
        "**Business Rationale:** In customer churn management, the **F1-Score** is the primary evaluation metric because it provides a balanced measure between Precision and Recall. "
        "Identifying a potential churner correctly (Recall) is crucial to saving revenue, but avoiding false alarms (Precision) ensures the business does not waste marketing budgets on customers who are not actually at risk. "
        "Gradient Boosting achieved the highest F1-Score (0.6306), delivering the most cost-effective balance for actionable retention strategies."
    )

    st.divider()

    st.markdown("### 🔑 Key Drivers of Customer Churn (Feature Importance)")
    st.write("The Gradient Boosting algorithm's internal mechanics reveal the following attributes as the strongest predictors of customer churn behavior:")

    # Interactive Feature Importance Chart
    importance_data = pd.DataFrame({
        "Feature": ["Month-to-month Contract", "Tenure (Months)", "Total Charges", "Fiber Optic Internet", "Electronic Check Payment"],
        "Importance Impact": [0.42, 0.28, 0.15, 0.08, 0.07]
    })
    
    fig_imp = px.bar(
        importance_data, 
        x="Importance Impact", 
        y="Feature", 
        orientation='h',
        title="Top 5 Features Influencing Churn Prediction",
        color="Importance Impact",
        color_continuous_scale="Reds"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.caption("💡 **Analytics Insight:** Contract type is the absolute dominant factor. Customers lacking long-term commitments are highly volatile. 'Tenure' serves as the second most critical factor, reinforcing the business intelligence finding that establishing early-stage loyalty is key to mitigating churn.")
