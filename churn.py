import streamlit as st
import pandas as pd
import numpy as np
import heapq
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Telecom Churn & Retention Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI/UX Styling with Teal, Pink, and Cream aesthetic
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    
    .top-banner {
        background: linear-gradient(135deg, #14b8a6 0%, #db2777 100%);
        color: #fdfbf7;
        padding: 1.4rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .top-banner h1 {
        color: #fdfbf7 !important;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .top-banner p {
        color: #fdfbf7;
        opacity: 0.9;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #14b8a6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        color: #db2777;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .kpi-sub {
        color: #14b8a6;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }
    
    .strategy-box {
        background-color: #fdfbf7;
        border: 1px solid #fbcfe8;
        border-left: 5px solid #db2777;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Retention Strategy Mapping
STRATEGY_HASH_MAP = {
    "High_Price": {
        "code": "STRAT-01",
        "issue": "High Tariff Sensitivity (Bill > Rs. 1,250)",
        "action": "Apply 18% Loyalty Rebate for 6 Billing Cycles",
        "channel": "Automated Billing Trigger + Account Call",
        "cost_factor": 0.18,
        "expected_risk_drop": 0.32
    },
    "Short_Tenure_Contract": {
        "code": "STRAT-02",
        "issue": "Early Lifecycle Drop-off (Month-to-Month Plan)",
        "action": "Convert to Annual Plan with 1st Month Waiver + Router Upgrade",
        "channel": "In-App Push & Outbound Retention Desk",
        "cost_factor": 0.10,
        "expected_risk_drop": 0.38
    },
    "Tech_Support": {
        "code": "STRAT-03",
        "issue": "Service Reliability Friction (>= 4 Support Tickets)",
        "action": "Dispatch Tier-2 Field Engineer + 15% Bill Credit",
        "channel": "Priority Escalation Desk",
        "cost_factor": 0.12,
        "expected_risk_drop": 0.42
    },
    "Low_Usage_Drift": {
        "code": "STRAT-04",
        "issue": "Passive Disengagement & Moderate Billing Risk",
        "action": "Complimentary Speed Boost (100 Mbps) + OTT Bundle",
        "channel": "SMS & Email Retention Workflow",
        "cost_factor": 0.06,
        "expected_risk_drop": 0.22
    }
}
STRATEGY_HASH_MAP_BY_CODE = {v["code"]: v for v in STRATEGY_HASH_MAP.values()}

# Priority Queue for targeting high-risk customers
class RetentionPriorityQueue:
    def __init__(self):
        self.heap = []
        
    def push_customer(self, customer_id, churn_prob, monthly_charge, tenure, calls, contract_label, strat_code):
        annual_value = monthly_charge * 12
        priority_score = churn_prob * annual_value
        heapq.heappush(
            self.heap,
            (-priority_score, customer_id, churn_prob, monthly_charge, annual_value, tenure, calls, contract_label, strat_code)
        )
        
    def pop_top_n(self, n):
        results = []
        temp_heap = list(self.heap)
        heapq.heapify(temp_heap)
        for rank in range(1, min(n, len(temp_heap)) + 1):
            neg_score, cid, prob, charge, ann_val, ten, calls, contract_lbl, scode = heapq.heappop(temp_heap)
            strat_cost = ann_val * STRATEGY_HASH_MAP_BY_CODE[scode]["cost_factor"]
            results.append({
                "Queue_Rank": f"#{rank}",
                "Customer_ID": cid,
                "Priority_Index": round(-neg_score, 1),
                "Churn_Prob": round(prob * 100, 1),
                "Monthly_Bill_Rs": round(charge, 0),
                "Annual_Value_Rs": round(ann_val, 0),
                "Tenure_Mo": ten,
                "Support_Tickets": calls,
                "Contract": contract_lbl,
                "Assigned_Strategy": scode,
                "Intervention_Cost_Rs": round(strat_cost, 0)
            })
        return pd.DataFrame(results)

# Data Generation & Caching
@st.cache_data
def load_or_generate_data():
    np.random.seed(42)
    n_samples = 800
    
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.round(np.random.uniform(349, 2199, n_samples), 2)
    support_calls = np.random.poisson(1.9, n_samples)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.52, 0.28, 0.20])
    data_usage_gb = np.round(np.random.normal(185, 65, n_samples).clip(20, 500), 1)
    
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.96, 1.04, n_samples), 2)
    
    logit = (
        -1.65 - 0.048 * tenure + 0.0013 * monthly_charges
        + 0.52 * support_calls - 1.15 * contract_type - 0.003 * data_usage_gb
    )
    probs = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < probs).astype(int)
    
    contract_names = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
    
    df = pd.DataFrame({
        "CustomerID": [f"CUST-{1000+i}" for i in range(n_samples)],
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Data_Usage_GB": data_usage_gb,
        "Support_Calls": support_calls,
        "Contract_Type": contract_type,
        "Contract_Label": [contract_names[c] for c in contract_type],
        "Churn": churn
    })
    df.to_csv("telecom_churn_dataset.csv", index=False)
    return df

@st.cache_resource
def train_and_benchmark_models(df):
    features = ["Tenure_Months", "Monthly_Charges", "Total_Charges", "Data_Usage_GB", "Support_Calls", "Contract_Type"]
    X = df[features]
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42)
    rf_model.fit(X_train_sc, y_train)
    rf_preds = rf_model.predict(X_test_sc)
    rf_probs = rf_model.predict_proba(X_test_sc)[:, 1]
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_sc, y_train)
    lr_preds = lr_model.predict(X_test_sc)
    lr_probs = lr_model.predict_proba(X_test_sc)[:, 1]
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
    results = {
        "features": features,
        "rf": {
            "acc": accuracy_score(y_test, rf_preds), "prec": precision_score(y_test, rf_preds),
            "rec": recall_score(y_test, rf_preds), "f1": f1_score(y_test, rf_preds),
            "auc": auc(rf_fpr, rf_tpr), "fpr": rf_fpr, "tpr": rf_tpr, "cm": confusion_matrix(y_test, rf_preds)
        },
        "lr": {
            "acc": accuracy_score(y_test, lr_preds), "prec": precision_score(y_test, lr_preds),
            "rec": recall_score(y_test, lr_preds), "f1": f1_score(y_test, lr_preds),
            "auc": auc(lr_fpr, lr_tpr), "fpr": lr_fpr, "tpr": lr_tpr, "cm": confusion_matrix(y_test, lr_preds)
        },
        "importances": pd.DataFrame({
            "Feature": ["Tenure (Months)", "Monthly Bill (Rs)", "Total Billed (Rs)", "Monthly Data (GB)", "Support Tickets", "Contract Tier"],
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=True)
    }
    all_probs = rf_model.predict_proba(scaler.transform(X))[:, 1]
    return rf_model, lr_model, scaler, results, all_probs

def get_strategy_key(tenure, charge, calls, contract):
    if calls >= 4: return "Tech_Support"
    elif contract == 0 and tenure <= 9: return "Short_Tenure_Contract"
    elif charge > 1250: return "High_Price"
    else: return "Low_Usage_Drift"

df = load_or_generate_data()
rf_model, lr_model, scaler, results, all_probs = train_and_benchmark_models(df)
df["Churn_Prob"] = all_probs
df["Strategy_Key"] = [get_strategy_key(r.Tenure_Months, r.Monthly_Charges, r.Support_Calls, r.Contract_Type) for _, r in df.iterrows()]
df["Strategy_Code"] = df["Strategy_Key"].map(lambda k: STRATEGY_HASH_MAP[k]["code"])

CUSTOMER_HASH_TABLE = df.set_index("CustomerID").to_dict(orient="index")

# Sidebar
st.sidebar.markdown("### Project Navigation")
module = st.sidebar.radio(
    "Select System Module:",
    ["1. EDA & Model Benchmark", "2. Live Diagnostic & What-If Simulator", 
     "3. Priority Queue (Heap) & ROI Engine", "4. Data Structures & Complexity Lab"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Developed By")
st.sidebar.markdown("**Tanishka Prakash**")
st.sidebar.markdown("*AI & ML Developer*")
st.sidebar.markdown("*BCA, Amity University*")

st.markdown("""
<div class="top-banner">
    <h1>Customer Churn Prediction & Intelligent Retention System</h1>
    <p>Integrated ML Classification, Hash Mapping & Priority Queue Scheduling</p>
</div>
""", unsafe_allow_html=True)

# Module 1
if module == "1. EDA & Model Benchmark":
    k1, k2, k3, k4 = st.columns(4)
    at_risk_count = (df["Churn_Prob"] >= 0.45).sum()
    rev_at_risk = df.loc[df["Churn_Prob"] >= 0.45, "Monthly_Charges"].sum() * 12
    
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Active Cohort</div>
            <div class="kpi-value">{len(df)} Accounts</div>
            <div class="kpi-sub">Baseline Churn: {(df['Churn'].mean()*100):.1f}%</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Random Forest ROC-AUC</div>
            <div class="kpi-value">{results['rf']['auc']:.3f}</div>
            <div class="kpi-sub">Test Accuracy: {(results['rf']['acc']*100):.1f}%</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Flagged High-Risk Users</div>
            <div class="kpi-value">{at_risk_count} Users</div>
            <div class="kpi-sub">Threshold: P(Churn) >= 45%</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-title">Annual Revenue at Risk</div>
            <div class="kpi-value">Rs. {rev_at_risk/1e5:.2f} L</div>
            <div class="kpi-sub">Prioritized in Max-Heap</div></div>""", unsafe_allow_html=True)

    st.write("")
    tab_eda, tab_ml, tab_data = st.tabs(["📊 Exploratory Data Analysis", "🤖 Model Benchmarking", "🗂️ Dataset Explorer"])
    
    with tab_eda:
        c1, c2 = st.columns(2)
        with c1:
            fig_contract = px.histogram(
                df, x="Contract_Label", color=df["Churn"].map({0: "Retained (0)", 1: "Churned (1)"}),
                barmode="group", title="Churn Distribution by Contract Tier",
                color_discrete_map={"Retained (0)": "#14b8a6", "Churned (1)": "#db2777"}
            )
            fig_contract.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_contract, use_container_width=True)
            
        with c2:
            fig_box = px.box(
                df, x="Support_Calls", y="Monthly_Charges",
                color=df["Churn"].map({0: "Retained (0)", 1: "Churned (1)"}),
                title="Monthly Charges & Support Calls vs. Churn",
                color_discrete_map={"Retained (0)": "#14b8a6", "Churned (1)": "#db2777"}
            )
            fig_box.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_ml:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Classifier Performance Comparison")
            bench_df = pd.DataFrame({
                "Evaluation Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score"],
                "Random Forest (Selected)": [
                    f"{results['rf']['acc']*100:.2f}%", f"{results['rf']['prec']*100:.2f}%",
                    f"{results['rf']['rec']*100:.2f}%", f"{results['rf']['f1']*100:.2f}%", f"{results['rf']['auc']:.3f}"
                ],
                "Logistic Regression (Baseline)": [
                    f"{results['lr']['acc']*100:.2f}%", f"{results['lr']['prec']*100:.2f}%",
                    f"{results['lr']['rec']*100:.2f}%", f"{results['lr']['f1']*100:.2f}%", f"{results['lr']['auc']:.3f}"
                ]
            })
            st.table(bench_df)
            
            fig_imp = px.bar(
                results["importances"], x="Importance", y="Feature", orientation="h",
                title="Random Forest Feature Importance", color="Importance", color_continuous_scale="Teal"
            )
            fig_imp.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with col_r:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=results["rf"]["fpr"], y=results["rf"]["tpr"], mode="lines",
                name=f"Random Forest", line=dict(color="#14b8a6", width=3)))
            fig_roc.add_trace(go.Scatter(x=results["lr"]["fpr"], y=results["lr"]["tpr"], mode="lines",
                name=f"Logistic Regression", line=dict(color="#f472b6", width=2, dash="dash")))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="#94a3b8", dash="dot")))
            fig_roc.update_layout(title="ROC Curve Comparison", height=340, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_roc, use_container_width=True)
            
            cm = results["rf"]["cm"]
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                labels=dict(x="Predicted Class", y="Actual Class"),
                x=["Retained", "Churned"], y=["Retained", "Churned"], title="Confusion Matrix")
            fig_cm.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab_data:
        st.markdown("#### Generated Dataset")
        st.dataframe(df.head(50), use_container_width=True)

# Module 2
elif module == "2. Live Diagnostic & What-If Simulator":
    st.subheader("Interactive Customer Diagnostic & What-If Simulator")
    
    col_input, col_gauge, col_radar = st.columns([1.1, 1, 1.1])
    with col_input:
        st.markdown("##### 1. Configure Customer Parameters")
        in_tenure = st.slider("Account Tenure (Months)", 1, 72, 6)
        in_charge = st.slider("Monthly Subscription Bill (Rs.)", 349.0, 2199.0, 1550.0, step=25.0)
        in_usage = st.slider("Monthly Data Consumption (GB)", 20.0, 500.0, 120.0, step=10.0)
        in_calls = st.slider("Support Tickets (Last 30 Days)", 0, 10, 4)
        in_contract_lbl = st.selectbox("Contract Tier", ["Month-to-Month", "One Year", "Two Year"])
        
        c_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
        in_contract = c_map[in_contract_lbl]
        in_total = round(in_tenure * in_charge, 2)
        
    x_vec = np.array([[in_tenure, in_charge, in_total, in_usage, in_calls, in_contract]])
    prob_before = float(rf_model.predict_proba(scaler.transform(x_vec))[0][1])
    
    s_key = get_strategy_key(in_tenure, in_charge, in_calls, in_contract)
    strat = STRATEGY_HASH_MAP[s_key]
    
    sim_charge = in_charge * (1 - strat["cost_factor"])
    sim_calls = max(0, in_calls - 2) if s_key == "Tech_Support" else in_calls
    sim_contract = 1 if (s_key == "Short_Tenure_Contract" and in_contract == 0) else in_contract
    x_vec_after = np.array([[in_tenure + 3, sim_charge, in_total, in_usage, sim_calls, sim_contract]])
    prob_after = float(rf_model.predict_proba(scaler.transform(x_vec_after))[0][1])
    
    with col_gauge:
        st.markdown("##### 2. Real-Time Risk Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_before * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#db2777"}},
            delta={"reference": prob_after * 100, "increasing": {"color": "#db2777"}, "decreasing": {"color": "#14b8a6"}},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"color": "#0f172a"},
                "steps": [{"range": [0, 30], "color": "#ccfbf1"}, {"range": [30, 55], "color": "#fef3c7"}, {"range": [55, 100], "color": "#fbcfe8"}],
                "threshold": {"line": {"color": "#db2777", "width": 4}, "thickness": 0.75, "value": 45}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_radar:
        st.markdown("##### 3. Multi-Variable Risk Footprint")
        categories = ["Tenure", "Monthly Bill", "Data Usage", "Support Tickets", "Contract Stability"]
        user_norm = [in_tenure / 72, in_charge / 2200, in_usage / 500, min(in_calls / 8, 1.0), in_contract / 2]
        retained_avg = [0.62, 0.45, 0.48, 0.18, 0.65]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_norm, theta=categories, fill="toself", name="Selected Customer", line_color="#db2777"))
        fig_radar.add_trace(go.Scatterpolar(r=retained_avg, theta=categories, fill="toself", name="Retained Cohort Avg", line_color="#14b8a6"))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=340, margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    annual_clv = in_charge * 12
    offer_cost = annual_clv * strat["cost_factor"]
    net_saved = annual_clv - offer_cost
    
    st.markdown(f"""
    <div class="strategy-box">
        <h4 style="margin-top:0; color:#db2777;">Strategy Output — Code: <code>{strat['code']}</code></h4>
        <p><b>Diagnosed Root Cause:</b> {strat['issue']}</p>
        <p><b>Prescribed Action:</b> <span style="background:#fce7f3; padding:3px 8px; border-radius:4px; font-weight:600;">{strat['action']}</span></p>
        <p><b>Execution Channel:</b> {strat['channel']} &nbsp;|&nbsp; 
           <b>Annual Account Value:</b> Rs. {annual_clv:,.0f} &nbsp;|&nbsp; 
           <b>Intervention Cost:</b> Rs. {offer_cost:,.0f} &nbsp;|&nbsp; 
           <b>Net Revenue Protected:</b> <span style="color:#14b8a6; font-weight:700;">Rs. {net_saved:,.0f}/year</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Module 3
elif module == "3. Priority Queue (Heap) & ROI Engine":
    st.subheader("Max-Heap Scheduling & Financial Simulation")
    
    c_slider1, c_slider2 = st.columns(2)
    with c_slider1:
        top_k = st.slider("Daily Retention Team Capacity (Pop Top K):", 5, 100, 25, step=5)
    with c_slider2:
        conv_rate = st.slider("Expected Campaign Success Rate (%):", 20, 90, 60, step=5) / 100.0
        
    pq = RetentionPriorityQueue()
    at_risk_df = df[df["Churn_Prob"] >= 0.45]
    
    for _, row in at_risk_df.iterrows():
        pq.push_customer(
            row["CustomerID"], row["Churn_Prob"], row["Monthly_Charges"],
            row["Tenure_Months"], row["Support_Calls"], row["Contract_Label"], row["Strategy_Code"]
        )
        
    top_k_df = pq.pop_top_n(top_k)
    gross_rev_targeted = top_k_df["Annual_Value_Rs"].sum()
    total_campaign_cost = top_k_df["Intervention_Cost_Rs"].sum()
    expected_rev_saved = gross_rev_targeted * conv_rate
    net_roi_value = expected_rev_saved - total_campaign_cost
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Heap Size (At-Risk Pool)", f"{len(at_risk_df)} Accounts")
    m2.metric(f"Top-{top_k} Annual Revenue Pool", f"Rs. {gross_rev_targeted:,.0f}")
    m3.metric("Total Retention Budget Needed", f"Rs. {total_campaign_cost:,.0f}")
    m4.metric("Projected Net Revenue Saved", f"Rs. {net_roi_value:,.0f}")
    
    st.markdown("#### Extracted Priority Call List")
    st.dataframe(top_k_df, hide_index=True, use_container_width=True)

# Module 4
elif module == "4. Data Structures & Complexity Lab":
    st.subheader("Algorithmic Efficiency Dashboard")
    
    col_ds1, col_ds2 = st.columns(2)
    with col_ds1:
        st.markdown("#### Hash Table Key-Value Lookup")
        selected_cid = st.selectbox("Select Customer ID Key:", df["CustomerID"].head(100).tolist())
        
        t0 = time.perf_counter()
        record = CUSTOMER_HASH_TABLE[selected_cid]
        strat_info = STRATEGY_HASH_MAP[record["Strategy_Key"]]
        elapsed_us = (time.perf_counter() - t0) * 1e6
        
        st.success(f"Lookup Completed in **{elapsed_us:.2f} microseconds (µs)**")
        st.json({
            "CustomerID_Key": selected_cid,
            "Monthly_Charges_Rs": float(record["Monthly_Charges"]),
            "Predicted_Churn_Prob": f"{record['Churn_Prob']*100:.1f}%",
            "Action": strat_info["action"]
        })
        
    with col_ds2:
        st.markdown("#### Max-Heap Extraction vs. Array Sort")
        n_sizes = np.array([500, 2000, 10000, 25000, 50000, 100000])
        k_val = 25
        heap_ops = n_sizes + k_val * np.log2(n_sizes)
        sort_ops = n_sizes * np.log2(n_sizes)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=n_sizes, y=sort_ops, mode="lines+markers", name="Standard Sort", line=dict(color="#db2777", width=3)))
        fig_comp.add_trace(go.Scatter(x=n_sizes, y=heap_ops, mode="lines+markers", name="Binary Heap", line=dict(color="#14b8a6", width=3)))
        fig_comp.update_layout(title="Scaling: Heap Extract-K vs. Sorting", height=350, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_comp, use_container_width=True)