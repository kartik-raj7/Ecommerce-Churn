import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import plotly.graph_objects as go
import json
st.set_page_config(page_title="E-commerce Churn Predictor", layout="wide")
ecom_data = pd.read_excel("E Commerce Dataset.xlsx",sheet_name='E Comm')

st.title("🛒 E-commerce Customer Churn Predictor")
st.markdown("""Welcome to the **E-commerce Churn Predictor App**!
This app helps predict whether a customer is likely to churn based on their online behavior and demographics.
""")

#################################

# Step 1: Prepare data for Order Categories (double bar chart)
order_cat_churn = (
    ecom_data
    .groupby(['PreferedOrderCat', 'Churn'])
    .size()
    .reset_index(name='Count')
)

order_cat_churn['ChurnStatus'] = order_cat_churn['Churn'].map({0: 'Non-Churned', 1: 'Churned'})

# Step 2: Prepare data for Payment Modes (double line chart)
payment_mode_churn = (
    ecom_data
    .groupby(['PreferredPaymentMode', 'Churn'])
    .size()
    .reset_index(name='Count')
)

payment_mode_churn['ChurnStatus'] = payment_mode_churn['Churn'].map({0: 'Non-Churned', 1: 'Churned'})

# Step 3: Plot double bar chart for Order Categories
fig_order = go.Figure()

fig_order.add_trace(go.Bar(
    x=order_cat_churn[order_cat_churn['Churn'] == 0]['PreferedOrderCat'],
    y=order_cat_churn[order_cat_churn['Churn'] == 0]['Count'],
    name='Non-Churned',
    marker_color='orange'
))

fig_order.add_trace(go.Bar(
    x=order_cat_churn[order_cat_churn['Churn'] == 1]['PreferedOrderCat'],
    y=order_cat_churn[order_cat_churn['Churn'] == 1]['Count'],
    name='Churned',
    marker_color='blue'
))

fig_order.update_layout(
    title='Order Categories by Churn Status',
    xaxis_title='Order Category',
    yaxis_title='Number of Orders',
    barmode='group',
    height=500,
    margin=dict(t=50, b=40, l=40, r=40)
)

# Step 4: Plot double line chart for Payment Modes
fig_payment = go.Figure()

fig_payment.add_trace(go.Scatter(
    x=payment_mode_churn[payment_mode_churn['Churn'] == 0]['PreferredPaymentMode'],
    y=payment_mode_churn[payment_mode_churn['Churn'] == 0]['Count'],
    mode='lines+markers',
    name='Non-Churned',
    line=dict(color='orange')
))

fig_payment.add_trace(go.Scatter(
    x=payment_mode_churn[payment_mode_churn['Churn'] == 1]['PreferredPaymentMode'],
    y=payment_mode_churn[payment_mode_churn['Churn'] == 1]['Count'],
    mode='lines+markers',
    name='Churned',
    line=dict(color='blue')
))

fig_payment.update_layout(
    title='Payment Modes by Churn Status',
    xaxis_title='Payment Mode',
    yaxis_title='Number of Users',
    height=500,
    margin=dict(t=50, b=40, l=40, r=40)
)

# Step 5: Display charts side-by-side in Streamlit
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_order, use_container_width=True)

with col2:
    st.plotly_chart(fig_payment, use_container_width=True)
##############################

# Map churn for readability
ecom_data['ChurnLabel'] = ecom_data['Churn'].map({0: 'Not Churned', 1: 'Churned'})

# ---- 1. CityTier ----
city_group = ecom_data.groupby(['CityTier', 'ChurnLabel']).size().reset_index(name='Count')

fig_city = px.bar(
    city_group,
    x='CityTier',
    y='Count',
    color='ChurnLabel',
    barmode='group',
    title='Churned vs Non-Churned by City Tier',
    color_discrete_map={
    'Not Churned': '#ffa500',  # Green
    'Churned': '#347fbf'   # Red
}
)
fig_city.update_layout(xaxis_title='City Tier', yaxis_title='User Count')


# ---- 2. SatisfactionScore ----
score_group = ecom_data.groupby(['SatisfactionScore', 'ChurnLabel']).size().reset_index(name='Count')
score_group['Percent'] = score_group['Count'] / score_group.groupby('SatisfactionScore')['Count'].transform('sum') * 100

fig_stacked = px.bar(
    score_group,
    x='SatisfactionScore',
    y='Percent',
    color='ChurnLabel',
    barmode='stack',
    title='Churn Rate by Satisfaction Score',
    color_discrete_map={
    'Not Churned': '#ffa500',  # Green
    'Churned': '#347fbf'   # Red
}
)

fig_stacked.update_layout(
    xaxis_title='Satisfaction Score',
    yaxis_title='Percentage (%)'
)


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_stacked, use_container_width=True)

with col2:
    st.plotly_chart(fig_city, use_container_width=True)

##########################################

# Your feature scores as a list of tuples (feature, score)
features = [
    ("complain", 5),
    ("marital_status", 5),
    ("number_of_device_registered", 5),
    ("tenure", 5),
    ("satisfaction_score", 5),
    ("warehouse_to_home", 4),
    ("day_since_last_order", 4),
    ("prefered_order_cat_Laptop & Accessory", 4),
    ("city_tier", 4),
    ("number_of_address", 3),
    ("preferred_payment_mode_Credit Card", 3),
    ("preferred_payment_mode_E wallet", 3),
    ("preferred_login_device_Computer", 3),
    ("prefered_order_cat_Others", 3),
    ("prefered_order_cat_Mobile Phone", 3),
    ("preferred_payment_mode_COD", 3),
    ("cashback_amount", 3),
    ("preferred_login_device_Phone", 2),
    ("order_amount_hike_fromlast_year", 2),
    ("preferred_login_device_Mobile Phone", 2),
    ("prefered_order_cat_Grocery", 2),
    ("prefered_order_cat_Mobile", 2),
    ("gender", 1),
    ("order_count", 1),
    ("preferred_payment_mode_UPI", 1),
    ("preferred_payment_mode_CC", 1),
    ("preferred_payment_mode_Cash on Delivery", 1),
    ("preferred_payment_mode_Debit Card", 1),
]

# Convert to DataFrame
df_features = pd.DataFrame(features, columns=["Feature", "Score"])

st.markdown("""
<div style="display: flex; align-items: center;">
  <h3 style="margin: 0;">Feature Selection Summary</h3>
  <span title="I applied multiple feature selection algorithms and combined their results by consensus scoring to identify the most impactful features for churn prediction." style="margin-left: 8px; cursor: help;">ⓘ</span>
</div>
""", unsafe_allow_html=True)
# Show top 5 features
st.markdown("#### Top 5 Features")
st.table(df_features.head(5))

# Expandable section for full list
with st.expander("Show full feature ranking"):
    st.table(df_features)






# Model Evaluation and Selection

# Model performance data
model_metrics = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "KNN","Decision Tree","RF Classifier","XG Boost","XG Boost (Using Smote)"],
    "Accuracy": [0.89, 0.88, 0.88, 0.94, 0.96,0.96,0.96],
    "Precision": [0.70, 0.66, 0.60, 0.81, 0.87,0.87,0.90],
    "Recall": [0.61, 0.64, 0.77, 0.85, 0.87,0.90,0.88],
    "F1 Score": [0.65, 0.65, 0.68, 0.83, 0.87,0.89,0.89]
})


st.markdown("""
<div style="display: flex; align-items: center;">
  <h3 style="margin: 0;">Model Selection & Evaluation</h3>
  <span title="We trained and evaluated multiple machine learning models to predict customer churn. Performance was compared based on key metrics like Precision, Recall, and F1 Score." style="margin-left: 8px; cursor: help;">ⓘ</span>
</div>
""", unsafe_allow_html=True)
# Display comparison table
st.markdown("#### 📊 Model Comparison Table")
best_precision_idx = model_metrics["Precision"].idxmax()

# Styling function to highlight the best row based on Precision
def highlight_best_precision(row):
    color = 'background-color: black; color: lightgreen'
    return [color if row.name == best_precision_idx else '' for _ in row]
st.dataframe(model_metrics.style.apply(highlight_best_precision, axis=1))

# Visual: F1 score comparison
st.markdown("#### 🔎 Precision Score Comparison")
fig = px.bar(model_metrics, x="Model", y="Precision", color="Model", text="Precision",
             color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
st.plotly_chart(fig, use_container_width=True)

# Final model selection
best_model = model_metrics.sort_values(by="Precision", ascending=False).iloc[0]
st.success(f"✅ **Selected Model:** {best_model['Model']} (Precision Score: {best_model['Precision']})")

# Optional: Expand to show detailed reports
with st.expander("Show detailed classification report for XGBoost"):
    st.code("""
    Classification Report for XGBoost:
    
                  precision    recall  f1-score   support

               0       0.97      0.99      0.98       936
               1       0.93      0.86      0.89       190
            
            ROC AUC Score: 0.9836
    """)


# Load the customer sets from JSON
with open("sample_customers.json", "r") as f:
    all_sets = json.load(f)

# Define FastAPI prediction endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Define feature order used by the model
selected_features = [
    'complain',
    'number_of_device_registered',
    'satisfaction_score',
    'tenure',
    'marital_status',
    'prefered_order_cat_Laptop & Accessory',
    'city_tier',
    'day_since_last_order',
    'warehouse_to_home',
    'preferred_payment_mode_Credit Card',
    'preferred_login_device_Computer',
    'preferred_payment_mode_COD',
    'prefered_order_cat_Mobile Phone',
    'cashback_amount',
    'prefered_order_cat_Others',
    'number_of_address',
    'preferred_payment_mode_E wallet',
    'preferred_login_device_Phone',
    'prefered_order_cat_Grocery',
    'order_amount_hike_fromlast_year',
    'preferred_login_device_Mobile Phone'
]

st.title("🛍️ Ecommerce Churn Prediction Dashboard")

set_name = st.selectbox("Select a customer set", list(all_sets.keys()))
customers = all_sets[set_name]

# Display 3 customers at a time with individual Predict buttons
cols = st.columns(3)

for idx, customer in enumerate(customers[:3]):  # Only show first 3 for now
    with cols[idx]:
        st.markdown(f"### 👤 Customer #{idx + 1}")

        # Display only selected features
        show_features = ['complain', 'marital_status', 'number_of_device_registered', 'tenure', 'satisfaction_score']
        feature_data = {k: customer.get(k, '-') for k in show_features}
        df = pd.DataFrame(feature_data.items(), columns=["Feature", "Value"])
        st.table(df)

        if st.button("Predict", key=f"predict_{idx}"):
            try:
                input_data = [
                    int(customer[feature]) if isinstance(customer[feature], bool) else customer[feature]
                    for feature in selected_features  # must include all features expected by model
                ]
                response = requests.post(API_URL, json={"features": input_data})
                result = response.json()

                predicted = result.get("predicted_class", "Error")
                prob = result.get("churn_probability", "-")
                actual = "Churn" if customer.get("churn") == 1 else "Not Churn"

                
                if actual==predicted: st.success(f"**Actual:** {actual}")
                else: st.error(f"**Actual:** {actual}")
                st.markdown(f"**Predicted:** {predicted}")
                st.markdown(f"**Churn Probability:** `{prob}`")

            except Exception as e:
                st.error(f"Failed to reach API: {e}")
