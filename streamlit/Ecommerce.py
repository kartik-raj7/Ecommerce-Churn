import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import json
st.set_page_config(page_title="E-commerce Churn Predictor", layout="wide")
ecom_data = pd.read_excel("E Commerce Dataset.xlsx",sheet_name='E Comm')

st.title("🛒 E-commerce Customer Churn Predictor")
st.markdown("""Welcome to the **E-commerce Churn Predictor App**!
This app helps predict whether a customer is likely to churn based on their online behavior and demographics.
""")

# st.markdown("""
# ### 💡 What is Customer Churn?
# Customer churn refers to when a customer stops using a company's product or service. Identifying customers at risk of churning can help businesses take proactive measures to retain them.

# In this project, we use a machine learning model trained on e-commerce customer data to predict churn likelihood.

# ### 🔍 What You'll Find in This App:
# - Explore the data and key insights.
# - Visualize patterns related to churn.
# - Input customer information and get churn predictions.
# """)
# st.markdown("📊 Sample Data")
# st.dataframe(ecom_data.head())

# st.markdown("### 📈 Dataset Summary")
# st.write("Number of rows:", ecom_data.shape[0])
# st.write("Number of columns:", ecom_data.shape[1])
# st.write("Churn rate in dataset:")
# st.write(ecom_data['Churn'].value_counts(normalize=True).map("{:.2%}".format))

# col1, col2 = st.columns(2)

# with col1:
#     # Bar chart for 'PreferedOrderCat' (note: double-check spelling, your data column name might be "PreferedOrderCat" or "PreferredOrderCat")
#     order_cat_counts = ecom_data['PreferedOrderCat'].value_counts().reset_index()
#     order_cat_counts.columns = ['OrderCategory', 'Count']

#     fig4 = px.bar(order_cat_counts, 
#                   x='OrderCategory', 
#                   y='Count', 
#                   color='Count',
#                   color_continuous_scale='Oranges',
#                   title="Popular Order Categories",
#                   labels={'Count': 'Number of Orders', 'OrderCategory': 'Order Category'},
#                   height=300, width=700)
#     fig4.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
#     st.plotly_chart(fig4, use_container_width=True)
#     st.caption("🔹 Laptops & Mobile Accessories are the most popular, while Grocery is the least.")

# with col2:
#     # Pie chart for 'PreferredPaymentMode'
#     payment_mode_counts = ecom_data['PreferredPaymentMode'].value_counts().reset_index()
#     payment_mode_counts.columns = ['PaymentMode', 'Count']

#     fig2 = px.pie(payment_mode_counts, 
#                   names='PaymentMode', 
#                   values='Count', 
#                   color_discrete_sequence=px.colors.sequential.Reds,
#                   title="Payment Mode Preferences")
#     fig2.update_traces(textposition='inside', textinfo='percent+label')
#     fig2.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=500, width=700)
#     st.plotly_chart(fig2, use_container_width=True)

#################################
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Step 1: Data prep
order_cat_counts = ecom_data['PreferedOrderCat'].value_counts().reset_index()
order_cat_counts.columns = ['Category', 'Count']
order_cat_counts['Type'] = 'Order Category'

payment_mode_counts = ecom_data['PreferredPaymentMode'].value_counts().reset_index()
payment_mode_counts.columns = ['Category', 'Count']
payment_mode_counts['Type'] = 'Payment Mode'

# Combine both
combined_df = pd.concat([order_cat_counts, payment_mode_counts])

# Step 2: Plot using Plotly
fig = go.Figure()

# Bar chart for Order Categories
fig.add_trace(go.Bar(
    x=order_cat_counts['Category'],
    y=order_cat_counts['Count'],
    name='Order Categories',
    marker_color='orange',
    yaxis='y1'
))

# Line chart for Payment Modes
fig.add_trace(go.Scatter(
    x=payment_mode_counts['Category'],
    y=payment_mode_counts['Count'],
    name='Payment Modes',
    mode='lines+markers',
    line=dict(color='red'),
    yaxis='y2'
))

# Step 3: Layout for dual axes
fig.update_layout(
    title='Popular Order Categories vs Payment Mode Preferences',
    xaxis=dict(title='Category'),
    yaxis=dict(title='Number of Orders (Bar)', side='left'),
    yaxis2=dict(title='Payment Preferences (Line)', overlaying='y', side='right'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500, width=800,
    margin=dict(t=50, b=40, l=40, r=40)
)

# Step 4: Streamlit display
st.plotly_chart(fig, use_container_width=True)
# st.caption("🔹 Bar chart shows product categories; line chart overlays payment mode preferences.")
# st.caption("🔹 Laptops & Mobile Accessories are the most popular, while Grocery is the least.")
# st.caption("🔹 Most users prefer Debit Cards, followed by Credit Cards. COD is rarely used.")
################################
churned = ecom_data[ecom_data["Churn"] == 1]

# # 2. Preferred Payment Mode
# fig2 = px.pie(churned, names="PreferredPaymentMode", 
#               color_discrete_sequence=px.colors.sequential.Reds)
# fig2.update_layout(title="Payment Methods Used by Churned Users", width=600, height=600)

# # 3. Marital Status
# fig3 = px.pie(churned, names="MaritalStatus", 
#               color_discrete_sequence=px.colors.sequential.Greens)
# fig3.update_layout(title="Marital Status of Churned Users", width=600, height=600)

# 5. Number of Orders (OrderCount)
# order_counts = churned['OrderCount'].value_counts().reset_index()
# order_counts.columns = ['OrderCount', 'Count']

# fig5 = px.pie(order_counts, names='OrderCount', values='Count', 
#               color_discrete_sequence=px.colors.sequential.Oranges)
# fig5.update_layout(title="Orders Placed by Churned Users", width=600, height=600)

device_counts = churned['PreferredLoginDevice'].value_counts().reset_index()
device_counts.columns = ['Category', 'Count']

city_counts = churned['CityTier'].value_counts().reset_index()
city_counts.columns = ['Category', 'Count']

satisfaction_counts = churned['SatisfactionScore'].value_counts().reset_index()
satisfaction_counts.columns = ['Category', 'Count']

# Normalize to percentage
device_counts['Type'] = 'PreferredLoginDevice'
city_counts['Type'] = 'CityTier'
satisfaction_counts['Type'] = 'SatisfactionScore'

combined = pd.concat([device_counts, city_counts, satisfaction_counts])
combined['Percentage'] = (combined['Count'] / combined.groupby('Type')['Count'].transform('sum')) * 100

# Plot
fig = px.bar(combined, x='Category', y='Percentage', color='Type', 
             barmode='group', title="Distribution of Churned Users by Category",
             height=500)

fig.update_layout(xaxis_title="Category", yaxis_title="Percentage (%)")

st.plotly_chart(fig, use_container_width=True)




import streamlit as st
import pandas as pd

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

# UI
# st.title("🛍️ Ecommerce Churn Prediction Dashboard")
# set_name = st.selectbox("Select a customer set", list(all_sets.keys()))

# if st.button("Predict Churn for Selected Set"):
#     customers = all_sets[set_name]

#     for idx, customer in enumerate(customers):
#         input_data = [int(customer[feature]) if isinstance(customer[feature], bool) else customer[feature]
#                       for feature in selected_features]

#         try:
#             response = requests.post(API_URL, json={"features": input_data})
#             result = response.json()
#         except Exception as e:
#             st.error(f"Failed to reach API: {e}")
#             continue

#         st.subheader(f"Customer #{idx + 1}")
#         col1, col2 = st.columns(2)

#         with col1:
#             customer_data = {k: v for k, v in customer.items() if k != "churn"}
#             df = pd.DataFrame(customer_data.items(), columns=["Feature", "Value"])
#             st.table(df)

#         with col2:
#             actual = "Churn" if customer.get("churn") == 1 else "Not Churn"
#             predicted = result.get("predicted_class", "Error")
#             prob = result.get("churn_probability", "-")

#             st.markdown(f"**Actual:** {actual}")
#             st.markdown(f"**Predicted:** {predicted}")
#             st.markdown(f"**Churn Probability:** `{prob}`")

#         st.markdown("---")
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
