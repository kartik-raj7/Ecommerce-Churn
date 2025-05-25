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
st.markdown("""
### 💡 What is Customer Churn?
Customer churn refers to when a customer stops using a company's product or service. Identifying customers at risk of churning can help businesses take proactive measures to retain them.

In this project, we use a machine learning model trained on e-commerce customer data to predict churn likelihood.

### 🔍 What You'll Find in This App:
- Explore the data and key insights.
- Visualize patterns related to churn.
- Input customer information and get churn predictions.
""")
st.markdown("📊 Sample Data")
st.dataframe(ecom_data.head())

st.markdown("### 📈 Dataset Summary")
st.write("Number of rows:", ecom_data.shape[0])
st.write("Number of columns:", ecom_data.shape[1])
st.write("Churn rate in dataset:")
st.write(ecom_data['Churn'].value_counts(normalize=True).map("{:.2%}".format))

col1, col2 = st.columns(2)

with col1:
    # Bar chart for 'PreferedOrderCat' (note: double-check spelling, your data column name might be "PreferedOrderCat" or "PreferredOrderCat")
    order_cat_counts = ecom_data['PreferedOrderCat'].value_counts().reset_index()
    order_cat_counts.columns = ['OrderCategory', 'Count']

    fig4 = px.bar(order_cat_counts, 
                  x='OrderCategory', 
                  y='Count', 
                  color='Count',
                  color_continuous_scale='Oranges',
                  title="Popular Order Categories",
                  labels={'Count': 'Number of Orders', 'OrderCategory': 'Order Category'},
                  height=300, width=700)
    fig4.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("🔹 Laptops & Mobile Accessories are the most popular, while Grocery is the least.")

with col2:
    # Pie chart for 'PreferredPaymentMode'
    payment_mode_counts = ecom_data['PreferredPaymentMode'].value_counts().reset_index()
    payment_mode_counts.columns = ['PaymentMode', 'Count']

    fig2 = px.pie(payment_mode_counts, 
                  names='PaymentMode', 
                  values='Count', 
                  color_discrete_sequence=px.colors.sequential.Reds,
                  title="Payment Mode Preferences")
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=500, width=700)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔹 Most users prefer Debit Cards, followed by Credit Cards. COD is rarely used.")


    
churned = ecom_data[ecom_data["Churn"] == 1]

# 1. Preferred Login Device
fig1 = px.pie(churned, names="PreferredLoginDevice", 
              color_discrete_sequence=px.colors.sequential.Blues)
fig1.update_layout(title="Login Device Preference Among Churned Users", width=600, height=600)

# 2. Preferred Payment Mode
fig2 = px.pie(churned, names="PreferredPaymentMode", 
              color_discrete_sequence=px.colors.sequential.Reds)
fig2.update_layout(title="Payment Methods Used by Churned Users", width=600, height=600)

# 3. Marital Status
fig3 = px.pie(churned, names="MaritalStatus", 
              color_discrete_sequence=px.colors.sequential.Greens)
fig3.update_layout(title="Marital Status of Churned Users", width=600, height=600)

# 4. City Tier
fig4 = px.pie(churned, names="CityTier", 
              color_discrete_sequence=px.colors.sequential.Purples)
fig4.update_layout(title="City Tier of Churned Users", width=600, height=600)

# 5. Number of Orders (OrderCount)
# order_counts = churned['OrderCount'].value_counts().reset_index()
# order_counts.columns = ['OrderCount', 'Count']

# fig5 = px.pie(order_counts, names='OrderCount', values='Count', 
#               color_discrete_sequence=px.colors.sequential.Oranges)
# fig5.update_layout(title="Orders Placed by Churned Users", width=600, height=600)

#6. Satisfaction Score

fig6 = px.pie(churned, names="SatisfactionScore", 
              color_discrete_sequence=px.colors.sequential.Purples)
fig6.update_layout(title="Satisfaction Score of Churned Users", width=600, height=600)
# Display in two rows
# st.plotly_chart(fig1, use_container_width=True)
# st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# with col3:
st.plotly_chart(fig1, use_container_width=True)
st.caption("🔹 36.7% of users who used mobile phone as preferred login device have churned followed by people using computer accounting for 34.2%")
    
# with col4:
#     st.plotly_chart(fig2, use_container_width=True) 
    
   

# with col1:
#     st.plotly_chart(fig3, use_container_width=True)
   
# with col2:
st.plotly_chart(fig4, use_container_width=True)
st.caption("🔹 Users belonging to Tier-1 city have the most churn rate accounting for 56.1% of all churned users followed by people belonging to Tier 3 cities (38.8%)")



# Full width below
# with col5:
#     st.plotly_chart(fig5, use_container_width=True)
#     
# with col6:
st.plotly_chart(fig6, use_container_width=True)
st.caption("🔹 Users with staisfaction score 3 and above accounted for 78% users")
st.caption("🔹 Users who placed one or two orders churned more ")
st.caption("🔹 Users who have churned and are single accounted for 50.6% of users churned")
st.caption("🔹 Users Using Debit Card as Mode of Payment have the highest percentage of churn accounting for 36.7% followed by 20.7% using Credit Card that means people using cards for payment are more churn prone than people using other modes of payments")



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

st.markdown("### Feature Selection Summary")
st.write(
    "I applied multiple feature selection algorithms and combined their results by consensus scoring to identify the most impactful features for churn prediction."
)

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

st.markdown("### Model Selection & Evaluation")

st.write(
    "We trained and evaluated multiple machine learning models to predict customer churn. "
    "Performance was compared based on key metrics like Accuracy, Precision, Recall, and F1 Score."
)

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

        accuracy                           0.97      1126
       macro avg       0.95      0.92      0.94      1126
    weighted avg       0.96      0.97      0.96      1126
            
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
st.title("🛍️ Ecommerce Churn Prediction Dashboard")
set_name = st.selectbox("Select a customer set", list(all_sets.keys()))

if st.button("Predict Churn for Selected Set"):
    customers = all_sets[set_name]

    for idx, customer in enumerate(customers):
        input_data = [int(customer[feature]) if isinstance(customer[feature], bool) else customer[feature]
                      for feature in selected_features]

        try:
            response = requests.post(API_URL, json={"features": input_data})
            result = response.json()
        except Exception as e:
            st.error(f"Failed to reach API: {e}")
            continue

        st.subheader(f"Customer #{idx + 1}")
        col1, col2 = st.columns(2)

        with col1:
            customer_data = {k: v for k, v in customer.items() if k != "churn"}
            df = pd.DataFrame(customer_data.items(), columns=["Feature", "Value"])
            st.table(df)

        with col2:
            actual = "Churn" if customer.get("churn") == 1 else "Not Churn"
            predicted = result.get("predicted_class", "Error")
            prob = result.get("churn_probability", "-")

            st.markdown(f"**Actual:** {actual}")
            st.markdown(f"**Predicted:** {predicted}")
            # st.markdown(f"**Churn Probability:** `{prob}`")

        st.markdown("---")
