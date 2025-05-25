from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/xgb_churn_model.pkl")

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

class CustomerFeatures(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Churn Prediction API is live!"}

@app.post("/predict")
def predict(data: CustomerFeatures):
    try:
        if len(data.features) != len(selected_features):
            return {"error": f"Expected {len(selected_features)} features, got {len(data.features)}"}

        X_input = np.array(data.features).reshape(1, -1)
        prob = model.predict_proba(X_input)[0][1]  # likely a numpy.float32
        prediction = int(prob >= 0.55)

        # 🔥 Fix: Cast to standard Python types
        return {
            "churn_probability": round(float(prob), 3),  # cast to float
            "predicted_class": "Churn" if prediction else "Not Churn"
        }

    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}
