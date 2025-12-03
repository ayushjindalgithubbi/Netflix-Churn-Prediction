# Netflix Churn Prediction (Enhanced UI/UX Version)
# Functionality unchanged, UI/UX improved

import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(
    page_title="Netflix Customer Churn Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Background image
image_path = "netflix2.webp"
with open(image_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, rgba(0,0,0,0.9), rgba(0,0,0,0.85)),
    url("data:image/webp;base64,{encoded}") no-repeat center center fixed;
    background-size: cover;
}}
[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

.glass-card {{
    background: rgba(20, 20, 20, 0.85);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
}}

.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {{
    background-color: rgba(18, 18, 18, 0.9);
    color: white;
    border-radius: 10px;
}}

div.stButton > button {{
    border-radius: 30px;
    background: linear-gradient(90deg, #E50914, #b20710);
    color: white;
}}

.churn-card {{
    background: linear-gradient(135deg, #E50914, #141414);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    color: white;
}}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1 style='color:white'>🎬 Netflix Customer Churn Prediction</h1>", unsafe_allow_html=True)

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, lower=0.01, upper=0.99):
        self.cols = cols or []
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

@st.cache_resource
def load_model():
    return joblib.load("NetflixChurn_pipeline.pkl")

pipeline = load_model()

def predict_churn(user_input_df):
    prob = pipeline.predict_proba(user_input_df)[:, 1][0] * 100
    if prob >= 75:
        color = "red"
        message = "⚠ High Risk! Consider reaching out to the customer."
    elif prob >= 50:
        color = "orange"
        message = "🟠 Moderate Risk. Offer incentives to retain."
    else:
        color = "green"
        message = "✔ Low Risk. Customer likely to stay."
    return round(prob, 2), color, message

def center_input(widget_func, label, *args, **kwargs):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        return widget_func(label, *args, **kwargs)

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

gender = center_input(st.selectbox, "Gender", ["Select", "Male", "Female", "Other"])
age_input = center_input(st.text_input, "Age")
subscription_type = center_input(st.selectbox, "Subscription", ["Select", "Basic", "Standard", "Premium"])
watch_hours_input = center_input(st.text_input, "Watch Hours / Week")
last_login_input = center_input(st.text_input, "Days Since Last Login")
no_of_devices = center_input(st.selectbox, "Number of Devices", [1,2,3,4,5])
region = center_input(st.selectbox, "Region", ["Select", "Asia", "Europe", "Africa", "North America"])
device = center_input(st.selectbox, "Device", ["Select", "Mobile", "TV", "Laptop", "Tablet"])
payment_method = center_input(st.selectbox, "Payment", ["Select", "Card", "UPI", "Wallet"])
favorite_genre = center_input(st.selectbox, "Genre", ["Select", "Drama", "Action", "Comedy"])
avg_watch_time_per_day = center_input(st.number_input, "Avg Watch Time per Day", 0.0, 24.0)
number_of_profiles = center_input(st.selectbox, "Profiles", [1,2,3,4,5])

if st.button("Predict Churn"):
    try:
        user_input = pd.DataFrame([{
            "age": int(age_input),
            "gender": gender,
            "subscription_type": subscription_type,
            "watch_hours": float(watch_hours_input),
            "last_login_days": int(last_login_input),
            "no_of_devices": no_of_devices,
            "region": region,
            "device": device,
            "payment_method": payment_method,
            "favorite_genre": favorite_genre,
            "avg_watch_time_per_day": avg_watch_time_per_day,
            "number_of_profiles": number_of_profiles
        }])

        churn_prob, color, message = predict_churn(user_input)

        st.markdown(
            f"""
            <div class='churn-card'>
                <h2>{churn_prob}%</h2>
                <p>{message}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error("Please fill all the fields correctly.")

st.markdown("</div>", unsafe_allow_html=True)
