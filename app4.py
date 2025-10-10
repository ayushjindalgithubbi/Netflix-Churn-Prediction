# app_final.py – Final Netflix Churn Prediction (Refined Stable Version)
import streamlit as st
import pandas as pd
import joblib
import time
import base64
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------- Dummy Classes for Compatibility -----------------
class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, lower=0.01, upper=0.99):
        self.cols = cols or []
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="Netflix Customer Churn Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- Background Styling -----------------
image_path = "netflix2.webp"
with open(image_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/webp;base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-color: black;
}}
[data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
/* Headings and labels */
h1, h2, h3, label, p, span {{
    color: white !important;
    text-shadow: 2px 2px 10px #E50914 !important;
}}
/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {{
    background-color: #111 !important;
    color: white !important;
    border: 1px solid #E50914 !important;
    border-radius: 6px;
}}
/* Buttons */
div.stButton > button {{
    background-color: #E50914 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 25px;
    padding: 10px 30px;
}}
div.stButton > button:hover {{
    background-color: white !important;
    color: #E50914 !important;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown(
    """
    <h1 style='color: #E50914; font-size: 80px; font-weight: bold; text-align:center;
    text-shadow: 3px 3px 8px black; margin-top:0px; padding-top:0px;'>
    Customer Churn Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------------- Load Model -----------------
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("🔄 Loading model..."):
        time.sleep(1)
        pipeline = joblib.load("NetflixChurn_pipeline.pkl")
    return pipeline

pipeline = load_model()

# ----------------- Prediction Logic -----------------
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

# ----------------- Center Input Helper -----------------
def center_input(widget_func, label, *args, **kwargs):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        return widget_func(label, *args, **kwargs)

# ----------------- Input Form -----------------
st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

gender = center_input(st.selectbox, "Please select your Gender : ", ["Select", "Male", "Female", "Other"], key="gender")
age_input = center_input(st.text_input, "Please enter your Age : ", key="age", placeholder="Type your age")
age = None
if age_input:
    try:
        age = int(age_input)
        if age < 10 or age > 100:
            st.warning("Please enter age between 10 and 100")
            age = None
    except ValueError:
        st.warning("Please enter a valid number for age")

subscription_type = center_input(st.selectbox, "Please select your Subscription Type : ", ["Select", "Basic", "Standard", "Premium"], key="subscription_type")
watch_hours_input = center_input(st.text_input, "How many Watch Hours per week?", key="watch_hours", placeholder="0-168")
watch_hours = None
if watch_hours_input:
    try:
        watch_hours = float(watch_hours_input)
        if watch_hours < 0 or watch_hours > 168:
            st.warning("Enter watch hours 0-168")
            watch_hours = None
    except ValueError:
        st.warning("Enter a valid number")

last_login_input = center_input(st.text_input, "How many days since last login?", key="last_login_days", placeholder="0-365")
last_login_days = None
if last_login_input:
    try:
        last_login_days = int(last_login_input)
        if last_login_days < 0 or last_login_days > 365:
            st.warning("Enter days 0-365")
            last_login_days = None
    except ValueError:
        st.warning("Enter a valid number")

no_of_devices = center_input(st.selectbox, "Please enter number of devices are linked : ", [1,2,3,4,5], key="no_of_devices")
region = center_input(st.selectbox, "Please select your Region :", ["Select", "South America", "Europe", "North America", "Asia", "Africa", "Oceania"], key="region")
device = center_input(st.selectbox, "Please select your Device : ", ["Select", "Tablet", "Laptop", "Mobile", "TV", "Desktop"], key="device")
payment_method = center_input(st.selectbox, "Please select your Payment Method : ", ["Select", "Debit Card", "PayPal", "Crypto", "Gift Card", "Credit Card"], key="payment_method")
favorite_genre = center_input(st.selectbox, "Please select your Favourite Genre : ", ["Select", "Drama", "Documentary", "Romance", "Sci-Fi", "Horror", "Action", "Comedy"], key="favorite_genre")
avg_watch_time_per_day = center_input(st.number_input, "What is average watch time per day in hours?", min_value=0.0, max_value=24.0, step=0.1)
number_of_profiles = center_input(st.selectbox, "Please select no of profiles : ", [1,2,3,4,5])

# ----------------- Buttons -----------------
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
col3, col1, col2, col4 = st.columns([2.2,0.8,0.8,1.8])

def reset_all():
    st.session_state.update({
        "gender": "Select",
        "subscription_type": "Select",
        "age": "",
        "watch_hours": "",
        "last_login_days": "",
        "no_of_devices": 1,
        "region": "Select",
        "device": "Select",
        "payment_method": "Select",
        "favorite_genre": "Select",
        "avg_watch_time_per_day": 0.0,
        "number_of_profiles": 1
    })
    st.success("Data reset successfully!")

with col1:
    st.button("Reset", on_click=reset_all)

with col2:
    if st.button("Submit"):
        if gender=="Select" or subscription_type=="Select" or region=="Select" or device=="Select" or payment_method=="Select" or favorite_genre=="Select":
            st.warning("Please select valid options for all fields")
        else:
            user_input = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "subscription_type": subscription_type,
                "watch_hours": watch_hours,
                "last_login_days": last_login_days,
                "no_of_devices": no_of_devices,
                "region": region,
                "device": device,
                "payment_method": payment_method,
                "favorite_genre": favorite_genre,
                "avg_watch_time_per_day": avg_watch_time_per_day,
                "number_of_profiles": number_of_profiles
            }])
            churn_prob, color, message = predict_churn(user_input)
            st.session_state.update({
                "churn_prob": churn_prob,
                "churn_color": color,
                "churn_message": message,
                "user_input": user_input
            })

# ----------------- Display Result -----------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
if "churn_prob" in st.session_state:
    churn_prob = st.session_state["churn_prob"]
    color = st.session_state["churn_color"]
    message = st.session_state["churn_message"]

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px;'>
            <h1 style='color:{color}; font-size:80px; font-weight:bold; text-shadow: 3px 3px 10px black;'>{churn_prob}%</h1>
            <h3 style='color:{color}; text-shadow: 1px 1px 5px black;'>{message}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Business Insights ----
    if churn_prob >= 75:
        if st.button("Show Business Insights"):
            st.markdown("""
            <div style='background-color: rgba(255,0,0,0.1); padding: 20px; border-radius: 10px;'>
            <h3 style='color:#E50914;'>High Risk Customer Insights:</h3>
            <ul style='color:white; font-size:18px;'>
                <li>Churn Probability is high. Consider retention offers or loyalty discounts.</li>
                <li>Engagement or Watch hours are low — suggest personalized recommendations.</li>
                <li>Subscription Type: Check upgrade incentives.</li>
                <li>Favorite Genre: Promote related content to re-engage interest.</li>
                <li>Region: Focus targeted marketing offers or emails.</li>
                <li>Average Watch Time: Encourage binge sessions via curated lists.</li>
                <li>Multiple Profiles: Enable personalized watchlists for better stickiness.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
