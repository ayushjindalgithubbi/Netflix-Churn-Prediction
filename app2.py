# app.py - Netflix Churn Prediction
import streamlit as st
import pandas as pd
import joblib
import time
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

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
    background-color: white;
}}
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------- Title -----------------
st.markdown(
    """
    <h1 style='color: #E50914; font-size: 80px; font-weight: bold; text-align:center;
    text-shadow: 2px 2px 5px black; margin-top:0px; padding-top:0px;'>
    Customer Churn Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

# ----------------- Pipeline Helper Classes -----------------
class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        self.num_median = X[self.numeric_cols].median()
        self.cat_mode = X[self.categorical_cols].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = X[self.numeric_cols].fillna(self.num_median)
        X[self.categorical_cols] = X[self.categorical_cols].fillna(self.cat_mode)
        return X

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, cols, lower=0.01, upper=0.99):
        self.cols = cols
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.bounds = {}
        for col in self.cols:
            self.bounds[col] = (
                np.quantile(X[col], self.lower),
                np.quantile(X[col], self.upper)
            )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            low, high = self.bounds[col]
            X[col] = np.clip(X[col], low, high)
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["inactive_flag"] = X["last_login_days"].apply(lambda x: 1 if x > 30 else 0)
        X["engagement_ratio"] = X["watch_hours"] / (X["last_login_days"] + 1)
        return X

# ----------------- Load Model -----------------
pipeline = joblib.load("NetflixChurn_pipeline.pkl")

def predict_churn(user_input_df):
    prob = pipeline.predict_proba(user_input_df)[:, 1][0] * 100
    if prob > 65:
        color = "red"
        message = "⚠ High Risk! Consider reaching out to the customer."
    else:
        color = "green"
        message = "✔ Low Risk"
    return round(prob, 2), color, message

# ----------------- Helper to Center Inputs -----------------
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
col3,col1, col2,col4 = st.columns([2.2,0.8,0.8,1.8])

def reset_all():
    # Clear all inputs
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
        "number_of_profiles": 1,
        # Clear prediction and insights
        "churn_prob": None,
        "churn_color": None,
        "churn_message": None,
        "show_insights": False
    })


with col1:
    if st.button("Reset", on_click=reset_all):
        st.success("Data reset successfully!", icon="✅")


with col2:
    if st.button("Submit"):
        # Check if all required inputs are selected
        if (
            gender == "Select" or 
            subscription_type == "Select" or 
            region == "Select" or 
            device == "Select" or 
            payment_method == "Select" or 
            favorite_genre == "Select"
        ):
            st.warning("Please select valid options for all fields")
        else:
            # Prepare user input DataFrame
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

            # Make prediction
            churn_prob, color, message = predict_churn(user_input)

            # Store results in session_state
            st.session_state["churn_prob"] = churn_prob
            st.session_state["churn_color"] = color
            st.session_state["churn_message"] = message

            # Enable business insights button
            st.session_state["show_insights"] = True


# ----------------- Display Churn -----------------
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
if st.session_state.get("churn_prob") is not None:
    churn_prob = st.session_state["churn_prob"]
    color = st.session_state["churn_color"]
    message = st.session_state["churn_message"]

    st.markdown(
        f"""
        <div style='text-align:center; margin-top:20px;'>
            <h1 style='color:{color}; font-size:80px; font-weight:bold;'>{churn_prob}%</h1>
            <h3 style='color:{color};'>{message}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------- Business Insights Button -----------------
if "churn_prob" in st.session_state:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    # Business Insights button
    if st.button("Show Business Insights"):
        # Check for missing values
        required_fields = [age, gender, subscription_type, watch_hours, last_login_days,
                           no_of_devices, region, device, payment_method, favorite_genre,
                           avg_watch_time_per_day, number_of_profiles]
        if None in required_fields or "Select" in [gender, subscription_type, region, device, payment_method, favorite_genre]:
            st.warning("⚠ Please fill all the details to view Business Insights!")
        else:
            churn_prob = st.session_state["churn_prob"]
            color = st.session_state["churn_color"]

            # Extract input values for insights
            user_data = {
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
                "number_of_profiles": number_of_profiles,
                # Clear previous prediction and insights
                "churn_prob": None,
                "churn_color": None,
                "churn_message": None,
                "business_insights": None
            }
            
            
            # Generate insights



            insights = ""
            if churn_prob > 65:
                insights = f"""
                <div style='background-color:#FFE5E5; padding:20px; border-radius:15px'>
                    <h3 style='color:#E50914'>High Risk Customer Insights:</h3>
                    <ul>
                        <li>Churn Probability: <b>{round(churn_prob,2)}%</b>. Consider retention offers or discounts.</li>
                        <li>Engagement: Watch hours per week = <b>{round(watch_hours,0)}</b>. Suggest personalized recommendations.</li>
                        <li>Subscription: <b>{subscription_type}</b>. Upsell premium features or add-ons.</li>
                        <li>Favorite Genre: <b>{favorite_genre}</b>. Send curated content emails.</li>
                        <li>Region: <b>{region}</b>. Regional promotions can increase loyalty.</li>
                        <li>Average Daily Watch Time: <b>{round(avg_watch_time_per_day,2)} hrs</b>. Suggest binge-watching guides.</li>
                        <li>Number of Profiles: <b>{number_of_profiles}</b>. Encourage family/friend sharing for retention.</li>
                    </ul>
                </div>
                """
            else:
                insights = f"""
                <div style='background-color:#E5FFE5; padding:20px; border-radius:15px'>
                    <h3 style='color:green'>Low Risk Customer Insights:</h3>
                    <ul>
                        <li>Churn Probability: <b>{round(churn_prob,2)}%</b>. Maintain engagement.</li>
                        <li>Engagement: Watch hours per week = <b>{round(watch_hours,2)}</b>. Recommend new content similar to <b>{favorite_genre}</b>.</li>
                        <li>Subscription: <b>{subscription_type}</b>. Consider loyalty rewards or early access perks.</li>
                        <li>Average Daily Watch Time: <b>{round(avg_watch_time_per_day,2)} hrs</b>. Suggest content bundles.</li>
                        <li>Number of Profiles: <b>{number_of_profiles}</b>. Upsell family plan or extra profiles.</li>
                    </ul>
                </div>
                """
            st.markdown(insights, unsafe_allow_html=True)

st.markdown("""
<style>
/* ----------------- Labels / Questions ----------------- */
/* Center all question labels and make them red */
label, .stMarkdown > label {
    font-weight: bold;
    color: #E50914 !important;  /* red color */
    font-size: 48px !important;
    text-align: center !important;
    display: block;
}

/* ----------------- Input Fields ----------------- */
/* Center text inputs, number inputs, select boxes */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {
    display: block;
    margin-left: auto;
    margin-right: auto;
    color: dark grey;
    background-color: #ffffff;
    border: 0.5px solid #444444;
    border-radius: 5px;
    padding: 8px;
    font-size: 20px;
}

/* Center sliders */
.stSlider>div>div>input {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Center checkboxes and radio buttons */
.stCheckbox>div, .stRadio>div {
    display: flex;
    justify-content: center;
}

/* ----------------- Buttons ----------------- */
/* Center all buttons */
div.stButton > button {
    display: block;
    margin-left: auto;
    margin-right: auto;
    background-color: white !important;
    color: #444444 !important;
    font-weight: bold;
    font-size: 22px;
    padding: 12px 30px;
    border-radius: 20px;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

/* Hover effect for buttons */
div.stButton > button:hover {
    background-color: #E50914 !important;
    color: white !important;
}

/* ----------------- Optional: spacing between fields ----------------- */
.stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stCheckbox, .stRadio {
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


            
            


            
            