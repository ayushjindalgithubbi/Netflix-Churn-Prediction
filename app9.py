# app.py - Netflix Churn Prediction (Enhanced UI/UX)
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
# NOTE: Path kept same as your original file – change if needed for deployment.
image_path = "C:/Users/Ayush Jindal/OneDrive/Desktop/Netflix Churn rate Prediction/netflix2.webp"
with open(image_path, "rb") as f:
    data = f.read()
encoded = base64.b64encode(data).decode()

page_bg = f"""
<style>
/* App background with dark overlay */
[data-testid="stAppViewContainer"] {{
    background: 
        linear-gradient(135deg, rgba(0,0,0,0.90), rgba(0,0,0,0.85)),
        url("data:image/webp;base64,{encoded}") no-repeat center center fixed;
    background-size: cover;
}}

/* Transparent header */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Hide default Streamlit menu & footer for cleaner look */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* Global text styling */
html, body, [class*="css"]  {{
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Title styling */
.main-title {{
    text-align: left;
    color: #FFFFFF;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 0;
}}
.main-subtitle {{
    color: #E5E5E5;
    font-size: 16px;
    margin-top: 4px;
}}

/* Glassmorphism-style card for the main content */
.glass-card {{
    background: rgba(20, 20, 20, 0.78);
    border-radius: 20px;
    padding: 24px 26px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
    border: 1px solid rgba(229, 9, 20, 0.35);
    backdrop-filter: blur(12px);
}}

/* Section headings */
.section-title {{
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}}

/* Labels / questions */
label, .stMarkdown > label {{
    font-weight: 600;
    color: #FFFFFF !important;
    font-size: 14px !important;
}}

/* Input fields */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {{
    background-color: rgba(18, 18, 18, 0.90);
    color: #F5F5F5;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 14px;
}}
.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus,
.stSelectbox>div>div>select:focus {{
    outline: none;
    border: 1px solid #E50914;
}}

/* Buttons */
div.stButton > button {{
    border-radius: 999px;
    padding: 10px 28px;
    font-weight: 600;
    border: none;
    font-size: 14px;
}}
div.stButton > button[kind="secondary"] {{
    background-color: transparent !important;
    color: #E5E5E5 !important;
    border: 1px solid #555555 !important;
}}
div.stButton > button[kind="secondary"]:hover {{
    background-color: #262626 !important;
}}
/* Primary button */
div.stButton > button:not([kind="secondary"]) {{
    background: linear-gradient(90deg, #E50914, #b20710) !important;
    color: white !important;
}}
div.stButton > button:not([kind="secondary"]):hover {{
    background: #ffffff !important;
    color: #E50914 !important;
}}

/* Churn result card */
.churn-card {{
    background: radial-gradient(circle at top left, #E50914 0, #141414 55%);
    border-radius: 22px;
    padding: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 18px 42px rgba(0,0,0,0.8);
}}
.churn-value {{
    font-size: 54px;
    font-weight: 800;
    margin-bottom: 4px;
}}
.churn-label {{
    font-size: 16px;
    opacity: 0.9;
}}

/* Business insights card */
.insights-card {{
    background: rgba(10,10,10,0.88);
    border-radius: 16px;
    padding: 18px 18px 8px 18px;
    border: 1px solid rgba(229, 9, 20, 0.4);
}}
.insights-card h3 {{
    color: #E50914;
    font-size: 18px;
    margin-bottom: 12px;
}}
.insights-card p {{
    color: #E0E0E0;
    font-size: 13px;
    margin-bottom: 6px;
}}

/* Badge style */
.badge-pill {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------- Title -----------------
header_col1, header_col2 = st.columns([2, 1])

with header_col1:
    st.markdown(
        """
        <h1 class="main-title">🎬 Netflix Customer Churn Prediction</h1>
        <p class="main-subtitle">
            Simulate a subscriber profile and instantly see how likely they are to churn, 
            along with key business levers that influence the decision.
        </p>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        """
        <div style="text-align:right; margin-top:8px;">
            <span class="badge-pill" style="background:#111111; color:#CCCCCC; border:1px solid #333333;">
                ML Powered • Churn Risk
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

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
# Kept exactly the same path & logic as your original file
pipeline = joblib.load("C:/Users/Ayush Jindal/OneDrive/Desktop/Netflix Churn rate Prediction/NetflixChurn_pipeline.pkl")

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

# ----------------- Main Layout: Inputs + Result -----------------
left_col, right_col = st.columns([1.4, 1.1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Subscriber Profile & Usage</div>', unsafe_allow_html=True)

    # Group inputs into logical sections with subtle separators
    basic_tab, usage_tab, account_tab = st.tabs(["👤 Profile", "📺 Usage", "💳 Account & Preferences"])

    with basic_tab:
        gender = center_input(
            st.selectbox,
            "Please select your Gender : ",
            ["Select", "Male", "Female", "Other"],
            key="gender",
        )

        age_input = center_input(
            st.text_input,
            "Please enter your Age : ",
            key="age",
            placeholder="Type your age",
        )
        age = None
        if age_input:
            try:
                age = int(age_input)
                if age < 10 or age > 100:
                    st.warning("Please enter age between 10 and 100")
                    age = None
            except ValueError:
                st.warning("Please enter a valid number for age")

        subscription_type = center_input(
            st.selectbox,
            "Please select your Subscription Type : ",
            ["Select", "Basic", "Standard", "Premium"],
            key="subscription_type",
        )

    with usage_tab:
        watch_hours_input = center_input(
            st.text_input,
            "How many Watch Hours per week?",
            key="watch_hours",
            placeholder="0-168",
        )
        watch_hours = None
        if watch_hours_input:
            try:
                watch_hours = float(watch_hours_input)
                if watch_hours < 0 or watch_hours > 168:
                    st.warning("Enter watch hours 0-168")
                    watch_hours = None
            except ValueError:
                st.warning("Enter a valid number")

        last_login_input = center_input(
            st.text_input,
            "How many days since last login?",
            key="last_login_days",
            placeholder="0-365",
        )
        last_login_days = None
        if last_login_input:
            try:
                last_login_days = int(last_login_input)
                if last_login_days < 0 or last_login_days > 365:
                    st.warning("Enter days 0-365")
                    last_login_days = None
            except ValueError:
                st.warning("Enter a valid number")

        avg_watch_time_per_day = center_input(
            st.number_input,
            "What is average watch time per day in hours?",
            min_value=0.0,
            max_value=24.0,
            step=0.1,
        )

    with account_tab:
        no_of_devices = center_input(
            st.selectbox,
            "Please enter number of devices are linked : ",
            [1, 2, 3, 4, 5],
            key="no_of_devices",
        )

        region = center_input(
            st.selectbox,
            "Please select your Region :",
            ["Select", "South America", "Europe", "North America", "Asia", "Africa", "Oceania"],
            key="region",
        )

        device = center_input(
            st.selectbox,
            "Please select your Device : ",
            ["Select", "Tablet", "Laptop", "Mobile", "TV", "Desktop"],
            key="device",
        )

        payment_method = center_input(
            st.selectbox,
            "Please select your Payment Method : ",
            ["Select", "Debit Card", "PayPal", "Crypto", "Gift Card", "Credit Card"],
            key="payment_method",
        )

        favorite_genre = center_input(
            st.selectbox,
            "Please select your Favourite Genre : ",
            ["Select", "Drama", "Documentary", "Romance", "Sci-Fi", "Horror", "Action", "Comedy"],
            key="favorite_genre",
        )

        number_of_profiles = center_input(
            st.selectbox,
            "Please select no of profiles : ",
            [1, 2, 3, 4, 5],
        )

    st.markdown("<hr style='border: 0.5px solid #333333; margin: 12px 0 16px 0;'>",
                unsafe_allow_html=True)

    # ----------------- Buttons -----------------
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

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
        st.success("All fields have been reset.")

    with btn_col1:
        st.button("Reset", on_click=reset_all, kwargs={}, type="secondary")

    with btn_col2:
        if st.button("Predict Churn", key="submit_btn"):
            if (
                gender == "Select"
                or subscription_type == "Select"
                or region == "Select"
                or device == "Select"
                or payment_method == "Select"
                or favorite_genre == "Select"
            ):
                st.warning("Please select valid options for all mandatory fields.")
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

                # --- Prediction ---
                churn_prob, color, message = predict_churn(user_input)
                st.session_state["churn_prob"] = churn_prob
                st.session_state["churn_color"] = color
                st.session_state["churn_message"] = message

                # --- Feature Importance (unchanged logic) ---
                model = pipeline.named_steps["model"]
                try:
                    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
                except:
                    feature_names = [
                        "age", "gender", "subscription_type", "watch_hours",
                        "last_login_days", "region", "device", "payment_method",
                        "number_of_profiles", "avg_watch_time_per_day", "favorite_genre"
                    ]

                importances = model.feature_importances_
                feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                top_features = feat_imp_df.sort_values(by="Importance", ascending=False).head(5)

                # --- Business Insights (same conditions, upgraded styling) ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)

                with st.container():
                    st.markdown('<div class="insights-card">', unsafe_allow_html=True)

                    for feature in top_features["Feature"]:
                        if "last_login_days" in feature:
                            st.markdown(
                                '<p>- Customers inactive for many days are more likely to churn. '
                                '<b>Action:</b> Trigger re-engagement emails or time-bound offers.</p>',
                                unsafe_allow_html=True,
                            )
                        elif "avg_watch_time_per_day" in feature:
                            st.markdown(
                                '<p>- Low daily engagement signals rising churn risk. '
                                '<b>Action:</b> Push personalized watchlists and reminders.</p>',
                                unsafe_allow_html=True,
                            )
                        elif "subscription_type" in feature:
                            st.markdown(
                                '<p>- Basic plan users may be more price sensitive and churn more often. '
                                '<b>Action:</b> Provide targeted upgrade offers to Standard/Premium.</p>',
                                unsafe_allow_html=True,
                            )
                        elif "region" in feature:
                            st.markdown(
                                '<p>- Some regions exhibit higher churn patterns. '
                                '<b>Action:</b> Localize content and marketing campaigns.</p>',
                                unsafe_allow_html=True,
                            )
                        elif "payment_method" in feature:
                            st.markdown(
                                '<p>- Gift card or non-recurring payment methods may lead to faster churn. '
                                '<b>Action:</b> Encourage auto-renewal or card-on-file options.</p>',
                                unsafe_allow_html=True,
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close glass-card

# ----------------- Display Churn on Right Side -----------------
with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Churn Risk Summary</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "churn_prob" in st.session_state:
        churn_prob = st.session_state["churn_prob"]
        color = st.session_state["churn_color"]
        message = st.session_state["churn_message"]

        # Map color to gradient/badge
        if color == "red":
            badge_text = "High Risk"
            badge_bg = "#B81D24"
        else:
            badge_text = "Low Risk"
            badge_bg = "#46D369"

        st.markdown(
            f"""
            <div class="churn-card">
                <div class="badge-pill" style="background:{badge_bg}; color:#000; margin-bottom:8px;">
                    {badge_text}
                </div>
                <div class="churn-value">{churn_prob}%</div>
                <div class="churn-label">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <p style="color:#B3B3B3; font-size:13px;">
                Configure a subscriber on the left and click <b>Predict Churn</b> to see the risk score here.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
