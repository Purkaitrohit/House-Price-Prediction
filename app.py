import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import FunctionTransformer


def apply_binary_mapper(df):
    def binary_mapper(x):
        return x.map({'yes': 1, 'no': 0, 'Required': 1, 'Not Required': 0}).fillna(x)
    return df.apply(binary_mapper)

binary_transformer = FunctionTransformer(apply_binary_mapper)

# --- Load the trained pipeline ---
model = joblib.load("house_price_rf_pipeline.pkl")

# --- Page Config ---
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

# --- App Header ---
st.markdown(
    """
    <h1 style="text-align: center; color: #2E86C1;">
        🏡 Smart House Price Prediction
    </h1>
    <p style="text-align: center; color: #566573; font-size:18px;">
        Enter house details and get an instant AI-powered price estimate
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# --- Input Sections ---
st.markdown("<h4 style='text-align:center;'>📐 House Details</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("📏 Area (sq ft)", min_value=1600, max_value=16000, value=3000, step=50)
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=6, value=2)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=4, value=2)
with col2:
    stories = st.number_input("🏢 Stories", min_value=1, max_value=4, value=2)
    furnishingstatus = st.selectbox("🛋 Furnishing Status", ["semi-furnished", "unfurnished", "furnished"])
    price_category = st.selectbox("💰 Price Category", ["Low Pricing", "Medium Pricing", "High Pricing"])

st.markdown("<h4 style='text-align:center;'>🏗 Amenities</h4>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    mainroad = st.selectbox("🛣 Main Road Access", ["Required", "Not Required"])
    guestroom = st.selectbox("🚪 Guest Room", ["Required", "Not Required"])
    basement = st.selectbox("🏚 Basement", ["Required", "Not Required"])
with col4:
    hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["Required", "Not Required"])
    airconditioning = st.selectbox("❄ Air Conditioning", ["Required", "Not Required"])
    parking = st.selectbox("🚗 Parking", ["Required", "Not Required"])

# --- Preferred Area (Centered) ---
st.markdown("<h4 style='text-align:center;'>🌆 Preferred Area</h4>", unsafe_allow_html=True)
col_center = st.columns([1, 2, 1])  # left, center, right
with col_center[1]:
    st.markdown(
        "<p style='text-align:center; font-size:16px; color:#34495E;'>Choose if you want the house in a prime city location</p>",
        unsafe_allow_html=True
    )
    prefarea = st.selectbox("", ["Required", "Not Required"])
    
# --- Prepare DataFrame ---
input_data = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus,
    "price_category": price_category
}])

# --- Prediction Section ---
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>✨ Get Your Predicted House Price Estimate</h3>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1B4F72;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center Predict Button
btn_center = st.columns([1, 2, 1])
with btn_center[1]:
    if st.button("🔮 Predict Price", use_container_width=True):
        prediction = model.predict(input_data)[0]

        st.balloons()
        st.markdown(
            f"""
            <div style="
                background-color:#D6EAF8;
                padding:20px;
                border-radius:15px;
                border: 2px solid #2E86C1;
                text-align:center;
                width:100%;
            ">
                <h3 style="color:#1F618D;">💡 Estimated Price</h3>
                <h2 style="color:#117A65;">₹ {prediction:,.2f}</h2>
                <p style="color:#7D3C98;">Based on the features you selected</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
