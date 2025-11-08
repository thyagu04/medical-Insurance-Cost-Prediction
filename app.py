import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------
st.set_page_config(
    page_title="🏥 Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
)

# -----------------------------------------------------
# CUSTOM CSS - PREMIUM DEEP BLUE THEME
# -----------------------------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #001F3F 0%, #001233 50%, #000814 100%);
        color: #F1FAEE !important;
        font-family: 'Poppins', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001845, #001233);
        color: white;
    }
    .title {
        text-align: center;
        color: #00B4D8;
        font-size: 50px;
        font-weight: 900;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        text-shadow: 2px 2px 8px #023E8A;
    }
    .subtitle {
        text-align: center;
        color: #ADE8F4;
        font-size: 18px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    hr {
        border: 1px solid #00B4D8;
        border-radius: 5px;
        margin: 25px 0;
    }
    .result-box {
        background: linear-gradient(145deg, #023E8A, #0077B6);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 700;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
    }
    .result-box:hover {
        transform: scale(1.05);
    }
    .info-card {
        background-color: rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00B4D8;
        padding: 15px;
        margin-top: 15px;
        border-radius: 10px;
        font-size: 16px;
        color: #CAF0F8;
    }
    label, .stNumberInput label, .stSelectbox label {
        color: #CAF0F8 !important;
        font-weight: 600 !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00B4D8, #0077B6);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 12px 40px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 12px rgba(0,180,216,0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #48CAE4, #00B4D8);
        transform: scale(1.05);
    }
    .sidebar-title {
        color: #90E0EF;
        font-size: 22px;
        font-weight: 800;
        text-align: center;
    }
    .sidebar-section {
        background-color: rgba(255,255,255,0.05);
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 10px;
        color: #CAF0F8;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SIDEBAR DESIGN
# -----------------------------------------------------
st.sidebar.markdown("<h1 class='sidebar-title'>🎀 Medical Insurance Predictor</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='sidebar-section'>
✨ <b>Welcome!</b><br>
This AI-powered app predicts your <b>Medical Insurance Cost</b> 💰 using Machine Learning.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class='sidebar-section'>
🧠 <b>How to Use:</b><br>
1️⃣ Fill in your details<br>
2️⃣ Click <b>Predict Now 🚀</b><br>
3️⃣ View your estimate instantly
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------
model_path = os.path.join(os.path.dirname(__file__), "insurance_model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.sidebar.success("✅ Model Loaded Successfully!")
else:
    st.sidebar.error("❌ Model file missing! Put 'insurance_model.pkl' in the project folder.")
    st.stop()

# -----------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------
st.markdown("<h1 class='title'>💰 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Fill in your details to estimate your insurance cost instantly ⚡</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", 0, 120, 30)
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", 10.0, 60.0, 25.5, step=0.1)
    children = st.number_input("👶 Dependents", 0, 10, 1)

with col2:
    sex = st.selectbox("🧬 Gender", ["Male", "Female"])
    smoker = st.selectbox("🚬 Smoker", ["Yes", "No"])
    region = st.selectbox("🌍 Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# -----------------------------------------------------
# ✅ FIXED ENCODING (6 FEATURES)
# -----------------------------------------------------
def encode_input(age, sex, bmi, children, smoker, region):
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region = region_map[region]
    return np.array([[age, sex, bmi, children, smoker, region]])

# -----------------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------------
st.markdown("### 🚀 Predict Your Cost")

if st.button("🔮 Predict Now"):
    try:
        data = encode_input(age, sex, bmi, children, smoker, region)
        prediction = model.predict(data)[0]

        st.markdown(
            f"<div class='result-box'>💵 <b>Estimated Insurance Cost:</b><br><br> 🩺 ${prediction:,.2f}</div>",
            unsafe_allow_html=True,
        )
        st.balloons()

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
