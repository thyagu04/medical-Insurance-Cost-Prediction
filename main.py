# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import json
import requests
import time

# Configure page with wide layout
st.set_page_config(
    page_title="🏥 Medical Insurance Cost Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with beautiful gradient background
st.markdown("""
<style>
/* Keep layout and input/button styles but avoid changing app background or global theme */

/* Main content container: neutral, semi-opaque so contents remain distinct */
.main .block-container {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 14px;
    padding: 2.4rem;
    margin: 2rem auto;
    box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    max-width: 1200px;
}

/* Headers */
.main-header {
    font-size: 2.2rem;
    color: #0b2545 !important;
    text-align: center;
    margin-bottom: 0.25rem;
    font-weight: 800;
}

.sub-header {
    font-size: 1.1rem;
    color: #4b5563;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Prediction box & feature cards */
.prediction-box {
    background: rgba(255,255,255,0.98);
    border-radius: 12px;
    padding: 1.5rem;
    color: #0b2545;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    text-align: center;
}

.feature-card {
    background: rgba(255,255,255,0.96);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* Metric card - keep accent but neutral overall theme */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1rem;
    color: white;
    text-align: center;
}

/* Buttons - modern but not theming the whole app */
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

/* Inputs - thin and modern */
input[type="number"], input[type="text"], .stNumberInput input {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    color: #0b2545 !important;
    box-shadow: 0 2px 6px rgba(2,6,23,0.03) !important;
}

/* Hide any range inputs if present (defensive) */
input[type="range"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- helper: model loader ---
@st.cache_resource
def load_model_from_path(path: str):
    return joblib.load(path)

# --- Load Lottie animation ---
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
lottie_health = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_5njp3vgg.json")
lottie_doctor = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_k6myfzbd.json")

# --- mappings used in the notebook ---
SEX_MAP = {"Male": 0, "Female": 1}
SMOKER_MAP = {"Yes": 0, "No": 1}
REGION_MAP = {
    "Southeast": 0,
    "Southwest": 1,
    "Northeast": 2,
    "Northwest": 3
}

# Header with animation and floating effect
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header" style="color: #FF6F61; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 8px;">Predict healthcare costs with advanced machine learning</p>', unsafe_allow_html=True)
    # Animated header with multiple animations
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    anim_col1, anim_col2 = st.columns(2)
    with anim_col1:
        if lottie_health:
            st_lottie(lottie_health, height=180, key="health")
    with anim_col2:
        if lottie_doctor:
            st_lottie(lottie_doctor, height=180, key="doctor")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Enter the patient details below and click **Predict** to get instant insurance cost estimates.")

# Sidebar: model selection / upload with enhanced styling
with st.sidebar:
    st.header("🔧 Model Configuration")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("The app will try to load `insurance_model.pkl` from the app folder. You can also upload a model file.")
    default_model_path = "insurance_model.pkl"
    model = None
    model_status = st.empty()

    uploaded_file = st.file_uploader("📁 Upload a model (.pkl)", type=["pkl", "joblib"])
    if uploaded_file is not None:
        # Show upload progress
        with st.spinner("🔄 Uploading and loading model..."):
            time.sleep(1)
            # save to a temp file and load
            tmp_path = os.path.join(".", "uploaded_insurance_model.pkl")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                model = load_model_from_path(tmp_path)
                model_status.success("✅ Model loaded successfully!")
            except Exception as e:
                model_status.error(f"❌ Failed to load model: {e}")
    else:
        if os.path.exists(default_model_path):
            try:
                model = load_model_from_path(default_model_path)
                model_status.success(f"✅ Model loaded from `{default_model_path}`.")
            except Exception as e:
                model_status.error(f"❌ Failed to load `{default_model_path}`: {e}")
        else:
            model_status.info(f"ℹ️ No local model found. Upload a model to get started.")
    
    # Add some metrics in sidebar
    st.markdown("---")
    st.metric("Models Loaded", "1" if model else "0", delta="Ready" if model else "Waiting")
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
# --- Input form ---
with st.form(key="input_form"):
    st.subheader("👤 Patient Information")
    
    # Arrange input features in two columns with 3 inputs per column (3-3 pairs)
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("#### 🎂 Basic Metrics")
        # Left column: Age, BMI, Sex
        age = st.number_input("Age", min_value=0, max_value=120, value=37, step=1, help="Patient's age in years")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=26.79, step=0.1, format="%.1f", help="Body Mass Index")
        sex = st.selectbox("Sex", options=list(SEX_MAP.keys()), index=0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown("#### 👨‍👩‍👧‍👦 Additional Details")
        # Right column: Children, Smoker, Region
        children = st.number_input("Children (count)", min_value=0, max_value=20, value=0, step=1, help="Number of children/dependents covered by insurance")
        smoker = st.selectbox("Smoker?", options=list(SMOKER_MAP.keys()), index=1)
        region = st.selectbox("Region", options=list(REGION_MAP.keys()), index=0)
        
        # Fun visualization for children (keeps behavior)
        try:
            children_int = int(children)
        except Exception:
            children_int = 0

        if children_int > 0:
            child_emoji = "👶" * min(children_int, 5) + ("+" if children_int > 5 else "")
            st.write(f"**Dependents:** {child_emoji}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        submit = st.form_submit_button("🚀 Predict Insurance Cost", use_container_width=True)

# Prediction logic
def prepare_input(age, sex, bmi, children, smoker, region):
    """
    Returns a 2D numpy array in the same order the model expects:
    [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]
    """
    sex_enc = SEX_MAP[sex]
    smoker_enc = SMOKER_MAP[smoker]
    region_enc = REGION_MAP[region]
    arr = np.array([age, sex_enc, bmi, children, smoker_enc, region_enc], dtype=float).reshape(1, -1)
    return arr

def create_gauge_chart(value, min_val=0, max_val=50000):
    """Create a gauge chart for the prediction"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Insurance Cost (USD)", 'font': {'size': 20, 'color': '#2c3e50'}},
        delta = {'reference': (max_val-min_val)/2, 'increasing': {'color': "#EF553B"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [min_val, max_val/3], 'color': 'rgba(102, 126, 234, 0.3)'},
                {'range': [max_val/3, 2*max_val/3], 'color': 'rgba(118, 75, 162, 0.3)'},
                {'range': [2*max_val/3, max_val], 'color': 'rgba(239, 85, 59, 0.3)'}],
            'threshold': {
                'line': {'color': "#2c3e50", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "#2c3e50", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_feature_impact_chart(features, values, prediction):
    """Create a horizontal bar chart for feature impact"""
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=['#667eea', '#764ba2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    ))
    
    fig.update_layout(
        title="Feature Impact Analysis",
        xaxis_title="Impact on Cost (USD)",
        yaxis_title="Features",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def calculate_feature_impact(age, sex, bmi, children, smoker, region, base_prediction):
    """Calculate approximate impact of each feature"""
    # These are simplified calculations for demonstration
    age_impact = age * 250
    sex_impact = 1000 if sex == "Male" else 500
    bmi_impact = max(0, (bmi - 18.5) * 300)
    children_impact = children * 500
    smoker_impact = 15000 if smoker == "Yes" else 0
    region_impact = 1000 if region == "Southeast" else 500
    
    features = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
    impacts = [age_impact, sex_impact, bmi_impact, children_impact, smoker_impact, region_impact]
    
    return features, impacts

if submit:
    if model is None:
        st.error("❌ No model loaded. Please upload `insurance_model.pkl` in the sidebar or place it next to this app.")
    else:
        try:
            # Show loading animation
            with st.spinner("🔮 Predicting insurance cost..."):
                time.sleep(1.5)  # Simulate processing time
                
                X_input = prepare_input(age, sex, bmi, children, smoker, region)
                prediction = model.predict(X_input)
                pred_value = float(prediction[0])
            
            # Display prediction with enhanced visualization
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Create three columns for results display
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Gauge chart
                gauge_fig = create_gauge_chart(pred_value)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Main prediction box
                st.markdown('<div class="prediction-box pulse">', unsafe_allow_html=True)
                st.metric(label="Predicted Insurance Cost", value=f"${pred_value:,.2f}")
                
                # Cost category
                if pred_value < 8000:
                    st.success("💰 **Low Cost Range**")
                    st.write("This is below average for most plans")
                elif pred_value < 15000:
                    st.warning("💵 **Moderate Cost Range**")
                    st.write("This is typical for standard plans")
                else:
                    st.error("💸 **High Cost Range**")
                    st.write("Consider premium coverage options")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                # Quick stats
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Age Impact", f"+{age * 250:,.0f}")
                st.metric("BMI Impact", f"+{max(0, (bmi - 18.5) * 300):,.0f}")
                st.metric("Smoker Impact", f"+{15000 if smoker == 'Yes' else 0:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature impact visualization
            st.subheader("📈 Feature Impact Analysis")
            features, impacts = calculate_feature_impact(age, sex, bmi, children, smoker, region, pred_value)
            impact_fig = create_feature_impact_chart(features, impacts, pred_value)
            st.plotly_chart(impact_fig, use_container_width=True)
            
            # Show input features in an attractive way
            with st.expander("🔍 View Detailed Input Features"):
                feature_data = {
                    'Feature': ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'],
                    'Value': [age, sex, bmi, children, smoker, region],
                    'Encoded Value': [age, SEX_MAP[sex], bmi, children, SMOKER_MAP[smoker], REGION_MAP[region]],
                    'Impact (USD)': impacts
                }
                feature_df = pd.DataFrame(feature_data)
                st.dataframe(feature_df.style.background_gradient(subset=['Impact (USD)'], cmap='Blues'), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Example scenarios
st.markdown("---")
st.subheader("💡 Example Scenarios")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.markdown("#### 📋 Scenario A")
    st.markdown("""
    - **Age**: 30 years
    - **Sex**: Female  
    - **BMI**: 22.0
    - **Children**: 0
    - **Smoker**: No
    - **Region**: Southwest
    """)
    if st.button("Load Scenario A", key="scenario_a", use_container_width=True):
        st.info("Fill in the form with these values and click Predict!")
    st.markdown('</div>', unsafe_allow_html=True)

with scenario_col2:
    st.markdown("#### 📋 Scenario B")
    st.markdown("""
    - **Age**: 45 years
    - **Sex**: Male
    - **BMI**: 31.5
    - **Children**: 2
    - **Smoker**: Yes
    - **Region**: Southeast
    """)
    if st.button("Load Scenario B", key="scenario_b", use_container_width=True):
        st.info("Fill in the form with these values and click Predict!")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer / notes
st.markdown("---")
st.subheader("📝 Implementation Details")

notes_expander = st.expander("Click to view technical details and notes")
with notes_expander:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🔧 Technical Specifications:**
        - Model expects features: `age, sex, bmi, children, smoker, region`
        - Encoding schema:
          - Sex: Male=0, Female=1
          - Smoker: Yes=0, No=1  
          - Region: SE=0, SW=1, NE=2, NW=3
        - Built with Streamlit and scikit-learn
        """)
    
    with col2:
        st.markdown("""
        **💡 Usage Notes:**
        - For production use, ensure model includes preprocessing
        - Feature impacts are illustrative estimates
        - Actual costs may vary based on provider
        - Always consult with insurance professionals
        """)

# Add a beautiful footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #6B46C1 0%, #D53F8C 100%); 
         border-radius: 15px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
        <h2 style='margin: 0; color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
            🏥 Medical Insurance Cost Predictor ✨
        </h2>
        <p style='margin: 1rem 0; color: #FFE4E6; font-size: 1.2rem;'>
            Made with ❤️ by Vishal Kumar
        </p>
        <p style='margin: 0; color: #FDF2F8; font-style: italic;'>
            Empowering healthcare decisions through intelligent predictions
        </p>
    </div>
    """, 
    unsafe_allow_html=True

)
