import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍎🍌🍊",
    layout="wide"
)

# ==============================================
# STYLING
# ==============================================
st.markdown("""
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1508615263227-c5fc8d0a9f0f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(240, 248, 255, 0.78);
        min-height: 100vh;
    }
    .title-big {
        font-size: 3.6rem;
        font-weight: bold;
        text-align: center;
        margin: 70px 0 20px 0;
        color: #2c3e50;
    }
    .subtitle {
        font-size: 1.45rem;
        text-align: center;
        color: #5a7d9c;
        margin-bottom: 50px;
    }
    .big-button {
        font-size: 1.4rem !important;
        height: 100px !important;
        width: 100% !important;
        border-radius: 12px !important;
        margin: 18px 0 !important;
        background-color: #a7d0f0 !important;
        color: #1a3c5e !important;
        border: none !important;
        box-shadow: 0 5px 14px rgba(0,0,0,0.14) !important;
    }
    .big-button:hover {
        background-color: #8bbde5 !important;
        transform: translateY(-2px);
    }
    .result-box {
        background: white;
        padding: 24px;
        border-radius: 14px;
        border-left: 5px solid #4a90e2;
        box-shadow: 0 6px 20px rgba(0,0,0,0.09);
        margin: 22px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================
# SESSION STATE
# ==============================================
if 'page' not in st.session_state:
    st.session_state.page = "home"

# ==============================================
# MODEL LOADING
# ==============================================
@st.cache_resource
def load_fruit_model():
    possible_paths = [
        "model_fruit.h5",
        "./model_fruit.h5",
        "models/model_fruit.h5",
        os.path.join(os.path.dirname(__file__), "model_fruit.h5")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return load_model(path), path
            except Exception as e:
                st.warning(f"Failed to load model from {path}\nError: {str(e)}")
    st.error("❌ Model file 'model_fruit.h5' not found!")
    return None, None

model, loaded_path = load_fruit_model()

# ==============================================
# CONSTANTS
# ==============================================
IMG_SIZE = (150, 150)

CLASS_NAMES = [
    'RottenBanana_3', 'GreenApple_1', 'GreenOrange_1', 'FreshOrange_2',
    'RedApple_2', 'RottenApple_3', 'GreenBanana_2', 'RottenOrange_3', 'YellowBanana_1'
]

# ==============================================
# FRUIT INFO MAPPING
# ==============================================
def get_fruit_info(label):
    if not label:
        return "Unknown", 0, "Cannot classify"
        
    if 'Banana' in label:
        if 'Yellow' in label:          return "Banana", 2, "Ripe"
        if 'Green' in label:      return "Banana", 3, "Unripe"
        if 'Rotten' in label:          return "Banana", 3, "Rotten"
        return "Banana", 1, "Unripe"
    
    if 'Apple' in label:
        if 'Red' in label:             return "Apple", 2, "Ripe"
        if 'Rotten' in label:          return "Apple", 3, "Rotten"
        return "Apple", 1, "Unripe"
    
    if 'Orange' in label:
        if 'OrangeOrange' in label:    return "Orange", 2, "Ripe"
        if 'Rotten' in label:          return "Orange", 3, "Rotten"
        return "Orange", 1, "Unripe"
    
    return "Unknown", 0, "Cannot classify"

# ==============================================
# GET REFERENCE IMAGE
# ==============================================
def get_first_image(folder_name):
    base = "reference_images/fruits"
    folder_path = os.path.join(base, folder_name)
    if not os.path.exists(folder_path):
        return None
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder_path, file)
    return None

# ==============================================
# HOME PAGE
# ==============================================
if st.session_state.page == "home":
    st.markdown('<div class="title-big">🍎🍌🍊 Fruit Ripeness Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Ripeness Detection for Fruits</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("🍏 DETECTION FRUIT", key="btn_simple", use_container_width=True):
            st.session_state.page = "classify"
            st.rerun()
            
        if st.button("📊PREDICT RIPENESS", key="btn_full", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.caption("Artificial Intelligence Project • Semester 5 • 2025/2026")

# ==============================================
# SIMPLE CLASSIFY PAGE → Only Reference Images
# ==============================================
elif st.session_state.page == "classify":
    st.title("Detection Fruit - Ripeness")
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    uploaded_file = st.file_uploader("Upload fruit image (jpg/png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=480)
        
        if model is None:
            st.error("Model is not loaded.")
        else:
            with st.spinner("Analyzing..."):
                img_resized = img.convert('RGB').resize(IMG_SIZE)
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred_probs = model.predict(img_array)[0]
                top_idx = np.argmax(pred_probs)
                predicted_class = CLASS_NAMES[top_idx]
                fruit, grade, desc = get_fruit_info(predicted_class)
            
            st.subheader("📸 Ripeness Grade")
            
            if fruit == "Banana":
                cols = st.columns(3)
                with cols[0]: 
                    p = get_first_image("GreenBanana_2")
                    if p: st.image(p, caption="Grade 1 – Unripe")
                with cols[1]: 
                    p = get_first_image("YellowBanana_1")
                    if p: st.image(p, caption="Grade 2 – Ripe")
                with cols[2]: 
                    p = get_first_image("RottenBanana_3")
                    if p: st.image(p, caption="Grade 3 – Rotten")
            
            elif fruit == "Apple":
                cols = st.columns(3)
                with cols[0]: 
                    p = get_first_image("GreenApple_1")
                    if p: st.image(p, caption="Grade 1 – Unripe")
                with cols[1]: 
                    p = get_first_image("RedApple_2")
                    if p: st.image(p, caption="Grade 2 – Ripe")
                with cols[2]: 
                    p = get_first_image("RottenApple_3")
                    if p: st.image(p, caption="Grade 3 – Rotten")
            
            elif fruit == "Orange":
                cols = st.columns(3)
                with cols[0]: 
                    p = get_first_image("GreenOrange_1")
                    if p: st.image(p, caption="Grade 1 – Unripe")
                with cols[1]: 
                    p = get_first_image("FreshOrange_2")  # or change to correct folder name
                    if p: st.image(p, caption="Grade 2 – Ripe")
                with cols[2]: 
                    p = get_first_image("RottenOrange_3")
                    if p: st.image(p, caption="Grade 3 – Rotten")

# ==============================================
# FULL PREDICTION PAGE → Only Result + Debug Probabilities
# ==============================================
elif st.session_state.page == "predict":
    st.title("Predict Ripeness - Detailed Mode")
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    uploaded_file = st.file_uploader("Upload fruit image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=480)
        
        if model is None:
            st.error("Model is not loaded.")
        else:
            with st.spinner("Analyzing..."):
                img_resized = img.convert('RGB').resize(IMG_SIZE)
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                pred_probs = model.predict(img_array)[0]
                top_idx = np.argmax(pred_probs)
                predicted_class = CLASS_NAMES[top_idx]
                confidence = float(pred_probs[top_idx])
                
                fruit, grade, desc = get_fruit_info(predicted_class)
            
            st.markdown("---")
            st.markdown(f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <b>Fruit:</b> {fruit}<br>
                <b>Grade:</b> {grade}<br>
                <b>Condition:</b> {desc}<br>
                <b>Confidence:</b> {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Show detailed probabilities (debug)"):
                for name, prob in zip(CLASS_NAMES, pred_probs):
                    st.write(f"{name:25} : {prob:.6f}")

# Footer
st.markdown("---")
st.caption("Artificial Intelligence Project • Semester 5 • 2025/2026")
