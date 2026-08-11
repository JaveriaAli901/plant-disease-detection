
import streamlit as st
from PIL import Image
from predict import predict_disease

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(to right,#E8F5E9,#F1F8E9);
}

/* Main Title */
.title{
    text-align:center;
    color:#1B5E20;
    font-size:50px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#388E3C;
    font-size:20px;
    margin-bottom:30px;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    background:white;
    padding:20px;
    border-radius:15px;
    border:2px solid #81C784;
}

/* Button */
.stButton>button{
    width:100%;
    background:#2E7D32;
    color:white;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
    border:none;
    padding:12px;
}

.stButton>button:hover{
    background:#1B5E20;
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#C8E6C9;
}

/* Success Box */
.stAlert{
    border-radius:15px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🌱 Plant Health AI")

st.sidebar.success("AI Powered Disease Detection")

st.sidebar.info("""
Upload a plant leaf image.

The AI model will detect the disease and show the confidence score.

Model: MobileNetV2

Classes: 38
""")

# -------------------------------
# Title
# -------------------------------
st.markdown('<div class="title">🌿 Plant Disease Detection</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">AI Powered Leaf Disease Classifier</div>', unsafe_allow_html=True)

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

    with col2:

        st.write("")

        st.write("")

        if st.button("🔍 Predict Disease"):

            with open("temp.jpg","wb") as f:
                f.write(uploaded_file.getbuffer())

            disease, confidence = predict_disease("temp.jpg")

            st.success("Prediction Completed ✅")

            st.markdown(f"## 🌿 {disease}")

            st.progress(float(confidence)/100)

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<hr>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='footer'>Developed with ❤️ by Javeria Ali</div>",
    unsafe_allow_html=True
)