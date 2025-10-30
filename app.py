import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(page_title="Bank Deposit Prediction", page_icon="💰", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {background-color:#f8f9fa;font-family:"Segoe UI",sans-serif;color:#333;}
.navbar {overflow:hidden;background-color:#0d6efd;padding:12px 16px;border-radius:8px;margin-bottom:25px;}
.navbar a {float:left;color:#f2f2f2;text-align:center;padding:10px 16px;text-decoration:none;
            font-size:18px;border-radius:6px;}
.navbar a:hover {background-color:#0056b3;color:white;}
.navbar a.active {background-color:#0b5ed7;color:white;}
.main {padding:20px 25px;background-color:white;border-radius:12px;
       box-shadow:0 0 10px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ---------- Navbar ----------
st.markdown("""
<div class="navbar">
  <a href="#home" class="active">🏠 Home</a>
  <a href="#about">ℹ️ About</a>
  <a href="#contact">📧 Contact</a>
</div>
""", unsafe_allow_html=True)

st.header("💰 Bank Deposit Prediction App")

# ---------- Load Model ----------
try:
    with open("bank deposit.pkl", "rb") as f:
    model = pickle.load(f)
from tensorflow import keras

try:
    model = keras.models.load_model("bank_deposit_model.keras")
except Exception as e:
    st.error(f"❌ Error loading Keras model: {e}")
    st.stop()

except ModuleNotFoundError as e:
    st.error(f"⚠️ Missing library or class used in model: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------- Input Section ----------
st.write("Enter basic customer details for prediction:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=1500)
with col2:
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=300)
    campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=3)

predict_btn = st.button("🔮 Predict")

# ---------- Prediction ----------
if predict_btn:
    try:
        X = np.array([[age, balance, duration, campaign]], dtype=float)

        # Pad to 41 features if needed
        expected_features = 41
        if X.shape[1] < expected_features:
            X = np.pad(X, ((0, 0), (0, expected_features - X.shape[1])), mode='constant')

        pred = model.predict(X)

        # Handle binary vs multi-class
        if pred.shape[-1] == 1:
            prob = float(pred[0][0])
            result = "✅ Will Subscribe" if prob > 0.5 else "❌ Will Not Subscribe"
            st.success(f"**Prediction:** {result}")
            st.info(f"Confidence: {prob:.2f}")
        else:
            class_idx = int(np.argmax(pred, axis=1)[0])
            st.success(f"**Predicted Class:** {class_idx}")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

# ---------- About ----------
st.markdown("---")
st.subheader("ℹ️ About")
st.write("""
This demo app uses a pre-trained **Keras (TensorFlow)** model to predict whether a customer will subscribe
to a term deposit.  
⚠️ Currently, only four inputs are collected and the rest are filled with zeros,
so predictions are not guaranteed to reflect real results.
""")

# ---------- Contact ----------
st.markdown("---")
st.subheader("📧 Contact")
st.write("""
**Developer:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [github.com/yourusername](https://github.com/yourusername)
""")
