import streamlit as st
import pickle
import base64


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Bank Deposit Prediction", page_icon="💰", layout="wide")

# -------------------- LOAD MODEL --------------------
import pickle
import importlib

try:
    with open("bank deposit.pkl", "rb") as f:
        model = pickle.load(f)
except ModuleNotFoundError as e:
    st.error(f"⚠️ Missing library or class used in model: {e}")
    st.info("Add the required package to requirements.txt or retrain the model using only standard sklearn objects.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()



from joblib import dump
dump(model, "bank_deposit.joblib")



# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    /* Background and text styling */
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
    }
    /* Navbar */
    .navbar {
        overflow: hidden;
        background-color: #0d6efd;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .navbar a {
        float: left;
        color: #f2f2f2;
        text-align: center;
        padding: 10px 16px;
        text-decoration: none;
        font-size: 18px;
        border-radius: 6px;
    }
    .navbar a:hover {
        background-color: #0056b3;
        color: white;
    }
    .navbar a.active {
        background-color: #0b5ed7;
        color: white;
    }
    .main {
        padding: 10px 25px;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- NAVBAR --------------------
st.markdown("""
<div class="navbar">
  <a href="#home" class="active">🏠 Home</a>
  <a href="#about">ℹ️ About</a>
  <a href="#contact">📧 Contact</a>
</div>
""", unsafe_allow_html=True)

# -------------------- HOME SECTION --------------------
st.header("💰 Bank Deposit Prediction App")
st.write("This app predicts whether a customer will subscribe to a term deposit based on input features.")

# Example Input Fields (customize as per your dataset)
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=500)
duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=300)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=2)

# Prediction button
if st.button("🔮 Predict"):
    # Prepare data (example — adjust shape/columns to match your model)
    X = [[age, balance, duration, campaign]]
    pred = model.predict(X)
    result = "✅ Will Subscribe" if pred[0] == 1 else "❌ Will Not Subscribe"
    
    st.success(f"**Prediction:** {result}")

# -------------------- ABOUT SECTION --------------------
st.markdown("---")
st.subheader("ℹ️ About")
st.write("""
This project uses a trained machine learning model to predict whether a bank customer will subscribe
to a term deposit.  
Developed using **Python**, **Scikit-learn**, and **Streamlit** for interactive deployment.
""")

# -------------------- CONTACT SECTION --------------------
st.markdown("---")
st.subheader("📧 Contact")
st.write("""
For any queries or collaborations:  
- **Developer:** Your Name  
- **Email:** your.email@example.com  
- **GitHub:** [github.com/yourusername](https://github.com/yourusername)
""")
