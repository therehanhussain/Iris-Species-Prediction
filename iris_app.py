import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
st.set_page_config(page_title="🌸 Iris Classifier Pro", layout="wide", page_icon="🌼")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        h1 {
            color: #6a1b9a;
            font-size: 3rem;
            text-align: center;
        }
        .footer {
            text-align: center;
            font-size: 15px;
            color: gray;
            padding-top: 1rem;
        }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌼 Iris Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter flower measurements to predict the Iris species using Machine Learning 🌿</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout: Input - Prediction - Charts
col1, col2 = st.columns([1, 2])

# Sidebar-like Input Section
with col1:
    st.header("🌿 Input Parameters")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)

    iris_species = ['Setosa', 'Versicolor', 'Virginica']
    species_colors = ['#66bb6a', '#ffa726', '#42a5f5']

    st.markdown("### 🧠 Predicted Species")
    st.success(f"🌸 **{iris_species[prediction]}**")
    st.markdown("#### 🔎 Confidence Levels")

    proba_df = pd.DataFrame(proba, columns=iris_species).T
    proba_df.columns = ["Probability"]

    fig1, ax1 = plt.subplots(figsize=(5, 2.5))
    sns.barplot(x=proba_df.index, y="Probability", data=proba_df, palette=species_colors, ax=ax1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Prediction Confidence")
    st.pyplot(fig1)

# Visualization Section
with col2:
    st.header("📊 Feature Importance")
    importance = model.feature_importances_
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma", ax=ax2)
    ax2.set_title("Which Features Drive the Prediction?")
    st.pyplot(fig2)

    st.markdown("### 📄 Data Snapshot")
    iris_df = pd.DataFrame([
        {"Feature": "Sepal Length", "Value": sepal_length},
        {"Feature": "Sepal Width", "Value": sepal_width},
        {"Feature": "Petal Length", "Value": petal_length},
        {"Feature": "Petal Width", "Value": petal_width},
    ])
    st.table(iris_df)

# Footer
st.markdown("---")
st.markdown('<div class="footer">🚀 Made with ❤️ using Streamlit | Created by MD Rehan Hussain</div>', unsafe_allow_html=True)
