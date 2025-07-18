import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Iris Classifier Pro", layout="wide", page_icon="🌼")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Styling
st.markdown("""
    <style>
        .main {
            background-color: #fdfdfd;
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem 3rem 2rem 3rem;
        }
        h1 {
            color: #4a148c;
            font-size: 3rem;
            text-align: center;
        }
        h3 {
            color: #1b1b1b;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🌼 Iris Species Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>A professional web application to classify Iris flowers using Machine Learning.</p>", unsafe_allow_html=True)

# Sidebar input form
with st.sidebar:
    st.header("🌿 Input Flower Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    st.markdown("---")
    st.markdown("📊 Powered by RandomForestClassifier")

# Predict
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)

iris_species = ['Setosa', 'Versicolor', 'Virginica']
species_colors = ['#66bb6a', '#ffa726', '#42a5f5']

# Prediction Output
st.markdown("### 🧠 Prediction Result")
st.success(f"🌼 **Predicted Species**: {iris_species[prediction]}")
st.markdown("---")

# Show probability bar chart
st.markdown("### 📊 Prediction Probabilities")
proba_df = pd.DataFrame(proba, columns=iris_species).T
proba_df.columns = ["Probability"]

fig1, ax1 = plt.subplots(figsize=(6, 2.5))
sns.barplot(x=proba_df.index, y="Probability", data=proba_df, palette=species_colors, ax=ax1)
ax1.set_ylim(0, 1)
ax1.set_title("Model Confidence")
st.pyplot(fig1)

# Show feature importance
st.markdown("### 🔍 Feature Importance")
importance = model.feature_importances_
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(6, 2.5))
sns.barplot(x="Importance", y="Feature", data=imp_df, palette="magma", ax=ax2)
ax2.set_title("Feature Contributions to Model")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Built with <b>Streamlit</b> | Designed for clarity, speed, and simplicity.</p>", unsafe_allow_html=True)
