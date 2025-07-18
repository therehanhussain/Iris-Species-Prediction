import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Iris Classifier Pro",
    layout="wide",
    page_icon="🌸"
)

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            color: #333;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #6a1b9a;
            font-size: 2.8rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .stSlider > div {
            color: #444;
        }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict the species of Iris flower based on its features using a trained Random Forest model.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar input
with st.sidebar:
    st.header("🌿 Input Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    st.markdown("---")
    st.markdown("🤖 **Model**: Random Forest Classifier")

# Explanation section
with st.expander("ℹ️ About Iris Features"):
    st.markdown("""
    - **Sepal Length & Width**: Outer protective parts of the flower.
    - **Petal Length & Width**: Inner colorful parts, vary most among species.
    - This model classifies into:
        - **Setosa**
        - **Versicolor**
        - **Virginica**
    """)

# Make prediction
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_features)[0]
probabilities = model.predict_proba(input_features)[0]

species = ['Setosa', 'Versicolor', 'Virginica']
colors = ['#66bb6a', '#ffa726', '#42a5f5']
predicted_species = species[prediction]

# Show prediction
st.subheader("🔍 Prediction Result")
st.success(f"🌼 **Predicted Species**: {predicted_species}")
st.markdown("---")

# Show prediction probabilities
st.subheader("📊 Prediction Probabilities")
proba_df = pd.DataFrame({
    "Species": species,
    "Probability": probabilities
})
fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.barplot(x="Species", y="Probability", data=proba_df, palette=colors, ax=ax1)
ax1.set_ylim(0, 1)
ax1.set_title("Prediction Confidence")
st.pyplot(fig1)

# Feature importance
st.subheader("📌 Feature Importance")
importance = model.feature_importances_
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.barplot(x="Importance", y="Feature", data=imp_df.sort_values("Importance"), palette="magma", ax=ax2)
ax2.set_title("Model Feature Contribution")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>🌼 Built with ❤️ using Streamlit | © 2025 IrisApp AI</p>", unsafe_allow_html=True)
