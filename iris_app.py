import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 🎯 Streamlit page config
st.set_page_config(page_title="Iris Classifier Pro", layout="wide", page_icon="🌸")

# 📦 Load the trained ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# 🌈 Styling - Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.05);
        }
        h1, h2, h3 {
            color: #4A148C;
            text-align: center;
        }
        .stSlider > div > div {
            color: #4A148C;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# 🌼 Header
st.markdown("<h1>🌼 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>🧠 Predict the species of Iris flower using machine learning</h3>", unsafe_allow_html=True)
st.markdown("---")

# 📊 Sidebar for Inputs
with st.sidebar:
    st.header("🌿 Input Flower Features")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)
    st.info("Model: RandomForestClassifier 🌲")

    with st.expander("ℹ️ About Iris Features"):
    st.markdown("""
    - **Sepal Length & Width**: Outer part of the flower.
    - **Petal Length & Width**: Inner colorful part, varies most between species.
    - Based on these, the model predicts **Setosa**, **Versicolor**, or **Virginica**.
    """)

# 🔍 Prepare data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)

# 🌸 Define class labels and colors
iris_species = ['Setosa', 'Versicolor', 'Virginica']
species_colors = ['#66bb6a', '#ffa726', '#42a5f5']

# 🧠 Show Prediction
st.subheader("🎯 Predicted Species:")
st.success(f"🌸 **{iris_species[prediction]}**")

if st.button("📥 Download Prediction"):
    df = pd.DataFrame(input_data, columns=feature_names)
    df["Predicted Species"] = iris_species[prediction]
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", data=csv, file_name="iris_prediction.csv", mime='text/csv')

# 📈 Probability Chart
st.subheader("📊 Prediction Confidence")
proba_df = pd.DataFrame(probabilities, columns=iris_species).T.rename(columns={0: "Probability"})
proba_df["Color"] = species_colors

fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.barplot(x=proba_df.index, y="Probability", palette=proba_df["Color"].tolist(), data=proba_df, ax=ax1)
ax1.set_ylim(0, 1)
ax1.set_ylabel("Probability")
ax1.set_xlabel("Species")
ax1.set_title("Model Prediction Confidence")
st.pyplot(fig1)

# 📌 Feature Importance
st.subheader("📌 Feature Importance in Model")
importance = model.feature_importances_
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.barplot(x="Importance", y="Feature", palette="magma", data=imp_df, ax=ax2)
ax2.set_title("Model Feature Importance")
st.pyplot(fig2)

# 🚀 Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Built with ❤️ by <b>Rehan</b> using Streamlit & Machine Learning.</p>", unsafe_allow_html=True)

[theme]
primaryColor="#4A148C"
backgroundColor="#f5f5f5"
secondaryBackgroundColor="#ffffff"
textColor="#262730"
font="sans serif"
