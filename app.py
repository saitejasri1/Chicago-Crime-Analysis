import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shap
# Load mappings from JSON files
with open("consolidated_type_mapping.json", "r") as f:
    crime_type_mapping = json.load(f)

with open("severity_mapping.json", "r") as f:
    severity_mapping = json.load(f)

# Reverse the mapping (values to keys)
severity_mapping_reverse = {v: k for k, v in severity_mapping.items()}


# Reverse mapping for crime type
crime_type_reverse_mapping = {int(v): k for k, v in crime_type_mapping.items()}

# Reverse mapping for severity
# Already reversed as `severity_mapping_reverse` above

# Load models
crime_type_model = joblib.load("Encoded_Crime_Type_lgbm_model.pkl")
crime_likelihood_model = joblib.load("Crime_Likelihood_lgbm_model.pkl")
crime_severity_model = joblib.load("Encoded_Crime_Severity_lgbm_model.pkl")

# Page title and theme
st.set_page_config(page_title="Crime Insights App", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: "Arial";
    }
    .main {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Crime Insights and Prediction App")

# Input Section
st.header("📝 Enter Details for Prediction")
zip_code = st.text_input("ZIP Code", placeholder="Enter ZIP Code")
longitude = st.number_input("Longitude", value=-87.6298)
latitude = st.number_input("Latitude", value=41.8781)
hour = st.slider("Hour of the Day", 0, 23, 12)
is_weekend = st.selectbox("Is it Weekend?", ["No", "Yes"], index=0)
is_weekend = 1 if is_weekend == "Yes" else 0

if st.button("🔮 Predict Crime"):
    # Input Data
    input_data = {
        "FBI Code": 18,  # Example static value
        "Arrest": 0,  # Example static value
        "Longitude": longitude,
        "Latitude": latitude,
        "Hour": hour,
        "ZIP": int(zip_code) if zip_code.isdigit() else 0,
        "Is_Weekend": is_weekend,
    }
    input_df = pd.DataFrame([input_data])

    # Predictions
    crime_type_pred = int(np.argmax(crime_type_model.predict(input_df), axis=1)[0])
    crime_likelihood_pred = int(np.argmax(crime_likelihood_model.predict(input_df), axis=1)[0])
    crime_severity_pred = int(np.argmax(crime_severity_model.predict(input_df), axis=1)[0])

    # Decode Predictions
    crime_type_label = crime_type_reverse_mapping.get(crime_type_pred, "Unknown")
    crime_likelihood_label = "High" if crime_likelihood_pred == 1 else "Low"
    crime_severity_label = severity_mapping_reverse.get(crime_severity_pred, "Unknown")

    # Display Predictions
    st.subheader("🔮 Predictions")
    st.write(f"**Crime Type:** {crime_type_label}")
    st.write(f"**Crime Likelihood:** {crime_likelihood_label}")
    st.write(f"**Crime Severity:** {crime_severity_label}")
    # # SHAP Explainability
    # st.subheader("📈 Model Explainability")
    # st.write("Input Data Shape:", input_df.shape)
    # st.write("Expected Input Shape (Model):", crime_type_model.n_features_)
    # # Display the expected input features from the model
    # # Expected features (manually or loaded with the model)
    # expected_features = ['FBI Code', 'Arrest', 'Longitude', 'Latitude', 'Hour', 'ZIP', 'Is_Weekend']
    # st.write("Expected Input Features (Model):", expected_features)

    # # Check for missing features
    # # Validate the input DataFrame
    # missing_features = set(expected_features) - set(input_df.columns)
    # if missing_features:
    #     st.error(f"Missing Features: {missing_features}")
    # if len(expected_features) != input_df.shape[1]:
    #     st.error(f"Input shape mismatch! Expected {len(expected_features)} features but got {input_df.shape[1]}.")


    # # Verify input shape
    # if input_df.shape[1] != len(expected_features):
    #     st.error(f"Input data shape mismatch! Expected {len(expected_features)} features but got {input_df.shape[1]}.")

    # # SHAP Explainability
    # explainer = shap.TreeExplainer(crime_type_model)
    # shap_values = explainer.shap_values(input_df)

    # # SHAP Waterfall Plot for Single Input
    # if len(shap_values) > 0 and shap_values[0].shape[1] == input_df.shape[1]:
    #     st.write("**SHAP Waterfall Plot (Single Input):**")
    #     shap.waterfall_plot(
    #         shap.Explanation(
    #             values=shap_values[crime_type_pred][0],
    #             base_values=explainer.expected_value[crime_type_pred],
    #             data=input_df.iloc[0]
    #         )
    #     )
    # else:
    #     st.error("Mismatch between SHAP values and input data dimensions.")


    # # SHAP Summary Plot
    # st.write("**SHAP Summary Plot:**")
    # shap.summary_plot(shap_values, input_df, show=False)
    # st.pyplot(bbox_inches="tight", dpi=100, pad_inches=0.1)
    # Get feature importance
    feature_importance = crime_type_model.feature_importance()
    feature_names = ['FBI Code', 'Arrest', 'Longitude', 'Latitude', 'Hour', 'ZIP', 'Is_Weekend']

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    # Display feature importance as a bar chart
    st.subheader("🔑 Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="cool", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Safety Tips
    st.subheader("🛡️ Safety Tips")
    safety_tips = {
        "High": "Stay indoors during late hours. Avoid poorly lit areas.",
        "Moderate": "Travel in groups. Keep emergency numbers handy.",
        "Low": "Be cautious of surroundings. Report suspicious activities.",
    }
    st.write(safety_tips.get(crime_severity_label, "Stay alert and safe!"))
# Visualization Section
st.header("📊 Previous Crime Insights")

# Example Crime Data Visualization
crime_data = pd.DataFrame({
    "Crime Type": ["Theft", "Assault", "Burglary", "Robbery"],
    "Count": [150, 90, 60, 40],
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=crime_data, x="Crime Type", y="Count", hue="Crime Type", palette="cool", ax=ax)
ax.set_title("Crime Distribution in Area")
st.pyplot(fig)

st.markdown("---")
st.write("Thank you for using the Crime Insights App! Stay safe! 🔒")
