import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Function to load model and preprocessing info
@st.cache_resource
def load_model_and_preprocessing():
    model = joblib.load('final_xgb_model.pkl')
    preprocessing_info = joblib.load('preprocessing_info.pkl')
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    return model, preprocessing_info, feature_names

def preprocess_input(input_data, preprocessing_info, feature_names):
    # Create a DataFrame from input data
    df = pd.DataFrame([input_data])
    
    # Get feature sets from preprocessing_info
    static_cat_features = preprocessing_info['static_cat_features']
    static_num_features = preprocessing_info['static_num_features']
    time_varying_features = preprocessing_info['time_varying_features']
    
    # Standardize categorical inputs to match training data
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.capitalize()  # Ensure 'Male' and 'Female'
    
    if 'race' in df.columns:
        df['race'] = df['race'].str.upper()  # Ensure 'C', 'AA', 'OTHERS'
        df['race'] = np.where(df['race'].isin(['C', 'AA']), df['race'], 'OTHERS')
    
    # Print the prepared categorical data for debugging
    print("Categorical data after preparation:")
    for col in static_cat_features:
        if col in df.columns:
            print(f"{col}: {df[col].values}")
    
    # Instead of using the encoder, directly create the one-hot encoded features
    # based on the feature_names.json file
    
    # Detect all one-hot encoded columns in feature_names
    encoded_cols = {}
    for feature in static_cat_features:
        encoded_cols[feature] = [f for f in feature_names if f.startswith(f"{feature}_")]
    
    # Manually create one-hot encoded features
    for feature, columns in encoded_cols.items():
        if feature in df.columns:
            feature_value = df[feature].iloc[0]
            
            # Create the one-hot encoded columns
            for col in columns:
                # Extract the category value from the column name
                category_value = col[len(feature)+1:]
                
                # Check if this is the right category
                if feature == 'CYP3A5_3_count' and category_value == '1.0' and feature_value == '1.0':
                    df[col] = 1
                elif feature == 'CYP3A5_3_count' and category_value == '2.0' and feature_value == '2.0':
                    df[col] = 1
                elif feature == 'CYP3A5_3_count' and category_value == 'nan' and (feature_value == 'nan' or feature_value == 'Unknown'):
                    df[col] = 1
                elif feature == 'CYP3A4_22_count' and category_value == '1.0' and feature_value == '1.0':
                    df[col] = 1
                elif feature == 'CYP3A4_22_count' and category_value == '2.0' and feature_value == '2.0':
                    df[col] = 1
                elif feature == 'CYP3A4_22_count' and category_value == 'nan' and (feature_value == 'nan' or feature_value == 'Unknown'):
                    df[col] = 1
                elif feature == 'ABCB1_rs1045642_count' and category_value == '1.0' and feature_value == '1.0':
                    df[col] = 1
                elif feature == 'ABCB1_rs1045642_count' and category_value == '2.0' and feature_value == '2.0':
                    df[col] = 1
                elif feature == 'ABCB1_rs1045642_count' and category_value == 'nan' and feature_value == 'nan':
                    df[col] = 1
                elif feature == 'sex' and category_value == 'M' and feature_value == 'M':
                    df[col] = 1
                elif feature == 'race' and category_value == 'C' and feature_value == 'C':
                    df[col] = 1
                elif feature == 'race' and category_value == 'others' and feature_value == 'OTHERS':
                    df[col] = 1
                elif feature == 'Tobacco_Use' and category_value == 'Y' and feature_value == 'Y':
                    df[col] = 1
                elif feature == 'Alcohol_Use' and category_value == 'Y' and feature_value == 'Y':
                    df[col] = 1
                else:
                    df[col] = 0
    
    # Scale numerical features
    static_scaler = preprocessing_info['scalers']['static']
    dynamic_scaler = preprocessing_info['scalers']['dynamic']
    
    if static_num_features:
        df[static_num_features] = static_scaler.transform(df[static_num_features])
    
    time_varying_available = [col for col in time_varying_features if col in df.columns]
    if time_varying_available:
        df[time_varying_available] = dynamic_scaler.transform(df[time_varying_available])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the features used in the model
    X = df[feature_names].values
    return X

# Streamlit app
st.title("Tacrolimus Dose Predictor")

st.markdown("""
This application predicts the tacrolimus trough to dose ratio and the required dose to achieve a target trough based on patient features.
Enter the patient details below to get the predicted ratio and recommended dose. This tool is intended for research purposes and should not replace clinical judgment.
""")

# Load model and preprocessing info
try:
    model, preprocessing_info, feature_names = load_model_and_preprocessing()
except FileNotFoundError:
    st.error("Model or preprocessing files not found. Please ensure 'final_xgb_model.pkl', 'preprocessing_info.pkl', and 'feature_names.json' are available.")
    st.stop()

# Input form
st.header("Patient Information")

st.markdown("### Target Trough Concentration (ng/mL)")
target_trough = st.number_input("", min_value=0.0, max_value=30.0, value=10.0)

#target_trough = st.number_input("Target Trough Concentration (ng/mL)", min_value=0.0, max_value=30.0, value=10.0)

with st.form("patient_form"):
    st.markdown("### Required Fields")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)*", min_value=0, max_value=120)
        
        race_options = [
            {"display": "European", "value": "C"},
            {"display": "African American", "value": "AA"},
            {"display": "Others", "value": "Others"}
        ]
        race_selected = st.selectbox(
            "Race*", 
            options=race_options,
            format_func=lambda x: x["display"]
        )
        hct = st.number_input("Hematocrit (hct, %)*", min_value=0.0)
    
    with col2:
        prev_dose = st.number_input("Previous Daily Dose (mg)*", min_value=0.0)
        prev_c_whole = st.number_input("Previous Trough Concentration (ng/ml)*", min_value=0.0)
        hours_after_transplant = st.number_input("Time Post-transplant (hour)*", min_value=0.0, max_value=168.0)
        
    st.markdown("### Optional Fields")
    col4, col5 = st.columns(2)
    
    with col4:
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0)
        sex = st.selectbox("Sex", options=['M', 'F'])
        
        # Tobacco use options
        tobacco_options = [            
            {"display": "No", "value": "N"},
            {"display": "Yes", "value": "Y"}
        ]
        tobacco_selected = st.selectbox(
            "Tobacco Use History", 
            options=tobacco_options,
            format_func=lambda x: x["display"]
        )
    
    with col5:
        # CYP3A5*3 Genotype options
        cyp3a5_options = [
            {"display": "Unknown", "value": "Unknown"},
            {"display": "*1/*1", "value": "0"},
            {"display": "*1/*3", "value": "1"},
            {"display": "*3/*3", "value": "2"}
        ]
        cyp3a5_selected = st.selectbox(
            "CYP3A5*3 Genotype", 
            options=cyp3a5_options,
            format_func=lambda x: x["display"]
        )
        
        # CYP3A4*22 Genotype options
        cyp3a4_options = [
            {"display": "Unknown", "value": "Unknown"},
            {"display": "*1/*1", "value": "0"},
            {"display": "*1/*22", "value": "1"},
            {"display": "*22/*22", "value": "2"}
        ]
        cyp3a4_selected = st.selectbox(
            "CYP3A4*22 Genotype", 
            options=cyp3a4_options,
            format_func=lambda x: x["display"]
        )
        
        # Alcohol use options
        alcohol_options = [            
            {"display": "No", "value": "N"},
            {"display": "Yes", "value": "Y"}
        ]
        alcohol_selected = st.selectbox(
            "Alcohol Use History", 
            options=alcohol_options,
            format_func=lambda x: x["display"]
        )
    
    col6, col7 = st.columns(2)
    
    with col6:
        alb = st.number_input("Albumin (alb, g/dL)", min_value=0.0, value=4.0)
        alt = st.number_input("ALT (U/L)", min_value=0.0, value=20.0)
        ast = st.number_input("AST (U/L)", min_value=0.0, value=20.0)
    
    with col7:
        creat = st.number_input("Creatinine (mg/dL)", min_value=0.0, value=1.0)
        cyp3a_inhibitor = st.checkbox("CYP3A Inhibitor Present")
        cyp3a_inducer = st.checkbox("CYP3A Inducer Present")
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Validate required fields
    required_fields_empty = False
    
    if age <= 0:
        st.error("Please enter a valid age (required).")
        required_fields_empty = True
    
    if prev_dose <= 0:
        st.error("Please enter a valid previous daily dose (required).")
        required_fields_empty = True
    
    if prev_c_whole <= 0:
        st.error("Please enter a valid previous trough concentration (required).")
        required_fields_empty = True
        
    if hours_after_transplant <= 0:
        st.error("Please enter valid hours after transplant (required).")
        required_fields_empty = True
        
    if hct <= 0:
        st.error("Please enter a valid hematocrit value (required).")
        required_fields_empty = True
    
    # If required fields are missing, don't proceed with prediction
    if required_fields_empty:
        st.stop()
    
    # Handle CYP3A5 value
    if cyp3a5_selected["value"] == "Unknown":
        cyp3a5_value = "Unknown"
    else:
        cyp3a5_value = f"{cyp3a5_selected['value']}.0"
    
    # Handle CYP3A4 value
    if cyp3a4_selected["value"] == "Unknown":
        cyp3a4_value = "Unknown"
    else:
        cyp3a4_value = f"{cyp3a4_selected['value']}.0"
    
    input_data = {
        'age': age,
        'WeightKG': weight,
        'HeightCM': height,
        'sex': sex,
        'race': race_selected["value"],
        'prev_Dose': prev_dose,
        'prev_C_whole': prev_c_whole,
        'hours_after_transplant': hours_after_transplant,
        'CYP3A5_3_count': cyp3a5_value,
        'CYP3A4_22_count': cyp3a4_value,
        'hct': hct,
        'alb': alb,
        'alt': alt,
        'ast': ast,
        'creat': creat,
        'CYP3A_inhibitor': 1 if cyp3a_inhibitor else 0,
        'CYP3A_inducer': 1 if cyp3a_inducer else 0,
        'Tobacco_Use': tobacco_selected["value"],
        'Alcohol_Use': alcohol_selected["value"],
        # Add previous trough_dose_ratio (calculated from prev_C_whole / prev_Dose if dose > 0)
        'prev_trough_dose_ratio': prev_c_whole / prev_dose if prev_dose > 0 else 0,
        # Include other required features with default values
        'ABCB1_rs1045642_count': "0.0",
        'hours_from_first_dose': hours_after_transplant,  # Approximate as same for simplicity
        'hours_from_last_dose': 12.0
    }
    
    # Print input data for debugging
    print("Input data before preprocessing:")
    print(input_data)
    
    # Preprocess input
    X_processed = preprocess_input(input_data, preprocessing_info, feature_names)
    
    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(X_processed)
    
    # Predict
    pred_scaled = model.predict(dmatrix)[0]
    
    # Inverse transform prediction
    target_scaler = preprocessing_info['scalers']['target']
    pred = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    
    # Display results
    st.header("Prediction Results")
    
    # Calculate recommended dose for target trough
    recommended_dose = target_trough / pred if pred > 0 else float('inf')
    
    # Display with red and bold formatting
    st.markdown(f"""
    <div style='color: red; font-weight: bold;'>
    Recommended Tacrolimus Daily Dose for Target Trough {target_trough:.1f} ng/mL: {recommended_dose:.2f} mg
    </div>
    <div style='color: red; font-weight: bold;'>
    Predicted Tacrolimus Trough/Dose Ratio: {pred:.2f} (ng/ml per mg)
    </div>
    """, unsafe_allow_html=True)
    
    # Display input summary
    with st.expander("View Input Summary"):
        st.write(pd.DataFrame([input_data]).T.rename(columns={0: 'Value'}))

# Add some information about the model
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses an XGBoost model trained on data from 1624 patients to predict tacrolimus trough to dose ratios within the first week after kidney transplant.
The model considers patient demographics, previous dosing history, and genetic factors (e.g., CYP3A5 and CYP3A4 genotypes).
The predicted ratio is used to determine the appropriate dose to achieve a target tacrolimus trough concentration.
""")
