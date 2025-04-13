import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from src.symptom_prediction import predict_anemia_risk

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.feature_config import NORMAL_RANGES, FEATURES
from src.utils.preprocessing import process_user_input
from src.visualization import (
    create_anemia_distribution_pie, create_gender_distribution_plots,
    create_age_distribution_plots, create_hematological_parameter_plots,
    create_correlation_heatmap, create_prediction_gauge,
    create_feature_comparison_radar
)

@st.cache_resource
def load_model(model_path="models/anemia_prediction_model.pkl"):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model is trained and saved properly.")
        return None

@st.cache_data
def load_data(data_path="data/Anemia Dataset.xlsx"):
    try:
        df = pd.read_excel(data_path)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return None

@st.cache_resource
def load_symptom_model(model_path="models/symptom_anemia_prediction_model.pkl"):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Symptom prediction model file not found. Please ensure the model is trained and saved properly.")
        return None

def make_prediction(user_input, model):
    try:
        input_df = process_user_input(user_input)
        
        required_columns = set(['Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'Gender_Encoded'])
        missing_columns = required_columns - set(input_df.columns)
        
        if missing_columns:
            return None, None, f"Missing required columns: {', '.join(missing_columns)}"
        
        for feature, value in input_df.iloc[0].items():
            if feature in FEATURES and feature != 'Gender_Encoded':
                min_val = FEATURES[feature].get('min_value')
                max_val = FEATURES[feature].get('max_value')
                
                if min_val is not None and value < min_val:
                    st.warning(f"{feature} value ({value}) is below typical minimum ({min_val})")
                
                if max_val is not None and value > max_val:
                    st.warning(f"{feature} value ({value}) is above typical maximum ({max_val})")
        
        prediction = model.predict(input_df)[0]
        
        probability = None
        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_df)[0][1]
            elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
                if hasattr(model.named_steps['model'], 'predict_proba'):
                    probability = model.named_steps['model'].predict_proba(input_df)[0][1]
        except Exception as e:
            return prediction, None, f"Could not calculate probability: {str(e)}"
        
        return prediction, probability, None
        
    except KeyError as e:
        return None, None, f"Column error: {str(e)}"
    except ValueError as e:
        return None, None, f"Value error: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"

def display_prediction_result(prediction, probability):
    if prediction is None:
        return
    
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        if prediction == 1:
            st.error("Diagnosis: **Anemic**")
        else:
            st.success("Diagnosis: **Non-Anemic**")
        
        if probability is not None:
            st.write(f"Confidence: **{probability:.2%}**")
    
    with result_col2:
        if probability is not None:
            fig = create_prediction_gauge(probability, prediction)
            st.plotly_chart(fig)

def reset_input_values():
    reset_keys = [
        'age', 'hb', 'rbc', 'pcv', 'mcv', 'mch', 'mchc', 'gender'
    ]
    
    for key in reset_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state['gender'] = 'Select gender'

def validate_input(user_input):
    missing_fields = []
    
    required_fields = ['Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    for field in required_fields:
        if field not in user_input or user_input[field] is None or user_input[field] == 0:
            missing_fields.append(field)
    
    if user_input.get('Gender_Encoded') is None:
        missing_fields.append('Gender')
    
    if missing_fields:
        error_message = "Please fill in all input fields. Missing or invalid values in: " + ", ".join(missing_fields)
        return False, error_message
    
    return True, ""

def display_prediction_interface():
    st.subheader("Anemia Prediction")
    st.write("Enter patient's hematological parameters to predict anemia status.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 Generate Random Test Data", help="Fill the form with random sample data for testing"):
            import random
            
            st.session_state.gender = random.choice(["Female", "Male"])
            st.session_state.age = random.randint(18, 80)
            
            if st.session_state.gender == "Female":
                st.session_state.hb = round(random.uniform(9.0, 15.0), 1)
            else:
                st.session_state.hb = round(random.uniform(10.0, 16.5), 1)
                
            st.session_state.rbc = round(random.uniform(3.5, 5.5), 2)
            st.session_state.pcv = round(random.uniform(30.0, 45.0), 1)
            st.session_state.mcv = round(random.uniform(75.0, 95.0), 1)
            st.session_state.mch = round(random.uniform(25.0, 33.0), 1)
            st.session_state.mchc = round(random.uniform(31.0, 36.0), 1)
            
            st.success("Random test data generated! Click 'Predict Anemia Status' to analyze.")
    
    with col2:
        if st.button("🔄 Reset Form"):
            reset_input_values()
            st.rerun()
    
    with st.form("prediction_form", border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender", 
                options=["Select gender","Female", "Male"],
                index=0 if "gender" not in st.session_state else 
                      (0 if st.session_state.gender == "Select gender" else 
                       (1 if st.session_state.gender == "Female" else 2)),
                key="gender",
                help="Select the patient's gender - important as normal ranges differ by gender"
            )
            
            age = st.number_input(
                "Age", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.get("age", None),
                step=1,
                help="Enter patient's age in years - anemia risk and normal ranges can vary by age"
            )
            
            hb = st.number_input(
                "Hemoglobin (Hb) in g/dL", 
                min_value=5.0, 
                max_value=20.0, 
                value=st.session_state.get("hb", None),
                step=0.1,
                help="Hemoglobin is the protein in red blood cells that carries oxygen. Normal: Males 13.5-17.5, Females 12.0-15.5 g/dL"
            )
            
            rbc = st.number_input(
                "Red Blood Cell Count (RBC) in million/μL", 
                min_value=1.0, 
                max_value=8.0, 
                value=st.session_state.get("rbc", None),
                step=0.1,
                help="Total number of red blood cells per microliter. Normal: Males 4.5-5.9, Females 4.0-5.2 million/μL"
            )
        
        with col2:
            pcv = st.number_input(
                "Packed Cell Volume (PCV) in %", 
                min_value=10.0, 
                max_value=60.0, 
                value=st.session_state.get("pcv", None),
                step=0.1,
                help="Percentage of blood volume occupied by red blood cells. Normal: Males 40-52%, Females 37-47%"
            )
            
            mcv = st.number_input(
                "Mean Corpuscular Volume (MCV) in fL", 
                min_value=20.0, 
                max_value=120.0, 
                value=st.session_state.get("mcv", None),
                step=0.1,
                help="Average size of red blood cells. Normal: 80-100 fL. Low = microcytic, High = macrocytic"
            )
            
            mch = st.number_input(
                "Mean Corpuscular Hemoglobin (MCH) in pg", 
                min_value=10.0, 
                max_value=40.0, 
                value=st.session_state.get("mch", None),
                step=0.1,
                help="Average amount of hemoglobin per red blood cell. Normal: 27-33 pg"
            )
            
            mchc = st.number_input(
                "Mean Corpuscular Hemoglobin Concentration (MCHC) in g/dL", 
                min_value=25.0, 
                max_value=40.0, 
                value=st.session_state.get("mchc", None),
                step=0.1,
                help="Average concentration of hemoglobin in a given volume of red blood cells. Normal: 32-36 g/dL"
            )
        
        gender_encoded = 1 if gender == "Female" else (0 if gender == "Male" else None)
        
        user_input = {
            'Age': age,
            'Hb': hb,
            'RBC': rbc,
            'PCV': pcv,
            'MCV': mcv,
            'MCH': mch,
            'MCHC': mchc,
            'Gender_Encoded': gender_encoded
        }
        
        submitted = st.form_submit_button("Predict Anemia Status")
        
        if submitted:
            is_valid, error_message = validate_input(user_input)
            
            if not is_valid:
                st.warning(error_message)
                submitted = False
    
    return user_input, submitted

def display_exploratory_analysis(df):
    st.subheader("Exploratory Data Analysis")
    
    with st.expander("View dataset sample"):
        st.dataframe(df.head())
    
    st.write("### Dataset Overview")
    overview_col1, overview_col2 = st.columns(2)
    
    with overview_col1:
        st.write(f"Total records: {df.shape[0]}")
        st.write(f"Anemic cases: {df[df['Decision_Class'] == 1].shape[0]} ({df[df['Decision_Class'] == 1].shape[0]/df.shape[0]*100:.1f}%)")
        st.write(f"Non-anemic cases: {df[df['Decision_Class'] == 0].shape[0]} ({df[df['Decision_Class'] == 0].shape[0]/df.shape[0]*100:.1f}%)")
    
    with overview_col2:
        fig_diagnosis = create_anemia_distribution_pie(df)
        st.plotly_chart(fig_diagnosis)
    
    df_viz = df.copy()
    df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
    df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    st.write("### Gender Distribution")
    gender_col1, gender_col2 = st.columns(2)
    
    with gender_col1:
        gender_figs = create_gender_distribution_plots(df)
        st.plotly_chart(gender_figs['gender_pie'])
    
    with gender_col2:
        st.plotly_chart(gender_figs['gender_diagnosis_bar'])
    
    st.write("### Age Distribution")
    age_col1, age_col2 = st.columns(2)
    
    with age_col1:
        age_figs = create_age_distribution_plots(df)
        st.plotly_chart(age_figs['age_histogram'])
    
    with age_col2:
        st.plotly_chart(age_figs['age_anemia_bar'])
    
    st.write("### Hematological Parameters")
    
    param_options = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    selected_param = st.selectbox("Select hematological parameter to visualize:", param_options)
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        param_figs = create_hematological_parameter_plots(df, selected_param)
        st.plotly_chart(param_figs['histogram'])
    
    with param_col2:
        st.plotly_chart(param_figs['boxplot'])
    
    st.write(f"### {selected_param} Analysis")
    unit = FEATURES[selected_param]['unit'] if selected_param in FEATURES else ''
    
    st.write(f"**Normal Range for {selected_param}:**")
    
    if selected_param in NORMAL_RANGES:
        if 'all' in NORMAL_RANGES[selected_param]:
            min_val, max_val = NORMAL_RANGES[selected_param]['all']
            st.write(f"* General: {min_val} - {max_val} {unit}")
        else:
            for gender, range_vals in NORMAL_RANGES[selected_param].items():
                if gender != 'unit':
                    min_val, max_val = range_vals
                    st.write(f"* {gender.upper()}: {min_val} - {max_val} {unit}")
    
    st.write(f"**{selected_param} Statistics by Gender and Diagnosis:**")
    stats_df = df_viz.groupby(['Gender', 'Diagnosis'])[selected_param].agg(['mean', 'std', 'min', 'max']).round(2)
    st.dataframe(stats_df)
    
    st.write("### Correlation Between Parameters")
    
    corr_fig = create_correlation_heatmap(df)
    st.plotly_chart(corr_fig)
    
    st.write("### Research Findings")
    st.info("""
    According to the research by Mojumdar et al. (2025), the Chi-square test yielded a p-value of 4.1929 × 10^-29, 
    indicating no significant association between gender and diagnostic outcomes. However, both Z-test and T-test 
    revealed significant gender differences in hemoglobin levels, with p-values of 3.4789 × 10^-33 and 4.1586 × 10^-24, 
    respectively. This underscores the importance of gender when analyzing hemoglobin variations in anemia diagnosis.
    """)

def display_interpretation_guide():
    st.subheader("Interpretation Guide")
    
    tabs = st.tabs(["Hemoglobin", "RBC", "PCV", "MCV", "MCH", "MCHC"])
    
    with tabs[0]:
        st.write("### Hemoglobin (Hb)")
        st.write("""
        **What it is:** Hemoglobin is the protein in red blood cells that carries oxygen.
        
        **Normal ranges:**
        - Adult males: 13.5 to 17.5 g/dL
        - Adult females: 12.0 to 15.5 g/dL
        
        **In anemia:**
        - Low hemoglobin levels are the primary indicator of anemia
        - Values below normal range suggest insufficient oxygen-carrying capacity
        
        **Clinical significance:**
        - Severe anemia: Hb < 8 g/dL
        - Moderate anemia: Hb 8-10 g/dL
        - Mild anemia: Hb 10-12 g/dL (females) or 10-13 g/dL (males)
        """)
    
    with tabs[1]:
        st.write("### Red Blood Cell Count (RBC)")
        st.write("""
        **What it is:** The total number of red blood cells per volume of blood.
        
        **Normal ranges:**
        - Adult males: 4.5 to 5.9 million cells/μL
        - Adult females: 4.0 to 5.2 million cells/μL
        
        **In anemia:**
        - Low RBC count often accompanies low hemoglobin
        - Can help distinguish between different types of anemia
        
        **Clinical significance:**
        - Low RBC with normal MCV: Normocytic anemia (e.g., chronic disease, acute blood loss)
        - Low RBC with high MCV: Macrocytic anemia (e.g., vitamin B12 or folate deficiency)
        - Low RBC with low MCV: Microcytic anemia (e.g., iron deficiency, thalassemia)
        """)
    
    with tabs[2]:
        st.write("### Packed Cell Volume (PCV) / Hematocrit")
        st.write("""
        **What it is:** The percentage of blood volume that consists of red blood cells.
        
        **Normal ranges:**
        - Adult males: 40% to 52%
        - Adult females: 37% to 47%
        
        **In anemia:**
        - Reduced PCV indicates decreased red cell mass
        - Generally follows hemoglobin trends
        
        **Clinical significance:**
        - Used to monitor hydration status along with anemia
        - Helps assess response to treatment
        - Important for determining severity of anemia
        """)
    
    with tabs[3]:
        st.write("### Mean Corpuscular Volume (MCV)")
        st.write("""
        **What it is:** The average size of red blood cells.
        
        **Normal range:** 80 to 100 femtoliters (fL)
        
        **In anemia:**
        - Low MCV (<80 fL): Microcytic anemia (small RBCs)
        - Normal MCV (80-100 fL): Normocytic anemia (normal-sized RBCs)
        - High MCV (>100 fL): Macrocytic anemia (large RBCs)
        
        **Clinical significance:**
        - Critical for classifying anemia type:
          * Microcytic: Often iron deficiency, thalassemia
          * Normocytic: Chronic disease, acute blood loss, kidney disease
          * Macrocytic: Vitamin B12/folate deficiency, liver disease, alcoholism
        """)
    
    with tabs[4]:
        st.write("### Mean Corpuscular Hemoglobin (MCH)")
        st.write("""
        **What it is:** The average amount of hemoglobin per red blood cell.
        
        **Normal range:** 27 to 33 picograms (pg)
        
        **In anemia:**
        - Low MCH: Hypochromic anemia (less hemoglobin per cell)
        - Normal/high MCH: Normochromic or hyperchromic anemia
        
        **Clinical significance:**
        - Often correlates with MCV
        - Low in iron deficiency anemia and thalassemia
        - Helps distinguish between different causes of microcytic anemia
        """)
    
    with tabs[5]:
        st.write("### Mean Corpuscular Hemoglobin Concentration (MCHC)")
        st.write("""
        **What it is:** The average concentration of hemoglobin in a given volume of red blood cells.
        
        **Normal range:** 32 to 36 g/dL
        
        **In anemia:**
        - Low MCHC: Hypochromic anemia (less hemoglobin concentration)
        - Normal MCHC: Normochromic anemia
        
        **Clinical significance:**
        - Provides information about hemoglobin synthesis
        - Decreased in iron deficiency anemia
        - Helps in differential diagnosis of microcytic anemias
        """)

def display_clinical_interpretation(prediction, user_input):
    st.subheader("Clinical Interpretation")
    
    abnormal_params = 0
    gender = 'Female' if user_input['Gender_Encoded'] == 1 else 'Male'
    gender_code = 'f' if gender == 'Female' else 'm'
    
    for param, ranges in NORMAL_RANGES.items():
        if param in user_input:
            if 'all' in ranges:
                min_val, max_val = ranges['all']
            else:
                min_val, max_val = ranges[gender_code]
                
            if user_input[param] < min_val or user_input[param] > max_val:
                abnormal_params += 1
    
    if prediction == 1:
        st.write("### Suggested Anemia Classification:")
        
        if user_input['MCV'] < 80:
            anemia_type = "Microcytic Anemia"
            st.write(f"**{anemia_type}** (MCV < 80 fL)")
            st.write("""
            **Possible causes:**
            - Iron deficiency anemia
            - Thalassemia
            - Anemia of chronic disease (some cases)
            - Lead poisoning
            
            **Recommended additional tests:**
            - Serum ferritin
            - Iron studies (serum iron, TIBC, transferrin saturation)
            - Hemoglobin electrophoresis (for thalassemia)
            """)
            
        elif user_input['MCV'] > 100:
            anemia_type = "Macrocytic Anemia"
            st.write(f"**{anemia_type}** (MCV > 100 fL)")
            st.write("""
            **Possible causes:**
            - Vitamin B12 deficiency
            - Folate deficiency
            - Liver disease
            - Alcoholism
            - Myelodysplastic syndromes
            
            **Recommended additional tests:**
            - Serum B12 and folate levels
            - Liver function tests
            - Reticulocyte count
            """)
            
        else:
            anemia_type = "Normocytic Anemia"
            st.write(f"**{anemia_type}** (MCV 80-100 fL)")
            st.write("""
            **Possible causes:**
            - Anemia of chronic disease
            - Kidney disease
            - Hemolytic anemia
            - Acute blood loss
            - Mixed nutritional deficiencies
            
            **Recommended additional tests:**
            - CRP and ESR (inflammation markers)
            - Kidney function tests
            - Reticulocyte count
            - Bilirubin (for hemolysis)
            """)
        
        if user_input['MCHC'] < 32:
            st.write("**Features of Hypochromic Anemia** (MCHC < 32 g/dL)")
            
        if gender == 'Female':
            if user_input['Hb'] < 8:
                severity = "Severe"
            elif user_input['Hb'] < 10:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:
            if user_input['Hb'] < 8:
                severity = "Severe"
            elif user_input['Hb'] < 11:
                severity = "Moderate"
            else:
                severity = "Mild"
                
        st.write(f"**Severity: {severity}** based on hemoglobin level")
        
    else:
        if abnormal_params > 0:
            st.write(f"""
            The analysis suggests a **non-anemic** status, despite {abnormal_params} 
            parameter(s) outside the reference range. This could indicate:
            
            - Early changes not yet manifesting as clinical anemia
            - Compensated anemia
            - Recent recovery from anemia
            - Individual variations in baseline values
            
            Consider monitoring if other clinical symptoms are present.
            """)
        else:
            st.write("""
            All hematological parameters are within normal range, consistent with the 
            **non-anemic** prediction. No further hematological evaluation is indicated 
            based on these results alone.
            """)

def display_reference_table(user_input):
    st.subheader("Parameter Reference Ranges")
    
    reference_df = []
    gender = 'Female' if user_input['Gender_Encoded'] == 1 else 'Male'
    gender_code = 'f' if gender == 'Female' else 'm'
    
    for param, config in FEATURES.items():
        if param != 'Gender' and param != 'Gender_Encoded' and param in user_input:
            unit = config.get('unit', '')
            
            if param in NORMAL_RANGES:
                if 'all' in NORMAL_RANGES[param]:
                    reference_min, reference_max = NORMAL_RANGES[param]['all']
                    gender_specific = 'No'
                else:
                    reference_min, reference_max = NORMAL_RANGES[param][gender_code]
                    gender_specific = 'Yes'
                
                status = 'Normal'
                if user_input[param] < reference_min:
                    status = 'Low'
                elif user_input[param] > reference_max:
                    status = 'High'
                
                reference_df.append({
                    'Parameter': param,
                    'User Value': f"{user_input[param]:.1f} {unit}",
                    'Reference Range': f"{reference_min:.1f} - {reference_max:.1f} {unit}",
                    'Gender Specific': gender_specific,
                    'Status': status
                })
    
    if reference_df:
        reference_df = pd.DataFrame(reference_df)
        
        def color_status(val):
            if val == 'Low':
                return 'background-color: #B71C1C; color: white; font-weight: bold'
            elif val == 'High':
                return 'background-color: #0D47A1; color: white; font-weight: bold'
            else:
                return 'background-color: #1B5E20; color: white; font-weight: bold'
        
        st.dataframe(reference_df.style.applymap(color_status, subset=['Status']))
    else:
        st.write("No reference ranges available for the provided parameters.")

def display_symptom_prediction_interface():
    st.subheader("Anemia Prediction Based on Symptoms & Risk Factors")
    st.write("Select symptoms and risk factors to predict anemia status without blood tests.")
    
    st.info("⚠️ Note: You must select at least one symptom or risk factor to get a prediction.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📋 Generate Random Symptom Data", help="Fill the form with random symptom data for testing"):
            import random
            
            st.session_state.symptom_gender = random.choice(["Female", "Male"])
            st.session_state.symptom_age = random.randint(18, 80)
            
            for symptom in ['fatigue', 'shortness_breath', 'dizziness', 'pale_skin', 'heart_racing', 
                            'headaches', 'cold_hands_feet', 'brittle_nails', 'poor_concentration']:
                st.session_state[symptom] = random.choice([True, False]) if random.random() > 0.3 else False
                
            for risk in ['heavy_periods', 'recent_blood_loss', 'vegetarian_diet', 'gi_disorders', 'chronic_disease']:
                st.session_state[risk] = random.choice([True, False]) if random.random() > 0.6 else False
                
            if st.session_state.symptom_gender == "Male":
                st.session_state.heavy_periods = False
                
            st.success("Random symptom data generated! Click 'Predict Anemia Risk' to analyze.")
    
    with col2:
        if st.button("🔄 Reset Symptom Form"):
            reset_keys = [
                'symptom_gender', 'symptom_age', 'fatigue', 'shortness_breath', 'dizziness', 
                'pale_skin', 'heart_racing', 'headaches', 'cold_hands_feet', 'brittle_nails', 
                'poor_concentration', 'heavy_periods', 'recent_blood_loss', 'vegetarian_diet', 
                'gi_disorders', 'chronic_disease'
            ]
            
            for key in reset_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.session_state['symptom_gender'] = 'Select gender'
            st.rerun()
    
    with st.form("symptom_prediction_form", border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender", 
                options=["Select gender", "Female", "Male"],
                index=0 if "symptom_gender" not in st.session_state else 
                      (0 if st.session_state.symptom_gender == "Select gender" else 
                       (1 if st.session_state.symptom_gender == "Female" else 2)),
                key="symptom_gender"
            )
            
            age = st.number_input(
                "Age", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.get("symptom_age", None),
                step=1
            )
            
            st.markdown("##### Primary Symptoms:")
            fatigue = st.checkbox("Fatigue", value=st.session_state.get("fatigue", False), key="fatigue", 
                                help="Feeling unusually tired or having less energy than normal")
            shortness_breath = st.checkbox("Shortness of Breath", value=st.session_state.get("shortness_breath", False), 
                                         key="shortness_breath",
                                         help="Difficulty breathing, especially during physical activity")
            dizziness = st.checkbox("Dizziness", value=st.session_state.get("dizziness", False), key="dizziness",
                                   help="Feeling lightheaded or unsteady")
            pale_skin = st.checkbox("Pale Skin", value=st.session_state.get("pale_skin", False), key="pale_skin",
                                   help="Paleness of skin, nail beds, or inside of eyelids")
            heart_racing = st.checkbox("Racing/Irregular Heartbeat", value=st.session_state.get("heart_racing", False), 
                                      key="heart_racing",
                                      help="Feeling that your heart is beating faster than normal")
            
        with col2:
            st.markdown("##### Secondary Symptoms:")
            headaches = st.checkbox("Headaches", value=st.session_state.get("headaches", False), key="headaches")
            cold_hands_feet = st.checkbox("Cold Hands/Feet", value=st.session_state.get("cold_hands_feet", False), 
                                         key="cold_hands_feet")
            brittle_nails = st.checkbox("Brittle Nails", value=st.session_state.get("brittle_nails", False), 
                                       key="brittle_nails")
            poor_concentration = st.checkbox("Poor Concentration", value=st.session_state.get("poor_concentration", False), 
                                           key="poor_concentration")
            
            st.markdown("##### Risk Factors:")
            heavy_periods = st.checkbox("Heavy Menstrual Periods", 
                                       value=st.session_state.get("heavy_periods", False) and gender == "Female", 
                                       key="heavy_periods", disabled=gender != "Female")
            recent_blood_loss = st.checkbox("Recent Blood Loss/Donation", 
                                          value=st.session_state.get("recent_blood_loss", False), 
                                          key="recent_blood_loss")
            vegetarian_diet = st.checkbox("Vegetarian/Vegan Diet", 
                                         value=st.session_state.get("vegetarian_diet", False), 
                                         key="vegetarian_diet")
            gi_disorders = st.checkbox("GI Disorders", 
                                      value=st.session_state.get("gi_disorders", False), 
                                      key="gi_disorders",
                                      help="Gastrointestinal issues like ulcers, Crohn's disease, celiac disease")
            chronic_disease = st.checkbox("Chronic Disease", 
                                         value=st.session_state.get("chronic_disease", False), 
                                         key="chronic_disease",
                                         help="Conditions like kidney disease, cancer, rheumatoid arthritis")
        
        gender_encoded = 1 if gender == "Female" else (0 if gender == "Male" else None)
        
        symptoms = {
            'Gender': gender_encoded,
            'Age': age,
            'Fatigue': int(fatigue),
            'Shortness_of_Breath': int(shortness_breath),
            'Dizziness': int(dizziness),
            'Pale_Skin': int(pale_skin),
            'Heart_Racing': int(heart_racing),
            'Headaches': int(headaches),
            'Cold_Hands_Feet': int(cold_hands_feet),
            'Brittle_Nails': int(brittle_nails),
            'Poor_Concentration': int(poor_concentration),
            'Heavy_Periods': int(heavy_periods if gender == "Female" else 0),
            'Recent_Blood_Loss': int(recent_blood_loss),
            'Vegetarian_Diet': int(vegetarian_diet),
            'GI_Disorders': int(gi_disorders),
            'Chronic_Disease': int(chronic_disease)
        }
        
        submitted = st.form_submit_button("Predict Anemia Risk")
        
        if submitted:
            if gender == "Select gender":
                st.warning("Please select a gender to continue")
                submitted = False
            elif age < 1:
                st.warning("Please enter a valid age")
                submitted = False
                
            symptom_count = sum([
                fatigue, shortness_breath, dizziness, pale_skin, heart_racing, 
                headaches, cold_hands_feet, brittle_nails, poor_concentration,
                heavy_periods, recent_blood_loss, vegetarian_diet, gi_disorders, chronic_disease
            ])
            
            if symptom_count == 0:
                st.warning("Please select at least one symptom or risk factor to make a prediction")
                submitted = False
    
    return symptoms, submitted

def display_symptom_prediction_result(prediction, probability):
    if prediction is None:
        return
    
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        if prediction == 1:
            st.error("Risk Assessment: **High Risk for Anemia**")
        else:
            st.success("Risk Assessment: **Low Risk for Anemia**")
        
        if probability is not None:
            st.write(f"Confidence: **{probability:.2%}**")
    
    with result_col2:
        if probability is not None:
            fig = create_prediction_gauge(probability, prediction, title="Anemia Risk")
            st.plotly_chart(fig)
    
    st.subheader("Recommendations")
    
    if prediction == 1:
        if probability > 0.8:
            st.error("""
            **High risk for anemia detected.** Based on your symptoms and risk factors, 
            you should consult a healthcare provider as soon as possible for blood tests to confirm diagnosis.
            """)
        else:
            st.warning("""
            **Moderate to high risk for anemia detected.** Consider scheduling an appointment with 
            a healthcare provider for evaluation and blood tests.
            """)
    else:
        if probability < 0.2:
            st.success("""
            **Low risk for anemia detected.** Your symptoms are unlikely to be related to anemia. 
            If symptoms persist, consider consulting a healthcare provider for other potential causes.
            """)
        else:
            st.info("""
            **Low to moderate risk for anemia detected.** While anemia is less likely, if symptoms persist 
            or worsen, consider discussing with a healthcare provider at your next visit.
            """)
    
    st.info("""
    **Note:** This assessment is based on symptoms and risk factors only and cannot replace 
    a proper medical diagnosis. A blood test is required to definitively diagnose anemia.
    """)

def main():
    st.set_page_config(
        page_title="Anemia Diagnosis Tool",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.mayoclinic.org/diseases-conditions/anemia/symptoms-causes/syc-20351360',
            'Report a bug': "mailto:support@anemiaproject.org",
            'About': "# Anemia Diagnostic Assistant\nAn internship project for advanced anemia prediction using machine learning."
        }
    )
    
    st.markdown("""
    <style>
    :root {
        --background-color: #0e1117;
        --secondary-background-color: #1e2129;
        --primary-color: #ff4b4b;
        --secondary-color: #ff8a8a;
        --text-color: #fafafa;
        --font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 6px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stForm"] {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    div[data-testid="stExpander"] {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    div[data-testid="stMarkdown"] a {
        color: var(--primary-color);
        text-decoration: none;
        border-bottom: 1px dotted var(--primary-color);
    }
    
    div[data-testid="stMarkdown"] a:hover {
        border-bottom: 1px solid var(--primary-color);
    }
    
    .row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    
    .row-widget.stRadio > div > label {
        padding: 10px 15px;
        border-radius: 5px;
        margin-right: 10px;
        background-color: var(--secondary-background-color);
        transition: all 0.3s;
    }
    
    .row-widget.stRadio > div > label:hover {
        background-color: rgba(255,75,75,0.2);
    }
    
    .row-widget.stRadio > div [data-testid="stMarkdownContainer"] > p {
        font-size: 0.9em;
    }
    
    div[role="alert"] {
        border-radius: 8px;
        padding: 1rem;
    }
    
    div[data-baseweb="notification"] {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
        <h1 style="margin: 0; display: flex; align-items: center;"><span style="margin-right: 0.5rem;">🩸</span> Anemia Diagnostic Assistant</h1>
        <div style="background-color: #ff4b4b; color: white; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 14px;">
            Internship Project
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio("Go to", ["About", "Prediction Tool", "Symptom Assessment", "Exploratory Analysis", "Interpretation Guide"])
    
    df = load_data()
    model = load_model()
    symptom_model = load_symptom_model()
    
    if app_mode == "About":
        st.header("Introduction")
        st.markdown("""
        <div class="about-section">
            <p>This project implements a web application for predicting anemia risk using machine learning. The app allows users to input their hematological parameters and receive an anemia diagnosis based on a trained predictive model. 
            It leverages Streamlit for an interactive interface and uses advanced machine learning techniques to assess anemia status.</p> 
            <p>This project builds upon the foundational work of <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/">Mojumdar et al., 2025</a>, which
            used traditional statistical methods like Chi-Square tests and T-tests to explore associations between biological factors and anemia.
            Building on these insights, our goal is to apply machine learning techniques to uncover more complex, non-linear relationships in
            the data. By doing so, we aim to enhance diagnostic accuracy and provide deeper insights into the factors influencing anemia.</p>
            <p>What makes this project particularly unique is the collaboration with <a href="https://scholar.google.com.tw/citations?user=MwIr5fMAAAAJ&hl=en">Dr. Gem Wu</a>,
            a <strong>hematologist</strong> working at <strong>Chang Gung Memorial Hospital</strong>, Taiwan. With Dr. Gem Wu providing expert support on the hematological aspects of anemia, 
            we have been able to incorporate expert insights into the hematological aspects of anemia, 
            ensuring that our analysis is grounded in medical realities and clinical perspectives. 
            This collaboration enhances the accuracy and relevance of our findings, bridging the gap between data science and clinical expertise.</p>
            
            
        </div>
        """, unsafe_allow_html=True)    
        
        st.markdown("<hr style='margin: 30px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        st.header("About Anemia")
        st.markdown("""
        <div class="about-anemia">
            <p>Anemia is a condition characterized by a lack of healthy red blood cells or hemoglobin to carry
            adequate oxygen to the body's tissues. It can cause fatigue, weakness, pale skin, and shortness
            of breath. Early detection is crucial for effective treatment and management.</p>
             <p>This tool uses machine learning to predict anemia based on:</p>
            <ul>
                <li>Demographic information (age, gender)</li>
                <li>Hematological parameters (Hb, RBC, PCV, MCV, MCH, MCHC)</li>
                <li>Symptoms and risk factors</li>
            </ul>
            <p>The model was trained on data from a study conducted at <a href="http://data.mendeley.com/datasets/y7v7ff3wpj/1">Aalok Healthcare Ltd., Bangladesh</a>,
            featuring comprehensive hematological profiles of patients.</p>
            
           
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 30px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        st.header("References")
        st.markdown("""
        <div class="references">
            <ul>
                <li>Paper: Mojumdar et al., "AnaDetect: An extensive dataset for advancing anemia detection, diagnostic methods, and predictive analytics in healthcare", PMC (<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/">https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/</a>)</li>
                <li>Anemia: Approach and Evaluation (<a href="https://manualofmedicine.com/topics/hematology-oncology/anemia-approach-and-evaluation/">https://manualofmedicine.com/topics/hematology-oncology/anemia-approach-and-evaluation/</a>)</li>
                <li>Source Code (Anemia Detection with Machine Learning): "Anemia Detection with Machine Learning", GitHub repository (<a href="https://github.com/maladeep/anemia-detection-with-machine-learning">https://github.com/maladeep/anemia-detection-with-machine-learning</a>)</li>
                <li>Source Code (Anemia Prediction): "Anemia Prediction", GitHub repository (<a href="https://github.com/muscak/anemia-prediction">https://github.com/muscak/anemia-prediction</a>)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif app_mode == "Prediction Tool":
        if model is None or df is None:
            st.error("Could not load model or dataset. Please check the files.")
        else:
            user_input, submitted = display_prediction_interface()
            
            if submitted:
                with st.spinner("Analyzing hematological parameters..."):
                    prediction, probability, error = make_prediction(user_input, model)
                    display_prediction_result(prediction, probability)
                    display_reference_table(user_input)
                    display_clinical_interpretation(prediction, user_input)
                    
                    st.subheader("Parameter Comparison with Normal Ranges")
                    radar_fig = create_feature_comparison_radar(user_input, NORMAL_RANGES)
                    st.plotly_chart(radar_fig)
    
    elif app_mode == "Symptom Assessment":
        if symptom_model is None:
            st.error("Could not load symptom prediction model. Please ensure the model is trained and saved properly.")
        else:
            st.header("Symptom-Based Anemia Risk Assessment")
            st.write("""
            Use this tool to assess your risk of anemia based on symptoms and risk factors, 
            without requiring blood test results. This can help determine if you should consult 
            a healthcare provider for further evaluation.
            """)
            
            symptoms, submitted = display_symptom_prediction_interface()
            
            if submitted:
                with st.spinner("Analyzing symptoms and risk factors..."):
                    try:
                        symptom_count = sum([symptoms[k] for k in symptoms if k not in ['Gender', 'Age']])
                        
                        if symptom_count == 0:
                            st.warning("⚠️ No symptoms or risk factors selected. Please select at least one to get an accurate prediction.")
                        else:
                            prediction, probability = predict_anemia_risk(symptom_model, symptoms)
                            
                            display_symptom_prediction_result(prediction, probability)
                            
                            st.subheader("Symptom Analysis")
                            
                            st.write(f"You reported **{symptom_count}** out of 14 possible symptoms and risk factors.")
                            
                            if symptom_count > 0:
                                st.write("**Key contributing factors to your risk assessment:**")
                                
                                high_importance = ["Fatigue", "Shortness_of_Breath", "Pale_Skin", "Heavy_Periods", "Recent_Blood_Loss"]
                                medium_importance = ["Dizziness", "Heart_Racing", "Chronic_Disease", "GI_Disorders"]
                                
                                high_importance_present = [s for s in high_importance if symptoms.get(s, 0) == 1]
                                medium_importance_present = [s for s in medium_importance if symptoms.get(s, 0) == 1]
                                
                                if high_importance_present:
                                    st.write("**Strong indicators:**")
                                    for symptom in high_importance_present:
                                        readable_name = symptom.replace('_', ' ').title()
                                        st.write(f"- {readable_name}")
                                
                                if medium_importance_present:
                                    st.write("**Moderate indicators:**")
                                    for symptom in medium_importance_present:
                                        readable_name = symptom.replace('_', ' ').title()
                                        st.write(f"- {readable_name}")
                                
                                st.subheader("Possible Types of Anemia")
                                
                                if symptoms.get('Heavy_Periods', 0) == 1 or symptoms.get('Recent_Blood_Loss', 0) == 1:
                                    st.write("**Iron Deficiency Anemia** - Blood loss is a common cause of iron deficiency anemia, especially in women with heavy menstrual periods.")
                                
                                if symptoms.get('Vegetarian_Diet', 0) == 1:
                                    st.write("**Vitamin B12 Deficiency Anemia** - Vegetarian or vegan diets may increase risk of B12 deficiency without proper supplementation.")
                                
                                if symptoms.get('GI_Disorders', 0) == 1:
                                    st.write("**Anemia of Chronic Disease/Inflammation** - Gastrointestinal disorders can affect nutrient absorption and contribute to anemia.")
                                
                                if symptoms.get('Chronic_Disease', 0) == 1:
                                    st.write("**Anemia of Chronic Disease** - Chronic conditions can affect red blood cell production through various mechanisms.")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
    
    elif app_mode == "Exploratory Analysis":
        if df is None:
            st.error("Could not load dataset. Please check the file path.")
        else:
            display_exploratory_analysis(df)
    
    elif app_mode == "Interpretation Guide":
        display_interpretation_guide()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background-color: #1e2129; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="color: #ff4b4b; margin-top: 0; font-size: 1.1rem;">About this Internship Project</h4>
        <p style="font-size: 0.9rem; margin-bottom: 8px;">This app was developed as part of a first year internship program to demonstrate the application of machine learning in anemia diagnosis.</p>
        <p style="font-size: 0.9rem; margin-bottom: 8px;">Data source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/" style="color: #ff8a8a;">Mojumdar et al., 2025</a></p>
        <p style="font-size: 0.8rem; margin-bottom: 0;"><strong>Note:</strong> This app is for educational purposes only and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="position: fixed; bottom: 20px; font-size: 0.7rem; opacity: 0.7; width: 100%; text-align: center;">
        v1.0.0 | Anemia Diagnostic Assistant | © 2025 Internship Project
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()