import streamlit as st
import joblib

model=joblib.load(open('models/model.pkl','rb'))

st.header('Salary Prediction App')
st.balloons()

experience=st.number_input('Experience',min_value=0.0,max_value=50.0,step=0.1)
age=st.number_input('Age',min_value=18.0,max_value=70.0,step=0.1)
projects_completed=st.number_input('Projects Completed',min_value=0.0,max_value=100.0,step=1.0)
hours_per_week=st.number_input('Hours per week',min_value=0.0,max_value=168.0,step=1.0)

# categorical selectors (use the same categories used during preprocessing)
job_roles = ['AI Specialist', 'Analyst', 'Data Scientist', 'DL Engineer', 'ML Engineer']
education_levels = ["Bachelor's", "Master's", 'PhD']
locations = ['Rural', 'Suburban', 'Urban']

job_role = st.selectbox('Job Role', job_roles)
education_level = st.selectbox('Education Level', education_levels)
location = st.selectbox('Location', locations)

# load the preprocessing pipeline (one-hot encoder + scaler)
try:
    preprocessor = joblib.load(open('preprocesser/X_preprocessed.pkl', 'rb'))
except FileNotFoundError:
    st.error('Preprocessor not found. Run data_preprocessing.py first.')
    preprocessor = None

if st.button('Predict'):
    if preprocessor is None:
        st.warning('Cannot make prediction without preprocessor.')
    else:
        import pandas as pd
        # create dataframe for a single example
        input_dict = {
            'experience': [experience],
            'age': [age],
            'projects_completed': [projects_completed],
            'hours_per_week': [hours_per_week],
            'job_role': [job_role],
            'education_level': [education_level],
            'location': [location]
        }
        input_df = pd.DataFrame(input_dict)
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)
        st.success(f'Predicted Salary: ${prediction[0]:.2f}')

st.snow()