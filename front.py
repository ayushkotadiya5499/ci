import streamlit as st
import joblib
import pandas as pd

# --- Page configuration ---
st.set_page_config(
    page_title='Salary Predictor',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# --- Theme support (Day/Night) ---
THEMES = {
    'Day': {
        'background': '#F8FAFC',
        'text': '#0F172A',
        'card': '#FFFFFF',
        'accent': '#2563EB',
    },
    'Night': {
        'background': '#0B1220',
        'text': '#E2E8F0',
        'card': '#131E2B',
        'accent': '#7C3AED',
    },
}

if 'theme' not in st.session_state:
    st.session_state.theme = 'Day'

# Theme toggle button (uses emoji to switch between Day/Night)
cols = st.columns([9, 1])
with cols[1]:
    if st.button('🌙' if st.session_state.theme == 'Day' else '☀️', key='theme_toggle'):
        st.session_state.theme = 'Night' if st.session_state.theme == 'Day' else 'Day'

def apply_theme(theme_name: str):
    t = THEMES.get(theme_name, THEMES['Day'])
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {t['background']};
            color: {t['text']};
        }}
        .stButton>button {{
            background-color: {t['accent']} !important;
            color: white !important;
            border-radius: 0.6rem !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
        }}
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>div {{
            border-radius: 0.5rem !important;
        }}
        .css-1d391kg {{
            background-color: {t['card']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme(st.session_state.theme)

# --- Load model & preprocessing pipeline ---
model = joblib.load(open('models/model.pkl', 'rb'))
try:
    preprocessor = joblib.load(open('preprocesser/X_preprocessed.pkl', 'rb'))
except FileNotFoundError:
    preprocessor = None

# --- Header ---
st.markdown(
    "<div style='text-align: center; padding: 0.8rem 0 0.4rem;'>"
    "<h1 style='margin: 0; font-size: 2.2rem;'>💼 Salary Prediction App</h1>"
    "<p style='margin: 0.4rem 0 0; font-size: 1.05rem; color: #6B7280;'>"
    "Estimate your expected salary based on experience, role, and location.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('---')

# --- Input form ---
with st.form(key='input_form'):
    st.markdown('### 🔢 Numerical inputs')
    col1, col2 = st.columns(2)
    with col1:
        experience = st.number_input(
            'Experience (years)',
            min_value=0.0,
            max_value=50.0,
            step=0.1,
            help='Number of years working in the field.',
        )
        projects_completed = st.number_input(
            'Projects completed',
            min_value=0,
            max_value=100,
            step=1,
            help='Total number of professional projects delivered.',
        )

    with col2:
        age = st.number_input(
            'Age',
            min_value=18,
            max_value=70,
            step=1,
            help='Your current age.',
        )
        hours_per_week = st.number_input(
            'Hours per week',
            min_value=0,
            max_value=168,
            step=1,
            help='Average hours worked per week.',
        )

    st.markdown('### 🧩 Categorical inputs')
    job_roles = ['AI Specialist', 'Analyst', 'Data Scientist', 'DL Engineer', 'ML Engineer']
    education_levels = ["Bachelor's", "Master's", 'PhD']
    locations = ['Rural', 'Suburban', 'Urban']

    job_role = st.selectbox('Job Role', job_roles)
    education_level = st.selectbox('Education Level', education_levels)
    location = st.selectbox('Location', locations)

    submitted = st.form_submit_button('Predict', help='Click to generate a salary estimate.')

# --- Prediction logic ---
if submitted:
    if preprocessor is None:
        st.warning('Preprocessor not found. Run `data_preprocessing.py` first and try again.')
    else:
        input_df = pd.DataFrame(
            {
                'experience': [experience],
                'age': [age],
                'projects_completed': [projects_completed],
                'hours_per_week': [hours_per_week],
                'job_role': [job_role],
                'education_level': [education_level],
                'location': [location],
            }
        )
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)

        st.success(f'✅ Predicted Salary: **${prediction[0]:,.2f}**')
        st.markdown(
            "<p style='color: #64748B; margin: 0.3rem 0 0;'>" 
            "Keep in mind this is an estimate based on the provided inputs.</p>",
            unsafe_allow_html=True,
        )
        st.balloons()

# --- Footer ---
st.markdown('---')
with st.expander('ℹ️ About this app'):
    st.write(
        'This app uses a trained machine learning model to estimate an annual salary based on your experience, role, education, and location. '
        'Use the theme selector in the sidebar to switch between Day and Night modes.'
    )

with st.expander('📌 Tips for better predictions'):
    st.write(
        '- Be as realistic as possible with experience and hours per week.\n'
        '- This model is trained on sample data; predictions are not financial advice.\n'
        '- If you update the preprocessing pipeline, re-run `data_preprocessing.py` to refresh the model input pipeline.'
    )
    st.markdown('**What is a feature?**')
    st.write(
        'A *feature* is a single input variable used by the model to make a prediction. ' 
        'Providing accurate feature values helps the model produce better estimates.'
    )
    st.markdown('**Features used by this model**')
    st.write(
    """- **Experience (years):** Total years of professional experience; use decimal values for partial years.
        - **Age:** Your current age in years.
        - **Projects completed:** Count of professional projects you have delivered.
        - **Hours per week:** Typical average working hours per week.
        - **Job role:** Select the role that best matches your position (e.g., Data Scientist, ML Engineer).
        - **Education level:** Highest completed degree (Bachelor\'s, Master\'s, PhD).
        - **Location:** Broad location type (Rural, Suburban, Urban).
   """ )
    st.write(
        'Feature information tips: supply realistic numeric values, choose the categorical options that most closely match you, ' 
        'and re-run preprocessing if you changed how features are encoded.'
    )