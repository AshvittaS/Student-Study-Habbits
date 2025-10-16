import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
# --- Load model and discretizer ---
with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("kbins_discretizer.pkl", "rb") as f:
    kbins = pickle.load(f)

# --- Feature names ---
feature_names = [
    'study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage',
    'assignments_completed', 'participation_level_Low',
    'participation_level_Medium', 'internet_access_Yes',
    "parental_education_High School", "parental_education_Master's",
    'parental_education_PhD', 'extracurricular_Yes', 'part_time_job_Yes'
]

numeric_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 'assignments_completed']

# --- Streamlit setup ---
st.set_page_config(page_title="Student Result", page_icon="🏫", layout="wide")
st.markdown("""
<style>
body {background-color: #0B0C10; color: #C5C6C7;}
.stButton>button {background-color: #1F2833; color: #66FCF1; border-radius: 10px; height: 3em; font-weight: bold;}
.stSidebar {background-color: #1F2833; color: #66FCF1;}
h1, h2, h3 {color: #66FCF1;}
</style>
""", unsafe_allow_html=True)

# --- Galaxy Particles Background ---
st.markdown("""
<div id="particles-js" style="position: fixed; width: 100%; height: 100%; z-index: -1;"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 120, "density": {"enable": true, "value_area": 800}},
    "color": {"value": "#66FCF1"},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.7},
    "size": {"value": 2},
    "line_linked": {"enable": true, "distance": 120, "color": "#66FCF1", "opacity": 0.2, "width": 1},
    "move": {"enable": true, "speed": 1, "direction": "none", "random": true, "straight": false}
  },
  "interactivity": {
    "detect_on": "canvas",
    "events": {
      "onhover": {"enable": true, "mode": "repulse"},
      "onclick": {"enable": true, "mode": "push"}
    },
    "modes": {"repulse": {"distance": 100}, "push": {"particles_nb": 4}}
  },
  "retina_detect": true
});
</script>
""", unsafe_allow_html=True)

# --- Title ---
st.title(" Student Result Predictor")
st.markdown("Predict whether a student will **pass or fail**")

# --- Sidebar Inputs ---
st.sidebar.header("Student Inputs")

study_hours_per_week = st.sidebar.number_input("Study Hours per Week", min_value=0, max_value=60, value=10)
sleep_hours_per_day = st.sidebar.number_input("Sleep Hours per Day", min_value=0, max_value=12, value=7)
attendance_percentage = st.sidebar.number_input("Attendance Percentage", min_value=0, max_value=100, value=80)
assignments_completed = st.sidebar.number_input("Assignments Completed", min_value=0, max_value=20, value=10)

st.sidebar.subheader("Participation Level")
participation_level_Low = st.sidebar.selectbox("Low", [0, 1])
participation_level_Medium = st.sidebar.selectbox("Medium", [0, 1])

internet_access_Yes = st.sidebar.selectbox("Internet Access", [0, 1])
st.sidebar.subheader("Parental Education")
parental_education_High_School = st.sidebar.selectbox("High School", [0, 1])
parental_education_Masters = st.sidebar.selectbox("Master's", [0, 1])
parental_education_PhD = st.sidebar.selectbox("PhD", [0, 1])
extracurricular_Yes = st.sidebar.selectbox("Extracurricular Activities", [0, 1])
part_time_job_Yes = st.sidebar.selectbox("Part-Time Job", [0, 1])

# --- Predict Button ---
if st.sidebar.button("Predict"):

    # --- Prepare input for model ---
    input_df = pd.DataFrame([[ 
        study_hours_per_week,
        sleep_hours_per_day,
        attendance_percentage,
        assignments_completed,
        participation_level_Low,
        participation_level_Medium,
        internet_access_Yes,
        parental_education_High_School,
        parental_education_Masters,
        parental_education_PhD,
        extracurricular_Yes,
        part_time_job_Yes
    ]], columns=feature_names)

    input_df[numeric_cols] = kbins.transform(input_df[numeric_cols])

    # --- Prediction ---
    prediction = nb_model.predict(input_df)[0]
    probability = nb_model.predict_proba(input_df)[0][1]

    # --- Result card ---
    col1, col2 = st.columns([2,3])
    if prediction == 1:
        col1.metric(label="Predicted Result", value="PASS", delta=f"{probability*100:.1f}% Chance")
        st.success("✅ The student is likely to pass! Keep up the good habits.")
    else:
        col1.metric(label="Predicted Result", value="FAIL", delta=f"{(1-probability)*100:.1f}% Risk")
        st.error("⚠️ The student is at risk of failing. Consider improving study habits or attendance.")

    # --- Input overview ---
    with col2:
        st.subheader("Student Input Overview")
        st.markdown(f"""
        - Study Hours per Week: **{study_hours_per_week}**  
        - Sleep Hours per Day: **{sleep_hours_per_day}**  
        - Attendance: **{attendance_percentage}%**  
        - Assignments Completed: **{assignments_completed}**  
        - Participation (Low/Medium): **{participation_level_Low}/{participation_level_Medium}**  
        - Internet Access: **{'Yes' if internet_access_Yes else 'No'}**  
        - Parental Education (HS/Masters/PhD): **{parental_education_High_School}/{parental_education_Masters}/{parental_education_PhD}**  
        - Extracurricular: **{'Yes' if extracurricular_Yes else 'No'}**  
        - Part-Time Job: **{'Yes' if part_time_job_Yes else 'No'}**
        """)

    # --- Probability vs Study Hours chart ---
    study_range = pd.DataFrame({
        'study_hours_per_week': range(0, 61, 2),
        'sleep_hours_per_day': sleep_hours_per_day,
        'attendance_percentage': attendance_percentage,
        'assignments_completed': assignments_completed,
        'participation_level_Low': participation_level_Low,
        'participation_level_Medium': participation_level_Medium,
        'internet_access_Yes': internet_access_Yes,
        'parental_education_High School': parental_education_High_School,
        "parental_education_Master's": parental_education_Masters,
        'parental_education_PhD': parental_education_PhD,
        'extracurricular_Yes': extracurricular_Yes,
        'part_time_job_Yes': part_time_job_Yes
    })
    study_range[numeric_cols] = kbins.transform(study_range[numeric_cols])
    study_range['pass_prob'] = nb_model.predict_proba(study_range)[:,1]

    fig_line = px.line(
        study_range, x='study_hours_per_week', y='pass_prob',
        title='Probability vs Study Hours',
        markers=True
    )
    fig_line.update_layout(
        paper_bgcolor="#0B0C10",
        plot_bgcolor="#0B0C10",
        font_color="#C5C6C7",
        yaxis_title="Probability of Passing",
        xaxis_title="Study Hours per Week"
    )
    st.plotly_chart(fig_line, use_container_width=True)

