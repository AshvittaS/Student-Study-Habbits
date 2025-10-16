import pickle
import gradio as gr
import pandas as pd

# Load the trained Random Forest model
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

feature_names = [
    'study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage',
    'assignments_completed', 'participation_level_Low',
    'participation_level_Medium', 'internet_access_Yes',
    "parental_education_High School", "parental_education_Master's",
    'parental_education_PhD', 'extracurricular_Yes', 'part_time_job_Yes'
]

def predict_result(
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
):
    input_data = pd.DataFrame([[ 
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

    prediction = rf_model.predict(input_data)[0]
    proba = rf_model.predict_proba(input_data)[0][1]
    return f"Predicted Result: {prediction} (Probability of success: {proba:.2f})"

# Define inputs in multiple columns
with gr.Blocks(css=".gradio-container {max-width: 1400px;}") as iface:
    gr.Markdown("# Student Result Predictor\nPredicts whether a student will succeed based on study habits, attendance, and other factors.")
    with gr.Row():
        with gr.Column():
            study_hours_per_week = gr.Number(label="Study Hours per Week")
            sleep_hours_per_day = gr.Number(label="Sleep Hours per Day")
            attendance_percentage = gr.Number(label="Attendance Percentage")
            assignments_completed = gr.Number(label="Assignments Completed")
            participation_level_Low = gr.Number(label="Participation Level Low (0 or 1)")
            participation_level_Medium = gr.Number(label="Participation Level Medium (0 or 1)")
        with gr.Column():
            internet_access_Yes = gr.Number(label="Internet Access Yes (0 or 1)")
            parental_education_High_School = gr.Number(label="Parental Education: High School (0 or 1)")
            parental_education_Masters = gr.Number(label="Parental Education: Master's (0 or 1)")
            parental_education_PhD = gr.Number(label="Parental Education: PhD (0 or 1)")
            extracurricular_Yes = gr.Number(label="Extracurricular Yes (0 or 1)")
            part_time_job_Yes = gr.Number(label="Part Time Job Yes (0 or 1)")
    predict_btn = gr.Button("Predict")
    output = gr.Textbox(label="Prediction", lines=5)

    # Link button to function
    predict_btn.click(
        fn=predict_result,
        inputs=[
            study_hours_per_week, sleep_hours_per_day, attendance_percentage,
            assignments_completed, participation_level_Low, participation_level_Medium,
            internet_access_Yes, parental_education_High_School,
            parental_education_Masters, parental_education_PhD,
            extracurricular_Yes, part_time_job_Yes
        ],
        outputs=output
    )

iface.launch()
