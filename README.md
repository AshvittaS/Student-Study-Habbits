## Student Study Habits – Pass/Fail Prediction

Predicting student outcomes (pass/fail) from study habits, lifestyle factors, and access variables. This project includes end‑to‑end EDA, feature engineering, model training for two algorithms, and two live deployments.

### Live Apps
- Naive Bayes (Streamlit): [student-study-habbits-miv82ekk3rzvm5kpmnscih.streamlit.app](https://student-study-habbits-miv82ekk3rzvm5kpmnscih.streamlit.app/)
- Random Forest (Hugging Face Space): [Ashvitta07/Student_result_rf](https://huggingface.co/spaces/Ashvitta07/Student_result_rf)

### What’s unique in this project
- Dual model + dual deployment: a Categorical Naive Bayes app on Streamlit and a Random Forest app on Hugging Face. This showcases different modeling philosophies and hosting targets in one repo.
- Clear, interpretable probability exploration: the Streamlit app plots probability of passing against `study_hours_per_week` while holding other inputs constant, giving users actionable intuition rather than only a point prediction.
- Categorical NB with discretized numerics: continuous features are discretized with `KBinsDiscretizer` to suit NB’s assumptions while preserving signal.
- Class imbalance handled up front: oversampling performed with `RandomOverSampler` (from `imblearn`) ensures balanced learning and fair evaluation.
- Outlier capping that preserves distribution shape: whisker‑based capping replaces only extreme values with column means, keeping overall structure intact.
- Polished UI/UX: a dark theme, particle background, and Plotly charts in the Streamlit app; a clean, compact Gradio interface for the RF model.

### End‑to‑end flow
1) Data understanding and target creation
- Loaded `student_study_habits.csv` and engineered the binary target `result` as `(final_grade > 50).astype(int)`, then dropped `final_grade`.

2) Exploratory Data Analysis (EDA)
- Validated schema, types, and missingness; checked duplicates (none found).
- Assessed class balance and label distribution.
- Examined correlations among core numeric features: `study_hours_per_week`, `sleep_hours_per_day`, `attendance_percentage`, `assignments_completed`.
- Visualized distributions (KDEs) and boxplots to understand spread, skew, and potential outliers.

3) Robustness: outlier treatment and imbalance handling
- Applied an outlier capping function using IQR bounds; extreme values were replaced with column means to reduce undue leverage.
- Addressed class imbalance with `RandomOverSampler`, then proceeded with train/test splits.

4) Two modeling tracks
- Random Forest (baseline, strong performance):
  - Trained `RandomForestClassifier(class_weight='balanced')` on the oversampled data.
  - Evaluated with accuracy, classification report, confusion matrix, and ROC; obtained near‑perfect metrics on this dataset.
- Categorical Naive Bayes (interpretable probabilistic model):
  - Discretized numeric columns via `KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')`.
  - Trained `CategoricalNB` on the transformed dataset; reported accuracy and classification report, plus ROC.

5) Persistence and inference
- Saved trained artifacts: `random_forest_model.pkl`, `naive_bayes_model.pkl`, and `kbins_discretizer.pkl` for reproducible inference.
- Built lightweight inference pipelines that assemble inputs in the exact training feature order before calling `predict`/`predict_proba`.

6) Deployments
- Streamlit NB app: dark‑themed UI, particle background, sidebar inputs for all features, and a live probability‑vs‑study‑hours Plotly line chart for intuition.
- Hugging Face Space RF app: Gradio interface with organized numeric inputs, returning both class and probability.

### EDA highlights (concise)
- Label creation from `final_grade` simplifies the target to pass/fail for actionable deployment.
- Distribution plots helped verify sensible ranges (e.g., `study_hours_per_week` 0–60) and guided outlier capping.
- Correlation heatmap ensured core predictors don’t collapse into redundant signals and informed discretization choices for NB.
- Imbalance remediation (oversampling) stabilized both RF and NB metrics and confusion matrices.

### Feature set used for inference
`['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 'assignments_completed', 'participation_level_Low', 'participation_level_Medium', 'internet_access_Yes', 'parental_education_High School', "parental_education_Master's", 'parental_education_PhD', 'extracurricular_Yes', 'part_time_job_Yes']`

### Visual/UX details
- Streamlit: custom CSS for dark theme; particle background via `particles.js`; Plotly line chart for probability introspection; metric cards for concise results.
- Gradio: compact layout with grouped inputs for quick what‑if analysis on RF.

### Tech stack
- Python, pandas, numpy, seaborn, matplotlib, scikit‑learn, imbalanced‑learn, Plotly, Streamlit, Gradio.

### Screenshots

- EDA – Correlation Heatmap
  
  ![EDA Correlation](assets/eda_correlation.png)

- EDA – Distributions of Core Numeric Features
  
  ![EDA Distributions](assets/eda_distributions.png)

- ROC curve

    