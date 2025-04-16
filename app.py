import streamlit as st
import joblib
import numpy as np
import time

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Title and description
st.title("Student Performance Predictor")

# Display a static intro GIF
st.image("https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnNlZXRpZHdqZHEyNDJ0cmE0YnJsbGdqb3h4ZDgxcWVubnM4cjFjeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/iPj5oRtJzQGxwzuCKV/giphy.gif", width=600)

st.markdown("""
This application uses the best performing linear regression model, enhanced with polynomial feature engineering and feature selection, to predict the final exam score.
The model was trained on key metrics:
- **Study Hours per Week**
- **Attendance (%)**
- **Assignment Completion (%)**
- **Midterm Score**
- **Group Project Participation**

The input data is transformed using a polynomial expansion (degree 2), standardized, and then reduced to the top 10 features before prediction.
""")

# Input fields
st.markdown("### Enter the Student Performance Metrics")
study_hours = st.number_input("Study Hours per Week", min_value=0.0, value=11.5, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=82.5, step=1.0)
assignment_completion = st.number_input("Assignment Completion (%)", min_value=0.0, max_value=100.0, value=78.4, step=1.0)
midterm_score = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=74.2, step=1.0)
group_project = st.number_input("Group Project Participation", min_value=0.0, max_value=1.0, value=0.0, step=1.0)

# Prediction Button
if st.button("Predict"):
    with st.spinner('🔄 Running prediction... Please wait'):
        # Show loading GIF during processing
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExemp3c2IwemR5Z2s5bTVtaDBhb2QzaGZwa3kyazEyMWQ3ZjJ6YmdzciZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xT5LMLcvRrCS5Nf2Lu/giphy.gif", width=300)

        # Simulate loading time (optional)
        time.sleep(2)

        input_features = np.array([[study_hours, attendance, assignment_completion, midterm_score, group_project]])
        
        try:
            poly = joblib.load("poly.pkl")
            poly_scaler = joblib.load("poly_scaler.pkl")
            selector = joblib.load("selector.pkl")
            model = joblib.load("linear_regression_(poly)_student_performance_model.pkl")
        except Exception as e:
            st.error(f"Error loading transformation objects or model: {e}")
        else:
            input_poly = poly.transform(input_features)
            input_poly_scaled = poly_scaler.transform(input_poly)
            input_poly_selected = selector.transform(input_poly_scaled)
            
            prediction = model.predict(input_poly_selected)
            predicted_score = prediction[0]

            st.success("✅ Prediction Complete!")
            st.subheader("Predicted Final Exam Score")
            st.write(f"📘 Based on the input values, the predicted final exam score is **{predicted_score:.2f}**.")

            st.markdown("""
            **Note:** This prediction is generated using a regression model that incorporates polynomial feature engineering, feature selection, and stacking/regularization techniques.
            Adjust the input values to explore different scenarios.
            """)

# Footer: Your info
st.markdown("---")
st.markdown("""
**Developed by:** Eyimofe okikiola Orimolade  
**Student ID:** 246370037  
🔗 [GitHub Profile](https://github.com/lyon-zas)
""")
