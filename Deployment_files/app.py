import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Title and Description with Styling
st.set_page_config(page_title="Coursera Course Recommendation System", page_icon="📘")
st.title("Coursera Course Recommendation System")
st.markdown(
    """
    <style>
        body {
            background-color: #333333;
        }
        .big-font {
            font-size: 30px !important;
            font-weight: bold !important;
            color: #3366ff !important;
            text-align: center !important;
            margin-bottom: 20px !important;
        }
        .sub-header {
            font-size: 22px !important;
            font-weight: bold !important;
            color: #3366ff !important;
            margin-bottom: 10px !important;
        }
        .content-text {
            font-size: 18px !important;
            color: #333333 !important;
            margin-bottom: 10px !important;
        }
        .highlight {
            color: #3366ff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<p class='big-font'>Welcome to the Coursera Course Recommendation System</br>Let me be your guide in discovering the perfect courses to elevate your learning journey.</p>",
    unsafe_allow_html=True
)

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("./Dataset/coursea_data.csv")

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Data loading done!')

# Feature Engineering
label_encoders = {}
for column in ['course_difficulty', 'course_Certificate_type']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Convert 'course_students_enrolled' to numeric, handling 'k' and 'm' suffixes
data['course_students_enrolled'] = data['course_students_enrolled'].replace({'k': '*1e3', 'm': '*1e6'}, regex=True).map(pd.eval).astype(float)

# Train the model
X = data[['course_difficulty', 'course_rating', 'course_students_enrolled']]
y = data['course_title']
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X, y)

# Recommendation Function
def recommend(subject, rating, difficulty, num_recommendations=5):
    # Find courses related to the subject
    relevant_courses = data[data['course_title'].str.contains(subject, case=False)]
    
    if relevant_courses.empty:
        return []

    relevant_indices = relevant_courses.index
    
    # Encode difficulty level
    difficulty_encoded = label_encoders['course_difficulty'].transform([difficulty])[0]
    
    # Scale numerical variables
    input_data = scaler.transform([[difficulty_encoded, rating, 0]])  # Set students enrolled to 0
    
    # Predict relevant courses
    relevant_probabilities = model.predict_proba(input_data)[0]
    
    # Create a DataFrame with relevant courses and their probabilities
    relevant_courses['Probability'] = relevant_probabilities[relevant_indices]
    
    # Sort by probabilities and get top recommendations
    recommended_courses = relevant_courses.nlargest(num_recommendations, 'Probability')
    
    # Collect recommended courses information in a list
    recommendations = []
    for _, course in recommended_courses.iterrows():
        recommendations.append({
            "Course Title": course['course_title'],
            "Organization": course['course_organization'],
            "Certificate Type": label_encoders['course_Certificate_type'].inverse_transform([course['course_Certificate_type']])[0],
            "Rating": course['course_rating'],
            "Students Enrolled": course['course_students_enrolled'],
            "Probability": course['Probability']
        })

    return recommendations

# Main content area for user input
st.subheader("Custom Recommendation:")
st.markdown("<p class='content-text'>Please provide your preferences below:</p>", unsafe_allow_html=True)
subject = st.text_input("Enter your interest (subject): ")
rating = st.slider("Enter desired rating (0-5): ", 0.0, 5.0, 3.0, 0.1)
difficulty = st.selectbox("Enter desired difficulty level:", ['Beginner', 'Intermediate', 'Advanced', 'Mixed'])

if st.button('Recommend'):
    recommended_courses = recommend(subject, rating, difficulty)
    if recommended_courses:
        st.subheader("Recommendations:")
        for i, course in enumerate(recommended_courses, 1):
            st.write(f"{i}. Course Title: {course['Course Title']}")
            st.write(f"   Organization: {course['Organization']}")
            st.write(f"   Certificate Type: {course['Certificate Type']}")
            st.write(f"   Rating: {course['Rating']}")
            st.write(f"   Students Enrolled: {course['Students Enrolled']}")
            st.write(f"   Probability: {course['Probability']}")
    else:
        st.write("No relevant courses found based on your input.")
