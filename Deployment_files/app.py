import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Title and Description
st.title("Coursera Course Recommendation System")
st.write("Welcome to the Coursera Course Recommendation System. I'm here to make your life easy in finding your required courses.")

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("../Dataset/coursea_data.csv")

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
    relevant_indices = relevant_courses.index
    
    # Filter the feature matrix X to include only relevant courses
    X_relevant = X[relevant_indices]
    
    # Encode difficulty level
    difficulty_encoded = label_encoders['course_difficulty'].transform([difficulty])[0]
    
    # Scale numerical variables
    input_data = scaler.transform([[difficulty_encoded, rating, 0]])  # Set students enrolled to 0
    
    # Predict courses
    predictions = model.predict_proba(input_data)
    
    # Find similar courses
    similarity = cosine_similarity(X_relevant, input_data)
    top_indices = similarity.flatten().argsort()[::-1][:num_recommendations]
    
    # Collect recommended courses information in a list
    recommended_courses = []
    for index in top_indices:
        course = relevant_courses.iloc[index]
        recommended_courses.append({
            "Course Title": course['course_title'],
            "Organization": course['course_organization'],
            "Certificate Type": label_encoders['course_Certificate_type'].inverse_transform([course['course_Certificate_type']])[0],
            "Rating": course['course_rating'],
            "Students Enrolled": course['course_students_enrolled'],
            "Similarity": similarity[index][0]
        })

    return recommended_courses

# User Input and Recommendations
subject = st.text_input("Enter your interest (subject): ")
rating = st.slider("Enter desired rating (0-5): ", 0.0, 5.0, 3.0, 0.1)
difficulty = st.selectbox("Enter desired difficulty level:", ['Beginner', 'Intermediate', 'Advanced', 'Mixed'])

if st.button('Recommend'):
    recommended_courses = recommend(subject, rating, difficulty)
    if recommended_courses:
        for i, course in enumerate(recommended_courses, 1):
            st.write(f"{i}. Course Title: {course['Course Title']}")
            st.write(f"   Organization: {course['Organization']}")
            st.write(f"   Certificate Type: {course['Certificate Type']}")
            st.write(f"   Rating: {course['Rating']}")
            st.write(f"   Students Enrolled: {course['Students Enrolled']}")
            st.write(f"   Similarity: {course['Similarity']}")

