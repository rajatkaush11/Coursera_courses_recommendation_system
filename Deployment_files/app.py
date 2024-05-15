import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title and Description with Styling
st.set_page_config(page_title=" Coursera Course Recommendation System", page_icon="📘")
st.title("Coursera  Course Recommendation System ")
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
    "<p class='big-font'>Welcome to the <br> Coursera Course Recommendation System </br> &nbsp;Let me be your guide in discovering the perfect courses to elevate your learning journey. </p>",
    unsafe_allow_html=True
)

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("./Dataset/coursea_data.csv")

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Data loading done!')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['course_title'])

# Recommendation Function
def recommend(subject, rating, difficulty, num_recommendations=5):
    # TF-IDF Vectorization for user input subject
    input_tfidf = tfidf_vectorizer.transform([subject])
    
    # Combine user input with other preferences
    preferences = [f"Rating: {rating}", f"Difficulty: {difficulty}"]
    input_text = " ".join(preferences)
    input_tfidf = tfidf_vectorizer.transform([input_text])
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Get top recommendations
    top_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    recommended_courses = data.iloc[top_indices]
    
    # Calculate similarity score percentage
    similarity_scores = similarity_scores[0][top_indices] * 100
    
    return recommended_courses, similarity_scores

# Main content area for user input
st.subheader("Custom Recommendation:")
st.markdown("<p class='content-text'>Please provide your preferences below:</p>", unsafe_allow_html=True)
subject = st.text_input("Enter your interest (subject): ")
rating = st.slider("Enter desired rating (0-5): ", 0.0, 5.0, 3.0, 0.1)
difficulty = st.selectbox("Enter desired difficulty level:", ['Beginner', 'Intermediate', 'Advanced', 'Mixed'])

if st.button('Recommend'):
    recommended_courses, similarity_scores = recommend(subject, rating, difficulty)
    if not recommended_courses.empty:
        st.subheader("Recommendations:")
        for i, (index, course) in enumerate(recommended_courses.iterrows(), 1):
            st.write(f"{i}. Course Title: {course['course_title']}")
            st.write(f"   Organization: {course['course_organization']}")
            st.write(f"   Certificate Type: {course['course_Certificate_type']}")
            st.write(f"   Rating: {course['course_rating']}")
            st.write(f"   Students Enrolled: {course['course_students_enrolled']}")
            st.write(f"   Similarity Score: {similarity_scores[i-1]:.2f}%")
