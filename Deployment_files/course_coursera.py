# %%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity



# %%
# Load the data
data = pd.read_csv("../Dataset/coursea_data.csv")
data.head()



# %%
data.shape

# %%
data.info()

# %%
print("\nUnique values in 'course_title' column:")
print(data['course_title'].unique())

# %%
print("\nUnique values in 'course_students_enrolled' column:")
print(data['course_students_enrolled'].unique())

# %%
# Convert 'course_students_enrolled' to numeric, handling 'k' and 'm' suffixes
data['course_students_enrolled'] = data['course_students_enrolled'].str.replace('k', 'e3').str.replace('m', 'e6').astype(float)


# %%
print("\nUnique values in 'course_students_enrolled' column:")
print(data['course_students_enrolled'].unique())

# %%
# Check unique values in 'course_difficulty' and 'course_Certificate_type' columns
print("\nUnique values in 'course_difficulty' column:")
print(data['course_difficulty'].unique())

print("\nUnique values in 'course_Certificate_type' column:")
print(data['course_Certificate_type'].unique())

# %%
label_encoders = {}
for column in ['course_difficulty', 'course_Certificate_type']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Feature Engineering
X = data[['course_difficulty', 'course_rating', 'course_students_enrolled']]
y = data['course_title']

# Scaling numerical variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier()
model.fit(X, y)



# %%
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
    
    # Print recommended courses
    for index in top_indices:
        course = relevant_courses.iloc[index]
        print(f"Course Title: {course['course_title']}")
        print(f"Organization: {course['course_organization']}")
        print(f"Certificate Type: {label_encoders['course_Certificate_type'].inverse_transform([course['course_Certificate_type']])[0]}")
        print(f"Rating: {course['course_rating']}")
        print(f"Students Enrolled: {course['course_students_enrolled']}")
        print(f"Similarity: {similarity[index][0]}")
        print()


# Example usage
subject = input("Enter your interest (subject): ")
rating = float(input("Enter desired rating (0-5): "))
difficulty = input("Enter desired difficulty level (Beginner/Intermediate/Advanced/Mixed): ")

recommend(subject, rating, difficulty)

# %%



