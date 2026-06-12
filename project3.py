from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#  Dataset


courses = {
    "Python Programming":
        "python coding programming automation",

    "Web Development":
        "html css javascript frontend web",

    "Machine Learning":
        "python machine learning ai data",

    "Data Science":
        "python pandas numpy data analytics",

    "Cloud Computing":
        "cloud aws azure devops",

    "Cyber Security":
        "security ethical hacking network",

    "Java Programming":
        "java oop programming",

    "Android Development":
        "java kotlin android mobile app",

    "Artificial Intelligence":
        "ai machine learning neural networks"
}


# User Input


print("===== AI Recommendation System =====")

user_interest = input(
    "\nEnter your interests (comma separated): "
)

# Example:
# python, ai, machine learning

# Prepare Data


course_names = list(courses.keys())

course_descriptions = list(courses.values())

all_text = course_descriptions + [user_interest]


# TF-IDF Vectorization

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(all_text)

# Last vector = user profile

user_vector = tfidf_matrix[-1]

course_vectors = tfidf_matrix[:-1]


# Cosine Similarity


scores = cosine_similarity(
    user_vector,
    course_vectors
)

scores = scores.flatten()


# Ranking


recommendations = list(
    zip(course_names, scores)
)

recommendations.sort(
    key=lambda x: x[1],
    reverse=True
)

# Display Results


print("\nTop Recommendations:\n")

for course, score in recommendations[:5]:
    print(
        f"{course}  -->  Match Score: {score:.2f}"
    )
