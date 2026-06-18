# ==========================================================
# AI RECOMMENDATION ENGINE
# Tech Stack Recommender using TF-IDF + Cosine Similarity
# ==========================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CREATE DATASET
# ==========================================================

data = {
    "job_role": [
        "Data_Scientist",
        "Machine_Learning_Engineer",
        "Backend_Developer",
        "Full_Stack_Developer",
        "DevOps_Engineer",
        "Cloud_Architect",
        "Systems_Administrator",
        "Database_Administrator",
        "Software_Engineer",
        "Data_Analyst"
    ],

    "Python": [1,1,1,1,0,0,0,0,1,1],
    "Java": [0,0,1,1,0,0,0,0,1,0],
    "SQL": [1,1,1,1,1,1,1,1,1,1],
    "AWS": [1,1,0,0,1,1,0,0,0,0],
    "Docker": [0,1,0,0,1,1,1,0,0,0],
    "Kubernetes": [0,0,0,0,1,1,0,0,0,0],
    "Machine_Learning": [1,1,0,0,0,0,0,0,0,1],
    "DevOps": [0,0,0,0,1,0,1,0,0,0],
    "Git": [1,1,1,1,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

print("=" * 60)
print("AI CAREER RECOMMENDATION SYSTEM")
print("=" * 60)

print("\nDataset Loaded Successfully!")
print("Total Job Roles:", len(df))

# ==========================================================
# CONVERT SKILLS TO TEXT
# ==========================================================

skills_text = df.apply(
    lambda row: " ".join(
        [col for col in df.columns[1:] if row[col] == 1]
    ),
    axis=1
)

# ==========================================================
# TF-IDF VECTORIZATION
# ==========================================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(skills_text)

print("\nTF-IDF Matrix Created Successfully!")

available_skills = [
    skill.lower()
    for skill in vectorizer.get_feature_names_out()
]

print("\nAvailable Skills:")
print(", ".join(available_skills))

# ==========================================================
# USER INPUT
# ==========================================================

print("\nEnter at least 3 skills.")

user_skills = []

for i in range(3):

    skill = input(f"Skill {i + 1}: ").strip().lower()

    user_skills.append(skill)

print("\nYour Skills:")
print(user_skills)

# ==========================================================
# VALIDATE SKILLS
# ==========================================================

valid_skills = []

for skill in user_skills:

    if skill in available_skills:
        valid_skills.append(skill)
    else:
        print(f"Warning: '{skill}' not found in dataset.")

# ==========================================================
# COLD START HANDLING
# ==========================================================

if len(valid_skills) == 0:

    print("\nNo valid skills entered.")

    print("\nTrending Roles:")

    for role in df["job_role"].head(3):
        print("-", role)

    exit()

# ==========================================================
# USER VECTOR
# ==========================================================

user_text = " ".join(valid_skills)

user_vector = vectorizer.transform([user_text])

# ==========================================================
# COSINE SIMILARITY
# ==========================================================

similarity_scores = cosine_similarity(
    user_vector,
    tfidf_matrix
)[0]

# ==========================================================
# CREATE RESULTS
# ==========================================================

results = pd.DataFrame({
    "Job Role": df["job_role"],
    "Similarity Score": similarity_scores
})

results["Match Percentage"] = (
    results["Similarity Score"] * 100
)

# ==========================================================
# SORT RESULTS
# ==========================================================

results = results.sort_values(
    by="Similarity Score",
    ascending=False
)

# ==========================================================
# DISPLAY TOP 3 RECOMMENDATIONS
# ==========================================================

print("\n" + "=" * 60)
print("TOP 3 RECOMMENDED CAREER PATHS")
print("=" * 60)

top_3 = results.head(3)

for rank, (_, row) in enumerate(
    top_3.iterrows(),
    start=1
):

    role = row["Job Role"]
    score = row["Match Percentage"]

    filled = int(score / 5)

    bar = "█" * filled + "░" * (20 - filled)

    print(f"\n{rank}. {role}")
    print(f"Match Score: {score:.2f}%")
    print(f"[{bar}]")

# ==========================================================
# DETAILED BREAKDOWN
# ==========================================================

print("\n" + "=" * 60)
print("DETAILED BREAKDOWN")
print("=" * 60)

print(
    top_3[
        ["Job Role", "Match Percentage"]
    ].to_string(index=False)
)

# ==========================================================
# SHOW ALL JOB RANKINGS
# ==========================================================

print("\n" + "=" * 60)
print("ALL JOB RANKINGS")
print("=" * 60)

for _, row in results.iterrows():

    print(
        f"{row['Job Role']:30}"
        f"{row['Match Percentage']:.2f}%"
    )

# ==========================================================
# COMPLETION MESSAGE
# ==========================================================

print("\n" + "=" * 60)
print("RECOMMENDATION PROCESS COMPLETED")
print("=" * 60)