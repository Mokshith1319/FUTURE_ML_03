import nbformat as nbf
import os

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Define the cells
cells = []

# Cell 1
cells.append(nbf.v4.new_markdown_cell("""\
# Resume / Candidate Screening System

**Goal:** Build an ML system to automatically screen and rank resumes based on a given job role.

**Key Features:**
- Resume text cleaning & parsing using `spaCy` and `NLTK`.
- Skill extraction & matching with job descriptions.
- Candidate ranking based on role fit.
- Skill gap identification.

*Note: This notebook uses realistic synthetic resumes to demonstrate the screening pipeline.*
"""))

# Cell 2
cells.append(nbf.v4.new_code_cell("""\
# Import necessary libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

# Set visualization style
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
"""))

# Cell 3
cells.append(nbf.v4.new_code_cell("""\
# Download necessary NLTK data and spaCy models
print("Setting up NLTK and spaCy...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

try:
    nlp = spacy.load('en_core_web_sm')
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')
    print("Download complete.")
"""))

# Cell 4
cells.append(nbf.v4.new_markdown_cell("""\
## 1. Creating a Synthetic Resume Dataset
To simulate a real-world scenario, we will generate realistic synthetic resumes for several candidates across 6 different job roles:
1. Data Scientist
2. Software Engineer
3. Product Manager
4. Marketing Specialist
5. UI/UX Designer
6. HR Manager
"""))

# Cell 5
cells.append(nbf.v4.new_code_cell("""\
# Synthetic dataset of candidates
resume_data = [
    {
        "CandidateID": "C001",
        "Name": "Alice Smith",
        "Role": "Data Scientist",
        "Resume_Text": "Experienced Data Scientist with 4 years of experience in machine learning and predictive modeling. Skilled in Python, R, SQL, and TensorFlow. Built NLP pipelines using spaCy and NLTK. Strong background in statistics and data visualization using Tableau."
    },
    {
        "CandidateID": "C002",
        "Name": "Bob Johnson",
        "Role": "Software Engineer",
        "Resume_Text": "Full-stack Software Engineer proficient in Java, Spring Boot, and React.js. Experienced in building scalable microservices and working with Docker and Kubernetes. Managed CI/CD pipelines and AWS cloud infrastructure."
    },
    {
        "CandidateID": "C003",
        "Name": "Charlie Brown",
        "Role": "Data Scientist",
        "Resume_Text": "Junior Data Analyst transitioning to Data Science. Good knowledge of Python, Pandas, and Scikit-learn. Familiar with building linear regression and classification models. Learning Deep Learning and PyTorch. Experience with SQL and Excel."
    },
    {
        "CandidateID": "C004",
        "Name": "Diana Prince",
        "Role": "Product Manager",
        "Resume_Text": "Strategic Product Manager with a track record of leading cross-functional teams. Expertise in Agile methodologies, Scrum, and Jira. Strong focus on user experience and market research to drive product growth."
    },
    {
        "CandidateID": "C005",
        "Name": "Evan Wright",
        "Role": "Data Scientist",
        "Resume_Text": "Senior Machine Learning Engineer specializing in Computer Vision and Deep Learning. Proficient in Python, C++, PyTorch, and Keras. Deployed scalable ML models using Flask and Docker. Published papers on neural networks."
    },
    {
        "CandidateID": "C006",
        "Name": "Fiona Gallagher",
        "Role": "Marketing Specialist",
        "Resume_Text": "Creative Digital Marketing Specialist with expertise in SEO, SEM, and content strategy. Proficient in Google Analytics, HubSpot, and social media marketing campaigns. Excellent copywriting and communication skills."
    },
    {
        "CandidateID": "C007",
        "Name": "George Miller",
        "Role": "UI/UX Designer",
        "Resume_Text": "Passionate UI/UX Designer creating intuitive and visually appealing user interfaces. Proficient in Figma, Adobe XD, and Sketch. Strong understanding of user-centered design, wireframing, and prototyping."
    },
    {
        "CandidateID": "C008",
        "Name": "Hannah Abbott",
        "Role": "HR Manager",
        "Resume_Text": "Dedicated HR Professional with 6 years of experience in talent acquisition, employee relations, and performance management. Skilled in conducting interviews, screening candidates, and using HRIS systems."
    },
    {
        "CandidateID": "C009",
        "Name": "Ian Malcolm",
        "Role": "Data Scientist",
        "Resume_Text": "Data science enthusiast with knowledge of mathematics and probability. Experienced in Python, Machine Learning, Scikit-learn, and Data Analysis. Familiar with Big Data tools like Hadoop and Spark."
    }
]

# Convert to DataFrame
df_resumes = pd.DataFrame(resume_data)
df_resumes.head()
"""))

# Cell 6
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Defining the Target Job Description
We will screen the candidates for a **Data Scientist** role. Let's define the target skills required for this job.
"""))

# Cell 7
cells.append(nbf.v4.new_code_cell("""\
# Define the target job description and required skills
job_title = "Data Scientist"

job_description = \"\"\"
We are looking for a Data Scientist to analyze large amounts of raw information to find patterns. 
We will rely on you to build data products to extract valuable business insights. 
Required skills: Python, Machine Learning, Deep Learning, SQL, NLP, TensorFlow, PyTorch, Statistics, Data Visualization.
\"\"\"

# Extracting required skills as a standard set (lowercase for easier matching)
required_skills = {
    'python', 'machine learning', 'deep learning', 'sql', 'nlp', 
    'tensorflow', 'pytorch', 'statistics', 'data visualization', 
    'scikit-learn', 'pandas', 'predictive modeling'
}

print(f"Target Role: {job_title}")
print(f"Required Skills: {', '.join(required_skills)}")
"""))

# Cell 8
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Text Preprocessing (Cleaning & Parsing)
We need to clean the text data before analysis. This involves:
- Lowercasing the text
- Removing punctuation and special characters
- Removing standard English stopwords (e.g., 'and', 'the', 'is')
- Lemmatization (converting words to their base form) using spaCy
"""))

# Cell 9
cells.append(nbf.v4.new_code_cell("""\
# Get standard English stopwords
stop_words = set(stopwords.words('english'))

def clean_and_parse_text(text):
    \"\"\"
    Cleans and parses the input text by removing noise, stopwords, and lemmatizing.
    \"\"\"
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters using Regex
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    
    # 3. Process text with spaCy for tokenization and lemmatization
    doc = nlp(text)
    
    cleaned_tokens = []
    for token in doc:
        # Keep only alphabetic tokens that are not stopwords
        if token.text not in stop_words and len(token.text) > 1:
            # Lemmatize (e.g., 'managing' -> 'manage')
            cleaned_tokens.append(token.lemma_)
            
    # Rejoin tokens into a single string
    return " ".join(cleaned_tokens)

# Apply the cleaning function to the resumes
df_resumes['Cleaned_Resume'] = df_resumes['Resume_Text'].apply(clean_and_parse_text)

# Also clean the job description (for context, though we use the skill set directly)
cleaned_jd = clean_and_parse_text(job_description)

print("Sample Cleaned Resume (Before vs After):")
print("-" * 50)
print("Before:\\n", df_resumes['Resume_Text'].iloc[0])
print("\\nAfter:\\n", df_resumes['Cleaned_Resume'].iloc[0])
"""))

# Cell 10
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Skill Extraction & Candidate Scoring
Now we will extract skills from each cleaned resume and compare them against our `required_skills` set.

We will calculate a **Fit Score**:
`Fit Score (%) = (Number of matched skills / Total required skills) * 100`
"""))

# Cell 11
cells.append(nbf.v4.new_code_cell("""\
def extract_and_score_skills(cleaned_text, required_skills):
    \"\"\"
    Extracts matching skills from the resume and calculates the fit score and skill gap.
    \"\"\"
    matched_skills = set()
    
    # Check for multi-word and single-word skills
    for skill in required_skills:
        # We check if the skill exists as a substring in the cleaned text
        # (Note: For exact boundaries, regex boundaries \\b can be used, but simple `in` works fine here)
        if skill in cleaned_text:
            matched_skills.add(skill)
            
    # Calculate skill gaps (required skills not found in resume)
    skill_gaps = required_skills - matched_skills
    
    # Calculate fit score (percentage)
    fit_score = (len(matched_skills) / len(required_skills)) * 100
    
    return list(matched_skills), list(skill_gaps), round(fit_score, 2)

# Apply function to dataframe to create new columns
df_resumes[['Matched_Skills', 'Skill_Gaps', 'Fit_Score']] = df_resumes['Cleaned_Resume'].apply(
    lambda x: pd.Series(extract_and_score_skills(x, required_skills))
)

# Display the top candidates based on Fit Score
df_ranked = df_resumes.sort_values(by='Fit_Score', ascending=False).reset_index(drop=True)
df_ranked[['Name', 'Role', 'Fit_Score', 'Matched_Skills', 'Skill_Gaps']].head()
"""))

# Cell 12
cells.append(nbf.v4.new_markdown_cell("""\
## 5. Visualizing the Results

Let's visualize the ranking of the candidates and their skill coverage. We will:
1. Plot a Bar Chart for Candidate Rankings.
2. Plot a Heatmap showing which candidate possesses which required skill.
"""))

# Cell 13
cells.append(nbf.v4.new_code_cell("""\
# 1. Candidate Ranking Bar Chart
plt.figure(figsize=(10, 6))
# We use a horizontal bar plot for better readability
ax = sns.barplot(x='Fit_Score', y='Name', data=df_ranked, palette='viridis')

plt.title(f'Candidate Fit Scores for {job_title} Role', fontsize=16, fontweight='bold')
plt.xlabel('Fit Score (%)', fontsize=12)
plt.ylabel('Candidate Name', fontsize=12)
plt.xlim(0, 100)

# Add score labels on the bars
for index, value in enumerate(df_ranked['Fit_Score']):
    plt.text(value + 1, index, f'{value}%', va='center', fontsize=10)

plt.tight_layout()
plt.show()
"""))

# Cell 14
cells.append(nbf.v4.new_code_cell("""\
# 2. Skill Match Heatmap
# Create a matrix of Candidates vs Required Skills (1 = Has skill, 0 = Missing skill)
skill_matrix = []

for index, row in df_ranked.iterrows():
    candidate_skills = []
    for skill in required_skills:
        if skill in row['Matched_Skills']:
            candidate_skills.append(1) 
        else:
            candidate_skills.append(0) 
    skill_matrix.append(candidate_skills)

# Convert to DataFrame for seaborn
df_skill_matrix = pd.DataFrame(skill_matrix, index=df_ranked['Name'], columns=list(required_skills))

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_skill_matrix, annot=False, cmap='Blues', cbar=False, linewidths=.5, linecolor='lightgray')

plt.title('Skill Match Heatmap by Candidate', fontsize=16, fontweight='bold')
plt.xlabel('Required Skills', fontsize=12)
plt.ylabel('Candidate Name', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
"""))

# Cell 15
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Conclusion and Top Candidate Summary
Finally, let's print a clear summary of the top 5 candidates.
"""))

# Cell 16
cells.append(nbf.v4.new_code_cell("""\
# Display the Top 5 candidates clearly
print("=== TOP 5 CANDIDATES FOR DATA SCIENTIST ===")
top_5 = df_ranked.head(5)

for index, row in top_5.iterrows():
    print(f"\\nRank {index + 1}: {row['Name']} ({row['Role']})")
    print(f"Fit Score: {row['Fit_Score']}%")
    print(f"Key Strengths: {', '.join(row['Matched_Skills']) if row['Matched_Skills'] else 'None'}")
    print(f"Skill Gaps: {', '.join(row['Skill_Gaps']) if row['Skill_Gaps'] else 'None'}")
"""))

nb['cells'] = cells

# Save the notebook
with open('Resume_Candidate_Screening_System.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully: Resume_Candidate_Screening_System.ipynb")
