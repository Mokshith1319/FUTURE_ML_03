# Resume / Candidate Screening System - FUTURE_ML_03

This was my third task and honestly the most relatable one so far
because it directly relates to how companies actually hire people.
The goal was to build a system that automatically screens resumes
and ranks candidates based on how well they fit a given job role.

## What I did

Started by cleaning and parsing raw resume text. Removed all the
unnecessary stuff like special characters extra spaces and common
stopwords. Used spaCy for better text processing compared to task 2.

Then extracted skills from each resume and matched them against the
job description to calculate a fit score. Ranked all candidates based
on that score and also highlighted which skills were missing for each
candidate.

## What I used

- Python and Jupyter Notebook
- spaCy and NLTK for text processing
- Scikit-learn for feature extraction
- Pandas for ranking and scoring logic
- Matplotlib for visualization

## What I learned

- How to extract meaningful info from unstructured text
- Skill matching is trickier than it looks
- Ranking logic needs to be fair and explainable
- Realized how much resume parsing matters in real HR tools

## Dataset

Used a publicly available resume dataset with various job roles.

## Result

The system successfully ranked candidates and flagged missing skills
for each profile. Really made me think about how much of recruitment
can actually be automated with the right approach.
