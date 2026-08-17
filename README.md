# 🚀 AI-Powered Career Intelligence Platform

An AI-based resume analysis and career recommendation system that uses **Machine Learning and Natural Language Processing** to analyze resumes, extract skills, and recommend suitable career roles.

## 📌 Overview

The AI-Powered Career Intelligence Platform helps students, fresh graduates, and job seekers identify suitable career opportunities by analyzing their resumes. The system extracts technical skills, education details, and experience from resumes and predicts relevant job roles using a trained Machine Learning model.

## ✨ Features

- 📄 Resume upload and automatic PDF text extraction
- 🔍 Skill and keyword extraction using NLP
- 🤖 AI-based career role prediction
- 📊 Confidence score generation
- 🎯 Personalized career recommendations
  
 ---
 
# 🧠 Machine Learning Performance

The career recommendation model was trained using a large-scale resume dataset and evaluated using classification metrics.

### 🚀 Logistic Regression Model Accuracy

## ⭐ Achieved Accuracy: **95.51%**

The Logistic Regression classifier demonstrated strong performance in predicting suitable career categories from resume features.

The model uses:

- TF-IDF feature extraction
- Resume text classification
- Supervised learning approach
- Probability-based prediction

---
## ⚙️ Working Process

1. User uploads a resume in PDF format.
2. Resume content is extracted using PyMuPDF.
3. NLP techniques are applied for text preprocessing and skill extraction.
4. TF-IDF converts resume text into numerical features.
5. Logistic Regression model predicts suitable career roles.
6. Recommended roles and confidence scores are displayed.

## 📂 Dataset

The Machine Learning model is trained using a dataset containing **54,000+ resumes** collected from different professional domains.

The dataset includes resumes from:

- Software Development
- Data Science
- Artificial Intelligence
- Web Development
- Database Administration
- Testing
- Networking
- Engineering

The dataset helps the model learn the relationship between candidate skills and suitable career categories.

## 🧠 Machine Learning Pipeline

```
Resume Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression Model
        ↓
Career Role Prediction
        ↓
Recommendation Output
```

## 🛠️ Tech Stack

### Frontend
- HTML,CSS,Javascript


### Backend
- Flask
- REST API

### Machine Learning
- Python
- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization

### NLP & Processing
- Scipy
- Text preprocessing techniques

## 📁 Project Structure

```
AI-Career-Intelligence-Platform/

│
├── app.py                 # Flask backend
├── frontend.py            # frontend
│
├── models/
│   ├── career_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── feature_selector.pkl
│
├── requirements.txt
└── README.md

---

# 🚀 Milestone 2 — Advanced Machine Learning & Career Ranking

In Milestone 2, the baseline career recommendation system was enhanced using advanced ensemble Machine Learning models and semantic text representations to improve prediction performance and career recommendation quality.

## 🧠 Advanced Models

The following Machine Learning models were implemented and evaluated:

- 🌲 Random Forest Classifier
- ⚡ XGBoost Classifier
- 🧠 Sentence-BERT (SBERT) embeddings
- 🔧 Hyperparameter tuning
- 🔄 Cross-validation
- 📊 Accuracy and classification metrics
- 🎯 Top-K career recommendation

## 📊 Milestone 2 Model Performance

| Model | Accuracy |
|------|----------|
| Logistic Regression | **95.51%** |
| Random Forest | **96%** |
| XGBoost | **94%** |

### 🌲 Random Forest

The Random Forest classifier was optimized using hyperparameter tuning and cross-validation to improve generalization and reduce overfitting.

**Achieved Accuracy: 96%**

### ⚡ XGBoost

XGBoost was implemented as an advanced gradient boosting classifier for career-role prediction.

**Achieved Accuracy: 94%**

## 🧠 Sentence-BERT Skill Embeddings

Sentence-BERT was integrated to capture the semantic relationship between resume content, skills, and career roles.

**Model Used:**

`sentence-transformers/all-MiniLM-L6-v2`

SBERT converts resume and skill information into dense numerical embeddings, allowing the system to understand semantic similarity beyond simple keyword matching.

## ⚙️ Milestone 2 Pipeline

```text
Resume Dataset
       ↓
Data Cleaning & Preprocessing
       ↓
Resume Text & Skill Processing
       ↓
TF-IDF / SBERT Feature Representation
       ↓
Feature Engineering
       ↓
Hyperparameter Tuning
       ↓
Random Forest / XGBoost
       ↓
Career Role Prediction
       ↓
Confidence Score
       ↓
Top-K Career Recommendations

## 🎯 Applications

- Career guidance for students
- Resume analysis for fresh graduates
- Skill-based job recommendations
- Placement assistance
- Candidate profile analysis

## 🔮 Future Enhancements

- Integration with job portals
- Generative AI career assistant
- Skill gap analysis
- Personalized learning roadmap
- Resume improvement suggestions
- Cloud deployment

## 👩‍💻 Author

**Rakshita Handage**

Computer Science Engineering Student

