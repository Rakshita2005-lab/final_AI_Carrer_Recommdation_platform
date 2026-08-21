# 🚀 Milestone 3 — Skill Gap Analysis, API & CI Integration

In Milestone 3, the Career Intelligence Platform was extended from simple career prediction to a more complete **AI-powered career guidance system**.

The system not only predicts suitable career roles but also analyzes the user's existing skills, identifies missing competencies, and provides actionable recommendations for improving their career readiness.

## 🎯 Milestone 3 Objectives

* 🔍 Analyze the skills extracted from a resume
* 📊 Identify skill gaps for the predicted career role
* 🎯 Recommend skills that the candidate should improve
* 🚀 Expose prediction and recommendation functionality through REST APIs
* 📈 Track ML models using MLflow
* 🔄 Implement automated CI using GitHub Actions
* 🖥️ Build a Streamlit-based review interface
* 📄 Generate career and skill-gap reports

---

## 🧠 Skill Gap Analysis

The **Skill Gap Analysis module** compares the candidate's existing skills with the skills expected for the predicted career role.

### ⚙️ Working Process

```text
Resume Upload
      ↓
Resume Text Extraction
      ↓
Skill Extraction
      ↓
Career Role Prediction
      ↓
Required Skills for Target Role
      ↓
Compare Existing vs Required Skills
      ↓
Identify Missing Skills
      ↓
Skill Gap Report
      ↓
Personalized Improvement Recommendations
```

### 📌 Example

If the predicted role is **Data Scientist**, the system can analyze skills such as:

**Existing Skills:**

```text
Python
SQL
Machine Learning
Pandas
NumPy
```

**Required Skills:**

```text
Python
SQL
Machine Learning
Pandas
NumPy
Deep Learning
TensorFlow
Statistics
Power BI
```

**Identified Skill Gaps:**

```text
Deep Learning
TensorFlow
Statistics
Power BI
```

The system then provides actionable suggestions to help the candidate improve these competencies.

---

## 🎯 Actionable Skill Recommendations

Instead of only showing missing skills, the platform provides improvement suggestions.

```text
Missing Skill
      ↓
Skill Importance
      ↓
Recommended Learning Area
      ↓
Improvement Action
```

### Example Recommendations

| Skill Gap     | Recommendation                                               |
| ------------- | ------------------------------------------------------------ |
| Deep Learning | Learn neural networks, CNNs and model training               |
| TensorFlow    | Practice building and training deep learning models          |
| Statistics    | Strengthen probability, distributions and hypothesis testing |
| Power BI      | Build dashboards using real-world datasets                   |

This makes the platform more useful for **career preparation and placement readiness**.

---

# 🚀 FastAPI REST API Integration

The platform was extended with REST APIs to make the Machine Learning functionality accessible to other applications.

### 🔌 API Endpoints

```text
POST /predict
```

Predicts the most suitable career role from resume information.

```text
POST /recommend
```

Generates career recommendations and confidence scores.

```text
POST /skill-gap
```

Analyzes the candidate's current skills and identifies missing skills.

```text
GET /health
```

Checks whether the API service is running successfully.

### API Workflow

```text
Client / Frontend
       ↓
FastAPI REST API
       ↓
Resume Processing
       ↓
ML Model
       ↓
Career Prediction
       ↓
Skill Gap Analysis
       ↓
Recommendation
       ↓
JSON Response
```

---

# 📊 MLflow Model Tracking

**MLflow** was integrated to manage and track Machine Learning experiments.

The system can track:

* Model name
* Model parameters
* Accuracy
* Training metrics
* Model versions
* Experiment results

### MLflow Workflow

```text
Model Training
      ↓
Experiment Tracking
      ↓
Metrics & Parameters
      ↓
MLflow
      ↓
Model Registry
      ↓
Model Version Management
```

This improves **model reproducibility, version control, and deployment management**.

---

# 🔄 GitHub Actions CI Pipeline

A Continuous Integration pipeline was implemented using **GitHub Actions**.

Whenever changes are pushed to the repository, automated checks can be performed.

### CI Pipeline

```text
Git Push
   ↓
GitHub Actions
   ↓
Install Dependencies
   ↓
Run Tests
   ↓
Train / Validate Model
   ↓
Check Accuracy
   ↓
Build Validation
   ↓
Pipeline Success
```

An **accuracy gate** can be used to ensure that a newly trained model does not fall below the required performance threshold.

This helps prevent low-performing models from being accepted into the application.

---

# 🖥️ Streamlit Review UI

A Streamlit-based interface was developed to provide an interactive way to review career predictions and skill gaps.

### UI Features

* 📄 Resume upload
* 🎯 Predicted career role
* 📊 Confidence score
* 🏆 Top career recommendations
* 🧠 Extracted skills
* ⚠️ Missing skills
* 📈 Skill-gap visualization
* 💡 Improvement recommendations
* 📄 Career/skill-gap report generation

### Streamlit Workflow

```text
Upload Resume
      ↓
Resume Analysis
      ↓
Career Prediction
      ↓
Top-K Recommendations
      ↓
Skill Extraction
      ↓
Skill Gap Analysis
      ↓
Personalized Recommendations
      ↓
Report Generation
```

---

# 📊 Complete Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | **95.51%** |
| Random Forest       | **96%**    |
| XGBoost             | **94%**    |

The Random Forest model achieved the highest reported accuracy among the implemented models at **96%**.

---

# 🧠 Complete AI Career Intelligence Pipeline

```text
                    Resume
                       ↓
             PDF Text Extraction
                       ↓
              NLP Preprocessing
                       ↓
                Skill Extraction
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
      TF-IDF                    SBERT
          ↓                         ↓
          └────────────┬────────────┘
                       ↓
                Feature Engineering
                       ↓
             ML Model Prediction
                       ↓
             Predicted Career Role
                       ↓
                Top-K Ranking
                       ↓
               Skill Gap Analysis
                       ↓
             Missing Skill Detection
                       ↓
          Personalized Recommendations
                       ↓
              FastAPI / Streamlit
                       ↓
             Career & Skill Report
```

---

# 🏆 Skills Demonstrated Through the Project

## Programming & Development

* Python
* Java
* SQL
* REST API Development
* Flask
* FastAPI
* Streamlit

## Machine Learning

* Logistic Regression
* Random Forest
* XGBoost
* Classification
* Hyperparameter Tuning
* Cross-Validation
* Model Evaluation
* Top-K Prediction

## NLP

* Text Preprocessing
* TF-IDF
* NLP-based Skill Extraction
* Named Entity Recognition
* Sentence-BERT
* Semantic Embeddings

## AI / Generative AI Concepts

* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Embeddings
* AI-based Recommendation Systems

## MLOps

* MLflow
* Model Registry
* Experiment Tracking
* GitHub Actions
* Continuous Integration
* Automated Model Validation
* Accuracy Gates

## Data & Analytics

* Data Cleaning
* Feature Engineering
* Classification Metrics
* Resume Dataset Processing
* Skill Analysis
* Career Analytics

## Frontend & Deployment

* HTML
* CSS
* JavaScript
* Streamlit
* REST API Integration

---

# 💡 What Makes the Project Different?

The platform goes beyond traditional resume classification.

### Traditional System

```text
Resume → Prediction → Career Role
```

### AI Career Intelligence Platform

```text
Resume
   ↓
Skills & Information Extraction
   ↓
Career Prediction
   ↓
Top Career Recommendations
   ↓
Skill Gap Identification
   ↓
Missing Skill Detection
   ↓
Personalized Improvement Suggestions
   ↓
Career Readiness Report
```

Therefore, the system acts not only as a **career prediction model**, but also as a **career guidance and skill development platform**.

---

# 🔮 Future Enhancements

* 🤖 Generative AI career assistant
* 💬 Conversational resume analysis
* 🌐 Integration with job portals
* 🔎 Real-time job recommendation
* 📚 Personalized learning roadmap
* 📝 AI-powered resume improvement
* ☁️ Cloud deployment
* 🎓 Course recommendations based on skill gaps
* 📈 Career readiness score
* 🔗 Integration with LinkedIn and job platforms

---

## 👩‍💻 Author

**Rakshita Handage**
