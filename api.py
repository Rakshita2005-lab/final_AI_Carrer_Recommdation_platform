# api.py

import io
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pdfplumber

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
INDEX_FILE = BASE_DIR / "index.html"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI-Powered Career Intelligence Platform",
    description="Career prediction, recommendation and skill gap analysis API",
    version="1.2.0"
)


# ============================================================
# CORS
# ------------------------------------------------------------
# Opened up so the deployed frontend (hosted on Netlify/Render
# static site/etc, not just localhost) can actually call this
# API. Once you know your frontend's final URL, you can tighten
# this back down to that origin instead of "*".
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SAFE VALIDATION ERROR HANDLER
# ------------------------------------------------------------
# Prevents 500 crashes when invalid request data (e.g. raw
# binary file bytes) can't be JSON-encoded by the default
# FastAPI error formatter.
# ============================================================

def _sanitize(obj):
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes omitted>"
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize(exc.errors())},
    )


# ============================================================
# LOAD MODELS
# ============================================================

def load_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        print(f"Warning: {filename} not found")
        return None

    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


tfidf_vectorizer = load_model("tfidf_vectorizer.pkl")
feature_selector = load_model("feature_selector.pkl")
random_forest_model = load_model("random_forest_model.pkl")
career_model = load_model("career_model.pkl")
label_encoder = load_model("label_encoder.pkl")


# ============================================================
# ROLE SKILLS
# ============================================================

ROLE_SKILLS = {

    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "React", "Python",
        "SQL", "REST API", "Git"
    ],

    "Java Developer": [
        "Java", "OOP", "SQL", "Spring Boot", "Hibernate",
        "Maven", "Git", "REST API"
    ],

    "Project Manager": [
        "Project Management", "Agile", "Scrum", "Jira",
        "Communication", "Leadership", "Risk Management", "Planning"
    ],

    "Python Developer": [
        "Python", "OOP", "SQL", "Git", "REST API",
        "FastAPI", "Django", "Testing"
    ],

    "Front End Web Developer": [
        "HTML", "CSS", "JavaScript", "React", "Git",
        "Responsive Design", "REST API", "TypeScript"
    ],

    "Backend Developer": [
        "Python", "Java", "SQL", "REST API", "Git",
        "Docker", "API Development", "Database"
    ],

    "Data Scientist": [
        "Python", "SQL", "Machine Learning", "Statistics",
        "Pandas", "NumPy", "Scikit-learn", "Data Visualization"
    ],

    "Machine Learning Engineer": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow",
        "PyTorch", "SQL", "Docker", "MLOps"
    ]
}


# ============================================================
# ACTIONABLE SUGGESTIONS
# ============================================================

SKILL_SUGGESTIONS = {

    "Java": "Learn Java syntax, collections, exception handling and multithreading.",
    "OOP": "Practice inheritance, polymorphism, abstraction and encapsulation.",
    "Spring Boot": "Build REST APIs using Spring Boot.",
    "Hibernate": "Learn Hibernate/JPA for database integration.",
    "Maven": "Learn Maven dependency and project management.",
    "REST API": "Build and consume REST APIs.",
    "SQL": "Practice joins, subqueries, normalization and database design.",
    "Git": "Practice Git branching, merging and collaborative workflows.",
    "Python": "Strengthen Python programming, OOP and advanced Python concepts.",
    "FastAPI": "Build production-ready REST APIs using FastAPI.",
    "Django": "Create a complete web application using Django.",
    "Testing": "Learn pytest and unit testing.",
    "HTML": "Practice semantic HTML and accessibility.",
    "CSS": "Learn Flexbox, Grid and responsive layouts.",
    "JavaScript": "Strengthen ES6+, DOM manipulation and asynchronous JavaScript.",
    "React": "Build React applications using components and hooks.",
    "Responsive Design": "Create mobile-first responsive interfaces.",
    "TypeScript": "Learn TypeScript for scalable frontend development.",
    "Project Management": "Learn project planning, execution and monitoring.",
    "Agile": "Learn Agile principles and software development workflows.",
    "Scrum": "Understand Scrum roles, ceremonies and artifacts.",
    "Jira": "Practice sprint planning and issue tracking using Jira.",
    "Communication": "Improve communication with teams and stakeholders.",
    "Leadership": "Develop leadership and decision-making skills.",
    "Risk Management": "Learn how to identify, analyze and manage project risks.",
    "Planning": "Practice project scheduling and resource planning.",
    "Machine Learning": "Practice supervised and unsupervised machine learning projects.",
    "Deep Learning": "Learn neural networks using TensorFlow or PyTorch.",
    "TensorFlow": "Build and train deep learning models with TensorFlow.",
    "PyTorch": "Practice neural network development using PyTorch.",
    "Docker": "Learn containerization and Docker deployment.",
    "MLOps": "Learn model deployment, monitoring and ML lifecycle management."
}


# ============================================================
# REQUEST MODELS
# ============================================================

class ResumeRequest(BaseModel):
    text: str


class RecommendationRequest(BaseModel):
    text: str
    top_k: int = 5


class GapReportRequest(BaseModel):
    job_role: str
    skills: List[str]


# ============================================================
# SKILL EXTRACTION
# ============================================================

SKILL_KEYWORDS = set()

for skills in ROLE_SKILLS.values():
    SKILL_KEYWORDS.update(skills)


def extract_skills(text: str):

    text_lower = text.lower()

    detected_skills = []

    for skill in SKILL_KEYWORDS:

        if skill.lower() in text_lower:

            detected_skills.append(skill)

    return sorted(
        detected_skills,
        key=str.lower
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts plain text from PDF bytes using pdfplumber.
    Raises ValueError if the file cannot be parsed.
    """

    text_parts = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}")

    return "\n".join(text_parts).strip()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(text):

    if tfidf_vectorizer is None:
        raise ValueError("TF-IDF vectorizer not found.")

    # TF-IDF
    X = tfidf_vectorizer.transform([text])

    print("TF-IDF features:", X.shape[1])

    # Feature selector
    if feature_selector is not None:
        X = feature_selector.transform(X)

    print("After feature selector:", X.shape[1])

    # Make API input match the trained model
    if random_forest_model is not None:

        expected = random_forest_model.n_features_in_
        actual = X.shape[1]

        print("Model expects:", expected)
        print("Features received:", actual)

        if actual > expected:
            X = X[:, :expected]

        elif actual < expected:
            from scipy.sparse import hstack

            missing = expected - actual

            padding = np.zeros(
                (X.shape[0], missing)
            )

            X = hstack([X, padding])

        print("Final features sent to model:", X.shape[1])

    return X


# ============================================================
# MODEL PREDICTION
# ============================================================

def generate_recommendations(text, top_k=5):

    # --------------------------------------------------------
    # Try Random Forest first
    # --------------------------------------------------------

    model = random_forest_model

    if model is None:
        model = career_model

    if model is None:
        raise ValueError(
            "No trained career model was found."
        )

    X = prepare_features(text)

    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        classes = model.classes_

        results = []

        for index, probability in enumerate(probabilities):

            class_value = classes[index]

            try:

                if label_encoder is not None:
                    role = label_encoder.inverse_transform(
                        [class_value]
                    )[0]
                else:
                    role = str(class_value)

            except Exception:

                role = str(class_value)

            results.append(
                {
                    "role": role,
                    "probability": round(
                        float(probability * 100),
                        2
                    )
                }
            )

        results.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        return results[:top_k]

    # --------------------------------------------------------
    # Fallback prediction
    # --------------------------------------------------------

    prediction = model.predict(X)[0]

    try:

        if label_encoder is not None:
            role = label_encoder.inverse_transform(
                [prediction]
            )[0]
        else:
            role = str(prediction)

    except Exception:

        role = str(prediction)

    return [
        {
            "role": role,
            "probability": 100.0
        }
    ]


# ============================================================
# SUMMARY TEXT BUILDER
# ------------------------------------------------------------
# This backend doesn't extract "profile summary" or "education"
# entities from the resume text, so we generate a short,
# honest summary from what we DO have (skills + top prediction)
# instead of fabricating fields the model never produced.
# ============================================================

def build_profile_summary(skills, recommendations):

    if not skills:
        return "Not available."

    top_role = recommendations[0]["role"] if recommendations else None

    skill_preview = ", ".join(skills[:6])
    more = f" and {len(skills) - 6} more" if len(skills) > 6 else ""

    if top_role:
        return (
            f"Detected {len(skills)} relevant skills ({skill_preview}{more}), "
            f"most closely aligned with {top_role}."
        )

    return f"Detected {len(skills)} relevant skills ({skill_preview}{more})."


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        return {
            "message": "index.html not found"
        }

    return FileResponse(INDEX_FILE)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": random_forest_model is not None
    }


# ============================================================
# EXTRACT SKILLS (from raw text - JSON)
# ============================================================

@app.post("/extract-skills")
def extract_resume_skills(request: ResumeRequest):

    skills = extract_skills(request.text)

    return {
        "skills": skills
    }


# ============================================================
# EXTRACT SKILLS FROM PDF (file upload)
# ============================================================

@app.post("/extract-skills-from-file")
async def extract_resume_skills_from_file(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_bytes = await file.read()

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded PDF."
        )

    skills = extract_skills(resume_text)

    return {
        "skills": skills,
        "extracted_text_length": len(resume_text)
    }


# ============================================================
# PREDICT (PDF upload -> full result the frontend renders)
# ------------------------------------------------------------
# This is the endpoint the CareerCast frontend calls with
# multipart/form-data field "resume". It returns:
#   skills            -> list[str]
#   recommendations   -> list[str]  (top 5 role names, ranked)
#   confidence        -> list[float] (matching % per role, same order)
#   summary           -> str  (short profile summary)
#   education         -> str  ("Not available." - no extractor for this)
# ============================================================

@app.post("/predict")
async def predict(resume: UploadFile = File(...)):

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_bytes = await resume.read()

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded PDF."
        )

    try:
        ranked = generate_recommendations(resume_text, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    skills = extract_skills(resume_text)

    recommendations = [r["role"] for r in ranked]
    confidence = [r["probability"] for r in ranked]

    return {
        "skills": skills,
        "recommendations": recommendations,
        "confidence": confidence,
        "summary": build_profile_summary(skills, ranked),
        "education": "Not available."
    }


# ============================================================
# RECOMMEND (from raw text - JSON)
# ============================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    try:

        recommendations = generate_recommendations(
            request.text,
            top_k=request.top_k
        )

        skills = extract_skills(
            request.text
        )

        return {
            "recommendations": recommendations,
            "skills": skills
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# RECOMMEND FROM FILE (PDF upload)
# ============================================================

@app.post("/recommend-from-file")
async def recommend_from_file(file: UploadFile = File(...), top_k: int = 5):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_bytes = await file.read()

    try:
        resume_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract any text from the uploaded PDF."
        )

    try:

        recommendations = generate_recommendations(
            resume_text,
            top_k=top_k
        )

        skills = extract_skills(resume_text)

        return {
            "recommendations": recommendations,
            "skills": skills
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# SKILL GAP REPORT
# ============================================================

@app.post("/gap-report")
def gap_report(request: GapReportRequest):

    role = request.job_role

    required_skills = ROLE_SKILLS.get(role)

    if required_skills is None:

        raise HTTPException(
            status_code=404,
            detail=f"No skill mapping found for {role}"
        )

    # Normalize user skills

    user_skills_normalized = {
        skill.strip().lower()
        for skill in request.skills
    }

    matched_skills = []
    missing_skills = []

    for required_skill in required_skills:

        if required_skill.lower() in user_skills_normalized:

            matched_skills.append(
                required_skill
            )

        else:

            missing_skills.append(
                required_skill
            )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    total = len(required_skills)

    alignment = (
        len(matched_skills) / total * 100
        if total > 0
        else 0
    )

    alignment = round(
        alignment,
        2
    )

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    suggestions = []

    for skill in missing_skills:

        suggestion = SKILL_SUGGESTIONS.get(
            skill,
            f"Improve your knowledge of {skill}."
        )

        suggestions.append(
            {
                "skill": skill,
                "suggestion": suggestion
            }
        )

    return {
        "job_role": role,
        "your_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_alignment": alignment,
        "suggestions": suggestions
    }