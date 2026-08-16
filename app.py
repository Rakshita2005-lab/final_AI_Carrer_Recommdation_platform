from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import joblib
import numpy as np
import re
import os
import json
import traceback

# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)
CORS(app)

# ==========================================================
# Load Trained Models
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "career_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
feature_selector = joblib.load(os.path.join(MODEL_DIR, "feature_selector.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Extra artifact used only for the "Random Forest" comparison bar.
# If you don't have this file, comment the two lines out — the app
# still works, it just won't be able to re-score with RF specifically.
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
random_forest_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None

print("✅ Models Loaded Successfully")

# ==========================================================
# SBERT + career_embeddings.npy  (t-SNE panel)
# ==========================================================
# ⚠️ ASSUMPTION: career_embeddings.npy holds one SBERT vector per row of
# your training resume dataset, generated with 'all-MiniLM-L6-v2'.
# If you used a different sentence-transformers model to build that
# file, change SBERT_MODEL_NAME to match — otherwise the résumé's point
# will land in the wrong part of the embedding space.
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

_sbert_model = None  # lazy-loaded, see get_sbert()
EMBED_BACKEND = None  # "sentence_transformers" or "fastembed", set on first use

# Try sentence-transformers (torch-based) first, but don't let a failure
# crash the whole Flask app. On Windows, torch can fail with a DLL load
# error (WinError 1114) even when the package installs cleanly. If that
# happens, we fall back to `fastembed` — same MiniLM model family, but
# runs on ONNX Runtime instead of torch, so it sidesteps the DLL issue
# entirely. Install it with: pip install fastembed
try:
    import sentence_transformers  # noqa: F401
    EMBED_BACKEND = "sentence_transformers"
    print("✅ sentence-transformers import OK (using torch backend)")
except Exception as e:
    print(f"⚠️  sentence-transformers failed to import ({type(e).__name__}: {e})")
    try:
        import fastembed  # noqa: F401
        EMBED_BACKEND = "fastembed"
        print("✅ Falling back to fastembed (ONNX Runtime, no torch needed)")
    except Exception as e2:
        print(f"⚠️  fastembed also unavailable ({type(e2).__name__}: {e2})")
        print("⚠️  t-SNE panel will stay empty. Fix torch OR run: pip install fastembed")


def get_sbert():
    """Lazy-load an embedding model so the Flask server still boots even
    if the embedding backend isn't available yet. Returns an object with
    an .encode(list_of_strings) -> np.ndarray method either way, so
    calling code doesn't need to know which backend is active."""
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model

    if EMBED_BACKEND == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    elif EMBED_BACKEND == "fastembed":
        from fastembed import TextEmbedding

        # ⚠️ Model-name mapping: fastembed's naming differs from
        # sentence-transformers'. "BAAI/bge-small-en-v1.5" is fastembed's
        # closest widely-available equivalent to all-MiniLM-L6-v2 (also
        # 384-dim), but it is NOT numerically identical — points may sit
        # slightly differently in the t-SNE plot than a true MiniLM
        # embedding would. Good enough for visualization; if you need
        # exact parity, fix torch instead (see chat) so the same
        # all-MiniLM-L6-v2 model is used everywhere.
        fe_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        class _FastEmbedWrapper:
            def encode(self, texts):
                return np.array(list(fe_model.embed(texts)))

        _sbert_model = _FastEmbedWrapper()

    else:
        raise RuntimeError(
            "No embedding backend available — install sentence-transformers "
            "(and fix the torch DLL issue) or run: pip install fastembed"
        )

    return _sbert_model


EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "career_embeddings.npy")
if os.path.exists(EMBEDDINGS_PATH):
    career_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    # FIX: force a clean float64 2D array. If this .npy was ever saved from
    # a pandas column, a list-of-lists with ragged rows, or with
    # allow_pickle=True data, it can load as dtype=object. np.vstack()-ing
    # an object array together with a normal float array either throws or
    # produces garbage, which silently killed the t-SNE step before.
    try:
        career_embeddings = np.asarray(career_embeddings, dtype=np.float64)
    except Exception as cast_err:
        print(f"⚠️  Could not cast career_embeddings.npy to float64: {cast_err}")
    print(f"✅ career_embeddings.npy loaded — shape {career_embeddings.shape}, dtype {career_embeddings.dtype}")
else:
    career_embeddings = None
    print(f"⚠️  career_embeddings.npy NOT FOUND at {EMBEDDINGS_PATH} — t-SNE panel will stay empty.")

# Bucket job titles into broad categories for the t-SNE legend/colors.
# Used both as a fallback when the CSV has no category column, and to
# fill in categories per-row when the CSV only has a title column (e.g.
# a single "career" column with no category). Extend this dict to match
# your label_encoder classes / any new job titles you add.
# FIX: moved this above the CSV-loading block below, since that block
# now calls categorize() — it previously sat after that block and would
# have raised NameError: name 'categorize' is not defined at startup.
CATEGORY_MAP = {
    "software engineer": "Software Engineering",
    "software developer": "Software Engineering",
    "backend developer": "Software Engineering",
    "backend engineer": "Software Engineering",
    "frontend developer": "Software Engineering",
    "front end developer": "Software Engineering",
    "full stack developer": "Software Engineering",
    "full stack engineer": "Software Engineering",
    "web developer": "Software Engineering",
    "mobile developer": "Software Engineering",
    "java developer": "Software Engineering",
    "python developer": "Software Engineering",
    "devops engineer": "Cloud & DevOps",
    "cloud engineer": "Cloud & DevOps",
    "site reliability engineer": "Cloud & DevOps",
    "systems engineer": "Cloud & DevOps",
    "it security analyst": "Cloud & DevOps",
    "network administrator": "Cloud & DevOps",
    "data scientist": "Data Science",
    "data analyst": "Data Science",
    "data engineer": "Data Science",
    "database administrator": "Data Science",
    "ml engineer": "AI & ML",
    "machine learning engineer": "AI & ML",
    "ai engineer": "AI & ML",
    "research scientist": "Research",
    "research assistant": "Research",
    "research intern": "Research",
    "teaching assistant": "Research",
    "product manager": "Product Management",
    "program manager": "Product Management",
    "project manager": "Product Management",
    "it project manager": "Product Management",
    "business analyst": "Finance",
    "business intelligence analyst": "Finance",
    "quantitative analyst": "Finance",
    "consultant": "Finance",
}


def categorize(job_title):
    return CATEGORY_MAP.get(str(job_title).strip().lower(), "Other")


# ⚠️ ASSUMPTION: alongside career_embeddings.npy there is a CSV with the
# SAME ROW ORDER giving each point's job title + broad category (the
# legend in your screenshot: Software Engineering, Data Science,
# Product Management, AI & ML, Research, Finance). Expected path:
#   models/career_embeddings_meta.csv   with columns: job_title,category
# If this file doesn't exist yet, generate it once during training
# (zip your dataset's job-title column with a category mapping) — the
# app still runs without it, it'll just label every dataset point
# generically as "Dataset" instead of by category.
EMB_META_PATH = os.path.join(MODEL_DIR, "career_embeddings_meta.csv")
emb_meta = None
if os.path.exists(EMB_META_PATH):
    import pandas as pd
    emb_meta = pd.read_csv(EMB_META_PATH)

    # FIX: be tolerant of column-name variations instead of hard-crashing
    # inside compute_tsne() with a KeyError the first time a resume is
    # analyzed. Normalize whatever's there to exactly "job_title" and
    # "category".
    def _find_col(df, candidates):
        lower_map = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    title_col = _find_col(emb_meta, ["job_title", "title", "role", "job title", "position", "career"])
    category_col = _find_col(emb_meta, ["category", "job_category", "field", "domain"])

    rename_map = {}
    if title_col and title_col != "job_title":
        rename_map[title_col] = "job_title"
    if category_col and category_col != "category":
        rename_map[category_col] = "category"
    if rename_map:
        emb_meta = emb_meta.rename(columns=rename_map)
        print(f"ℹ️  career_embeddings_meta.csv columns renamed for compatibility: {rename_map}")

    if "job_title" not in emb_meta.columns:
        emb_meta["job_title"] = ""
        print("⚠️  No job title column found in career_embeddings_meta.csv — labels will be blank.")
    if "category" not in emb_meta.columns:
        # FIX: previously this set every row to the literal string
        # "Dataset", which is why every point in the t-SNE plot rendered
        # in one generic gray color regardless of job title. Look each
        # title up in CATEGORY_MAP instead so points get grouped/colored
        # by real category (Software Engineering, Data Science, etc.).
        emb_meta["category"] = emb_meta["job_title"].apply(categorize)
        print("ℹ️  No category column found in career_embeddings_meta.csv — categories derived from CATEGORY_MAP based on job_title.")

    print(f"✅ career_embeddings_meta.csv loaded — {len(emb_meta)} rows")
else:
    print(f"⚠️  career_embeddings_meta.csv NOT FOUND at {EMB_META_PATH} — dataset points will show as generic 'Dataset'.")

MAX_TSNE_POINTS = 300  # cap dataset points for speed on a free-tier server


def compute_tsne(resume_text):
    """Embed the résumé with SBERT, drop it into the same space as the
    training embeddings, and run t-SNE so the frontend can scatter-plot
    it against the dataset. Returns [] if the embeddings file / SBERT
    model aren't available so the rest of the response still works."""
    if career_embeddings is None:
        print("⚠️  compute_tsne skipped — career_embeddings is None (file missing or failed to load).")
        return []

    if career_embeddings.ndim != 2 or career_embeddings.shape[0] < 2:
        print(f"⚠️  compute_tsne skipped — career_embeddings has unusable shape {career_embeddings.shape}.")
        return []

    from sklearn.manifold import TSNE

    sbert = get_sbert()
    resume_vec = np.asarray(sbert.encode([resume_text]), dtype=np.float64)

    # FIX: guard against an embedding-dimension mismatch (e.g. SBERT model
    # producing 768-dim vectors while career_embeddings.npy is 384-dim) —
    # this used to blow up inside np.vstack with an unhelpful shape error.
    if resume_vec.shape[1] != career_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: résumé vector is {resume_vec.shape[1]}-dim "
            f"but career_embeddings.npy is {career_embeddings.shape[1]}-dim. "
            f"Make sure SBERT_MODEL_NAME matches the model used to build career_embeddings.npy."
        )

    base = career_embeddings
    meta = emb_meta

    if len(base) > MAX_TSNE_POINTS:
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(len(base), MAX_TSNE_POINTS, replace=False)
        base = base[idx_sample]
        if meta is not None:
            meta = meta.iloc[idx_sample].reset_index(drop=True)

    combined = np.vstack([base, resume_vec])

    # FIX: t-SNE requires perplexity < n_samples. With very small datasets
    # (e.g. 13 rows) the old formula could still land too close to the
    # edge on some sklearn versions. Clamp with a safety margin.
    n_samples = len(combined)
    perplexity = min(30, max(2, min(5, n_samples - 1)))
    if n_samples <= 3:
        print(f"⚠️  compute_tsne skipped — only {n_samples} total points, not enough for a stable t-SNE layout.")
        return []

    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
    coords = tsne.fit_transform(combined)

    points = []
    for i in range(len(base)):
        if meta is not None:
            job_title = str(meta.iloc[i]["job_title"])
            category = str(meta.iloc[i]["category"])
        else:
            job_title = ""
            category = "Dataset"
        points.append({
            "x": float(coords[i][0]),
            "y": float(coords[i][1]),
            "category": category,
            "label": job_title,
            "is_resume": False,
        })

    points.append({
        "x": float(coords[-1][0]),
        "y": float(coords[-1][1]),
        "category": "Your Resume",
        "label": "Your Resume",
        "is_resume": True,
    })
    return points

# ==========================================================
# Model comparison metrics  (bar chart panel)
# ==========================================================
# ⚠️ ASSUMPTION: these should come from your training notebook's
# evaluation step. Save them either as a flat mapping:
#   {"Logistic Regression": 0.61, "Random Forest": 0.72, "XGBoost": 0.86}
# or as a nested per-model mapping (what your current model_metrics.json
# actually looks like):
#   {"Random Forest": {"training_accuracy": 0.96, "testing_accuracy": 0.92,
#                       "balanced_accuracy": 0.86}, ...}
# Either shape works now — see the flattening step below.
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# FIX: this is the actual bug you hit. The frontend's bar chart does
# Object.values(modelComparison) and expects each value to be a single
# number. Your model_metrics.json changed to store a dict of {training_
# accuracy, testing_accuracy, balanced_accuracy} per model, so Chart.js
# was being handed objects instead of numbers and silently drew nothing.
# We flatten each model down to ONE representative score here, preferring
# testing_accuracy (or balanced_accuracy as a fallback) since that's the
# most meaningful "how good is this model on unseen résumés" number.
# PREFERRED_METRIC_KEYS = ["testing_accuracy", "balanced_accuracy", "training_accuracy", "f1", "macro_f1", "accuracy"]
PREFERRED_METRIC_KEYS = [
    "macro_f1",
    "f1",
    "testing_accuracy"
]


def _flatten_metrics(raw):
    flat = {}
    for name, val in raw.items():
        if isinstance(val, dict):
            chosen = None
            for key in PREFERRED_METRIC_KEYS:
                if key in val:
                    chosen = val[key]
                    break
            if chosen is None and val:
                # last resort: just take the first numeric value present
                chosen = next(iter(val.values()))
            flat[name] = round(float(chosen), 4) if chosen is not None else 0.0
        else:
            flat[name] = round(float(val), 4)
    return flat


if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        _raw_metrics = json.load(f)
    MODEL_METRICS = _flatten_metrics(_raw_metrics)
    print(f"✅ model_metrics.json loaded and flattened for charting: {MODEL_METRICS}")
else:
    MODEL_METRICS = {
        "Logistic Regression": 0.61,
        "Random Forest": 0.72,
        "XGBoost": 0.86,
    }
    print(f"⚠️  model_metrics.json NOT FOUND at {METRICS_PATH} — using fallback values: {MODEL_METRICS}")

# ==========================================================
# Skills Dictionary
# ==========================================================

SKILLS = [
    # Languages
    "python", "java", "c", "c++", "sql", "r",
    # Web Technologies
    "html", "css", "javascript", "typescript",
    # Frontend
    "react", "angular", "vue",
    # Backend
    "nodejs", "express", "flask", "django", "fastapi",
    "spring", "spring boot", ".net", "junit",
    # Databases
    "mysql", "postgresql", "oracle", "oracle sql", "mongodb",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes",
    # Version Control & Tools
    "git", "github", "jira", "postman", "vs code", "intellij idea",
    "jupyter notebook", "google colab", "mlflow", "google sheets",
    "google data studio",
    # Machine Learning
    "machine learning", "deep learning", "tensorflow", "keras",
    "scikit-learn", "xgboost", "opencv", "nlp",
    # Data Analysis
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "power bi",
    "tableau", "excel", "pivot tables", "vlookup", "power query", "dax",
    # Concepts
    "statistics", "probability", "feature engineering", "model evaluation",
    "supervised learning", "unsupervised learning", "a/b testing",
    "data cleaning", "descriptive statistics", "dashboarding",
    "kpi reporting", "oop", "object oriented programming",
    "data structures", "algorithms", "data structures and algorithms",
    "sdlc", "design patterns", "agile", "scrum",
    # Others
    "rest api", "streamlit"
]

# ==========================================================
# Job Role / Title Dictionary
# ==========================================================

ROLE_TITLES = [
    "data scientist intern", "software engineer intern", "data analyst intern",
    "machine learning engineer intern", "research intern",
    "data scientist", "data analyst", "data engineer", "ml engineer",
    "ai engineer", "software engineer", "software developer",
    "backend developer", "frontend developer", "full stack developer",
    "full stack engineer", "web developer", "mobile developer",
    "machine learning engineer", "business analyst", "business intelligence analyst",
    "quantitative analyst", "product manager", "project manager",
    "program manager", "devops engineer", "cloud engineer", "site reliability engineer",
    "qa engineer", "test engineer", "systems engineer",
    "system administrator", "database administrator", "research scientist",
    "research assistant", "teaching assistant", "consultant", "analyst",
    "intern"
]

# ==========================================================
# Extract Resume Text
# ==========================================================

def extract_text(pdf_file):
    text = ""
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    pdf.close()
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# ==========================================================
# Extract Skills
# ==========================================================

def extract_skills(text):
    skills = []
    for skill in SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            skills.append(skill.title())
    return sorted(list(set(skills)))

# ==========================================================
# Extract Roles / Job Titles
# ==========================================================

def extract_roles(text):
    roles = []
    for role in ROLE_TITLES:
        if re.search(r"\b" + re.escape(role) + r"\b", text):
            roles.append(role.title())
    roles = sorted(set(roles), key=len, reverse=True)
    filtered = []
    for r in roles:
        if not any(r.lower() in other.lower() and r != other for other in filtered):
            filtered.append(r)
    return sorted(filtered)

# ==========================================================
# Extract Education Entities (degrees + institutions)
# ==========================================================

def extract_education_entities(text):
    entities = set()

    degree_pattern = re.compile(
        r"\b(b\.?\s?tech|m\.?\s?tech|b\.?\s?e\.?|m\.?\s?s\.?|b\.?\s?s\.?|mca|bca|phd|"
        r"bachelor(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?|"
        r"master(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?)\b",
        re.IGNORECASE
    )
    for m in degree_pattern.finditer(text):
        cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(cleaned) > 1:
            entities.add(cleaned.title())

    uni_pattern = re.compile(
        r"\b(?:university|institute(?:\s+of\s+technology)?|college)\s+of\s+[a-z]+(?:,\s*[a-z]+)?|"
        r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+(?:university|institute of technology|college)\b",
        re.IGNORECASE
    )
    for m in uni_pattern.finditer(text):
        cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(cleaned) > 4:
            entities.add(cleaned.title())

    field_pattern = re.compile(
        r"\b(computer science|data science|information technology|electronics|"
        r"electrical engineering|mechanical engineering|mathematics|statistics)\b",
        re.IGNORECASE
    )
    for m in field_pattern.finditer(text):
        entities.add(m.group(0).strip().title())

    return sorted(entities, key=len, reverse=True)

# ==========================================================
# Extract Summary
# ==========================================================

def extract_summary(text):
    sentences = re.split(r"[.!?]", text)
    summary = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) > 8:
            summary.append(sentence)
        if len(summary) == 3:
            break
    if summary:
        return ". ".join(summary)
    return "Summary not available."

# ==========================================================
# Extract Experience
# ==========================================================

def extract_experience(text):
    pattern = re.compile(
        r"experience(.*?)(education|skills|projects|certifications|references|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        exp = match.group(1)
        exp = re.sub(r"\s+", " ", exp)
        return exp[:700]
    years = re.findall(r"\d+\+?\s+years?", text)
    if years:
        return ", ".join(years)
    return "Experience not found."

# ==========================================================
# Extract Education
# ==========================================================

def extract_education(text):
    pattern = re.compile(
        r"education(.*?)(experience|skills|projects|certifications|references|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        edu = match.group(1)
        edu = re.sub(r"\s+", " ", edu)
        return edu[:600]
    keywords = [
        "b.tech", "b.e", "bachelor", "master", "m.tech", "mca", "bca",
        "phd", "computer science"
    ]
    found = []
    for word in keywords:
        if word in text:
            found.append(word.upper())
    if found:
        return ", ".join(found)
    return "Education not found."

# ==========================================================
# Top-N Predictions
# ==========================================================

def get_predictions(X, clf, top_n=5):
    probabilities = clf.predict_proba(X)[0]
    top_idx = np.argsort(probabilities)[::-1][:top_n]
    recs, conf = [], []
    for idx in top_idx:
        role = label_encoder.inverse_transform([idx])[0]
        recs.append(role)
        conf.append(round(float(probabilities[idx] * 100), 2))
    return recs, conf

# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔥🔥🔥 NEW APP.PY /predict CALLED 🔥🔥🔥")
        if "resume" not in request.files:
            return jsonify({"error": "Resume file not found."}), 400

        file = request.files["resume"]

        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        text = extract_text(file)

        summary = extract_summary(text)
        experience = extract_experience(text)
        education = extract_education(text)
        skills = extract_skills(text)
        roles = extract_roles(text)
        education_entities = extract_education_entities(text)

        X = vectorizer.transform([text])
        X = feature_selector.transform(X)

        recommendations, confidence = get_predictions(X, model, top_n=5)

        try:
            tsne_points = compute_tsne(text)
        except Exception:
            # FIX: print a full traceback instead of just str(err), so the
            # real failing line is visible in the server console the next
            # time this is triggered from the frontend.
            print("⚠️  t-SNE step failed — full traceback below:")
            traceback.print_exc()
            tsne_points = []

        return jsonify({
            "summary": summary,
            "experience": experience,
            "education": education,
            "skills": skills,
            "roles": roles,
            "education_entities": education_entities,
            "recommendations": recommendations,
            "confidence": confidence,
            "model_comparison": MODEL_METRICS,
            "tsne": tsne_points
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Home Route
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "AI Career Recommendation API Running",
        "model": "Logistic Regression"
    })

# ==========================================================
# Health Check
# ==========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# ==========================================================
# Run Server
# ==========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)