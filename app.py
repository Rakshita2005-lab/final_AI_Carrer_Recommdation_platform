# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import fitz
# import joblib
# import numpy as np
# import re
# import os
# import json
# import traceback
# import difflib

# # ==========================================================
# # Flask App
# # ==========================================================

# app = Flask(__name__)
# CORS(app)

# # ==========================================================
# # Load Trained Models
# # ==========================================================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models")

# model = joblib.load(os.path.join(MODEL_DIR, "career_model.pkl"))
# vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
# feature_selector = joblib.load(os.path.join(MODEL_DIR, "feature_selector.pkl"))
# label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# # Extra artifact used only for the "Random Forest" comparison bar.
# # If you don't have this file, comment the two lines out — the app
# # still works, it just won't be able to re-score with RF specifically.
# RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
# random_forest_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None

# print("✅ Models Loaded Successfully")

# # ==========================================================
# # SBERT + career_embeddings.npy  (t-SNE panel)
# # ==========================================================
# # ⚠️ ASSUMPTION: career_embeddings.npy holds one SBERT vector per row of
# # your training resume dataset, generated with 'all-MiniLM-L6-v2'.
# # If you used a different sentence-transformers model to build that
# # file, change SBERT_MODEL_NAME to match — otherwise the résumé's point
# # will land in the wrong part of the embedding space.
# SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# _sbert_model = None  # lazy-loaded, see get_sbert()
# EMBED_BACKEND = None  # "sentence_transformers" or "fastembed", set on first use

# # Try sentence-transformers (torch-based) first, but don't let a failure
# # crash the whole Flask app. On Windows, torch can fail with a DLL load
# # error (WinError 1114) even when the package installs cleanly. If that
# # happens, we fall back to `fastembed` — same MiniLM model family, but
# # runs on ONNX Runtime instead of torch, so it sidesteps the DLL issue
# # entirely. Install it with: pip install fastembed
# try:
#     import sentence_transformers  # noqa: F401
#     EMBED_BACKEND = "sentence_transformers"
#     print("✅ sentence-transformers import OK (using torch backend)")
# except Exception as e:
#     print(f"⚠️  sentence-transformers failed to import ({type(e).__name__}: {e})")
#     try:
#         import fastembed  # noqa: F401
#         EMBED_BACKEND = "fastembed"
#         print("✅ Falling back to fastembed (ONNX Runtime, no torch needed)")
#     except Exception as e2:
#         print(f"⚠️  fastembed also unavailable ({type(e2).__name__}: {e2})")
#         print("⚠️  t-SNE panel will stay empty. Fix torch OR run: pip install fastembed")


# def get_sbert():
#     """Lazy-load an embedding model so the Flask server still boots even
#     if the embedding backend isn't available yet. Returns an object with
#     an .encode(list_of_strings) -> np.ndarray method either way, so
#     calling code doesn't need to know which backend is active."""
#     global _sbert_model
#     if _sbert_model is not None:
#         return _sbert_model

#     if EMBED_BACKEND == "sentence_transformers":
#         from sentence_transformers import SentenceTransformer
#         _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

#     elif EMBED_BACKEND == "fastembed":
#         from fastembed import TextEmbedding

#         # ⚠️ Model-name mapping: fastembed's naming differs from
#         # sentence-transformers'. "BAAI/bge-small-en-v1.5" is fastembed's
#         # closest widely-available equivalent to all-MiniLM-L6-v2 (also
#         # 384-dim), but it is NOT numerically identical — points may sit
#         # slightly differently in the t-SNE plot than a true MiniLM
#         # embedding would. Good enough for visualization; if you need
#         # exact parity, fix torch instead (see chat) so the same
#         # all-MiniLM-L6-v2 model is used everywhere.
#         fe_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

#         class _FastEmbedWrapper:
#             def encode(self, texts):
#                 return np.array(list(fe_model.embed(texts)))

#         _sbert_model = _FastEmbedWrapper()

#     else:
#         raise RuntimeError(
#             "No embedding backend available — install sentence-transformers "
#             "(and fix the torch DLL issue) or run: pip install fastembed"
#         )

#     return _sbert_model


# EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "career_embeddings.npy")
# if os.path.exists(EMBEDDINGS_PATH):
#     career_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
#     # FIX: force a clean float64 2D array. If this .npy was ever saved from
#     # a pandas column, a list-of-lists with ragged rows, or with
#     # allow_pickle=True data, it can load as dtype=object. np.vstack()-ing
#     # an object array together with a normal float array either throws or
#     # produces garbage, which silently killed the t-SNE step before.
#     try:
#         career_embeddings = np.asarray(career_embeddings, dtype=np.float64)
#     except Exception as cast_err:
#         print(f"⚠️  Could not cast career_embeddings.npy to float64: {cast_err}")
#     print(f"✅ career_embeddings.npy loaded — shape {career_embeddings.shape}, dtype {career_embeddings.dtype}")
# else:
#     career_embeddings = None
#     print(f"⚠️  career_embeddings.npy NOT FOUND at {EMBEDDINGS_PATH} — t-SNE panel will stay empty.")

# # Bucket job titles into broad categories for the t-SNE legend/colors.
# # Used both as a fallback when the CSV has no category column, and to
# # fill in categories per-row when the CSV only has a title column (e.g.
# # a single "career" column with no category). Extend this dict to match
# # your label_encoder classes / any new job titles you add.
# # FIX: moved this above the CSV-loading block below, since that block
# # now calls categorize() — it previously sat after that block and would
# # have raised NameError: name 'categorize' is not defined at startup.
# CATEGORY_MAP = {
#     "software engineer": "Software Engineering",
#     "software developer": "Software Engineering",
#     "backend developer": "Software Engineering",
#     "backend engineer": "Software Engineering",
#     "frontend developer": "Software Engineering",
#     "front end developer": "Software Engineering",
#     "full stack developer": "Software Engineering",
#     "full stack engineer": "Software Engineering",
#     "web developer": "Software Engineering",
#     "mobile developer": "Software Engineering",
#     "java developer": "Software Engineering",
#     "python developer": "Software Engineering",
#     "devops engineer": "Cloud & DevOps",
#     "cloud engineer": "Cloud & DevOps",
#     "site reliability engineer": "Cloud & DevOps",
#     "systems engineer": "Cloud & DevOps",
#     "it security analyst": "Cloud & DevOps",
#     "network administrator": "Cloud & DevOps",
#     "data scientist": "Data Science",
#     "data analyst": "Data Science",
#     "data engineer": "Data Science",
#     "database administrator": "Data Science",
#     "ml engineer": "AI & ML",
#     "machine learning engineer": "AI & ML",
#     "ai engineer": "AI & ML",
#     "research scientist": "Research",
#     "research assistant": "Research",
#     "research intern": "Research",
#     "teaching assistant": "Research",
#     "product manager": "Product Management",
#     "program manager": "Product Management",
#     "project manager": "Product Management",
#     "it project manager": "Product Management",
#     "business analyst": "Finance",
#     "business intelligence analyst": "Finance",
#     "quantitative analyst": "Finance",
#     "consultant": "Finance",
# }


# def categorize(job_title):
#     return CATEGORY_MAP.get(str(job_title).strip().lower(), "Other")


# # ⚠️ ASSUMPTION: alongside career_embeddings.npy there is a CSV with the
# # SAME ROW ORDER giving each point's job title + broad category (the
# # legend in your screenshot: Software Engineering, Data Science,
# # Product Management, AI & ML, Research, Finance). Expected path:
# #   models/career_embeddings_meta.csv   with columns: job_title,category
# # If this file doesn't exist yet, generate it once during training
# # (zip your dataset's job-title column with a category mapping) — the
# # app still runs without it, it'll just label every dataset point
# # generically as "Dataset" instead of by category.
# EMB_META_PATH = os.path.join(MODEL_DIR, "career_embeddings_meta.csv")
# emb_meta = None
# if os.path.exists(EMB_META_PATH):
#     import pandas as pd
#     emb_meta = pd.read_csv(EMB_META_PATH)

#     # FIX: be tolerant of column-name variations instead of hard-crashing
#     # inside compute_tsne() with a KeyError the first time a resume is
#     # analyzed. Normalize whatever's there to exactly "job_title" and
#     # "category".
#     def _find_col(df, candidates):
#         lower_map = {c.lower().strip(): c for c in df.columns}
#         for cand in candidates:
#             if cand in lower_map:
#                 return lower_map[cand]
#         return None

#     title_col = _find_col(emb_meta, ["job_title", "title", "role", "job title", "position", "career"])
#     category_col = _find_col(emb_meta, ["category", "job_category", "field", "domain"])

#     rename_map = {}
#     if title_col and title_col != "job_title":
#         rename_map[title_col] = "job_title"
#     if category_col and category_col != "category":
#         rename_map[category_col] = "category"
#     if rename_map:
#         emb_meta = emb_meta.rename(columns=rename_map)
#         print(f"ℹ️  career_embeddings_meta.csv columns renamed for compatibility: {rename_map}")

#     if "job_title" not in emb_meta.columns:
#         emb_meta["job_title"] = ""
#         print("⚠️  No job title column found in career_embeddings_meta.csv — labels will be blank.")
#     if "category" not in emb_meta.columns:
#         # FIX: previously this set every row to the literal string
#         # "Dataset", which is why every point in the t-SNE plot rendered
#         # in one generic gray color regardless of job title. Look each
#         # title up in CATEGORY_MAP instead so points get grouped/colored
#         # by real category (Software Engineering, Data Science, etc.).
#         emb_meta["category"] = emb_meta["job_title"].apply(categorize)
#         print("ℹ️  No category column found in career_embeddings_meta.csv — categories derived from CATEGORY_MAP based on job_title.")

#     print(f"✅ career_embeddings_meta.csv loaded — {len(emb_meta)} rows")
# else:
#     print(f"⚠️  career_embeddings_meta.csv NOT FOUND at {EMB_META_PATH} — dataset points will show as generic 'Dataset'.")

# MAX_TSNE_POINTS = 300  # cap dataset points for speed on a free-tier server


# def compute_tsne(resume_text):
#     """Embed the résumé with SBERT, drop it into the same space as the
#     training embeddings, and run t-SNE so the frontend can scatter-plot
#     it against the dataset. Returns [] if the embeddings file / SBERT
#     model aren't available so the rest of the response still works."""
#     if career_embeddings is None:
#         print("⚠️  compute_tsne skipped — career_embeddings is None (file missing or failed to load).")
#         return []

#     if career_embeddings.ndim != 2 or career_embeddings.shape[0] < 2:
#         print(f"⚠️  compute_tsne skipped — career_embeddings has unusable shape {career_embeddings.shape}.")
#         return []

#     from sklearn.manifold import TSNE

#     sbert = get_sbert()
#     resume_vec = np.asarray(sbert.encode([resume_text]), dtype=np.float64)

#     # FIX: guard against an embedding-dimension mismatch (e.g. SBERT model
#     # producing 768-dim vectors while career_embeddings.npy is 384-dim) —
#     # this used to blow up inside np.vstack with an unhelpful shape error.
#     if resume_vec.shape[1] != career_embeddings.shape[1]:
#         raise ValueError(
#             f"Embedding dimension mismatch: résumé vector is {resume_vec.shape[1]}-dim "
#             f"but career_embeddings.npy is {career_embeddings.shape[1]}-dim. "
#             f"Make sure SBERT_MODEL_NAME matches the model used to build career_embeddings.npy."
#         )

#     base = career_embeddings
#     meta = emb_meta

#     if len(base) > MAX_TSNE_POINTS:
#         rng = np.random.RandomState(42)
#         idx_sample = rng.choice(len(base), MAX_TSNE_POINTS, replace=False)
#         base = base[idx_sample]
#         if meta is not None:
#             meta = meta.iloc[idx_sample].reset_index(drop=True)

#     combined = np.vstack([base, resume_vec])

#     # FIX: t-SNE requires perplexity < n_samples. With very small datasets
#     # (e.g. 13 rows) the old formula could still land too close to the
#     # edge on some sklearn versions. Clamp with a safety margin.
#     n_samples = len(combined)
#     perplexity = min(30, max(2, min(5, n_samples - 1)))
#     if n_samples <= 3:
#         print(f"⚠️  compute_tsne skipped — only {n_samples} total points, not enough for a stable t-SNE layout.")
#         return []

#     tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
#     coords = tsne.fit_transform(combined)

#     points = []
#     for i in range(len(base)):
#         if meta is not None:
#             job_title = str(meta.iloc[i]["job_title"])
#             category = str(meta.iloc[i]["category"])
#         else:
#             job_title = ""
#             category = "Dataset"
#         points.append({
#             "x": float(coords[i][0]),
#             "y": float(coords[i][1]),
#             "category": category,
#             "label": job_title,
#             "is_resume": False,
#         })

#     points.append({
#         "x": float(coords[-1][0]),
#         "y": float(coords[-1][1]),
#         "category": "Your Resume",
#         "label": "Your Resume",
#         "is_resume": True,
#     })
#     return points

# # ==========================================================
# # Model comparison metrics  (bar chart panel)
# # ==========================================================
# # ⚠️ ASSUMPTION: these should come from your training notebook's
# # evaluation step. Save them either as a flat mapping:
# #   {"Logistic Regression": 0.61, "Random Forest": 0.72, "XGBoost": 0.86}
# # or as a nested per-model mapping (what your current model_metrics.json
# # actually looks like):
# #   {"Random Forest": {"training_accuracy": 0.96, "testing_accuracy": 0.92,
# #                       "balanced_accuracy": 0.86}, ...}
# # Either shape works now — see the flattening step below.
# METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# # FIX: this is the actual bug you hit. The frontend's bar chart does
# # Object.values(modelComparison) and expects each value to be a single
# # number. Your model_metrics.json changed to store a dict of {training_
# # accuracy, testing_accuracy, balanced_accuracy} per model, so Chart.js
# # was being handed objects instead of numbers and silently drew nothing.
# # We flatten each model down to ONE representative score here, preferring
# # testing_accuracy (or balanced_accuracy as a fallback) since that's the
# # most meaningful "how good is this model on unseen résumés" number.
# PREFERRED_METRIC_KEYS = [
#     "macro_f1",
#     "f1",
#     "testing_accuracy"
# ]


# def _flatten_metrics(raw):
#     flat = {}
#     for name, val in raw.items():
#         if isinstance(val, dict):
#             chosen = None
#             for key in PREFERRED_METRIC_KEYS:
#                 if key in val:
#                     chosen = val[key]
#                     break
#             if chosen is None and val:
#                 # last resort: just take the first numeric value present
#                 chosen = next(iter(val.values()))
#             flat[name] = round(float(chosen), 4) if chosen is not None else 0.0
#         else:
#             flat[name] = round(float(val), 4)
#     return flat


# if os.path.exists(METRICS_PATH):
#     with open(METRICS_PATH) as f:
#         _raw_metrics = json.load(f)
#     MODEL_METRICS = _flatten_metrics(_raw_metrics)
#     print(f"✅ model_metrics.json loaded and flattened for charting: {MODEL_METRICS}")
# else:
#     MODEL_METRICS = {
#         "Logistic Regression": 0.61,
#         "Random Forest": 0.72,
#         "XGBoost": 0.86,
#     }
#     print(f"⚠️  model_metrics.json NOT FOUND at {METRICS_PATH} — using fallback values: {MODEL_METRICS}")

# # ==========================================================
# # Skills Dictionary
# # ==========================================================

# SKILLS = [
#     # ---------------- Languages ----------------
#     "python", "java", "c", "c++", "c#", "sql", "r", "go", "golang",
#     "rust", "kotlin", "swift", "php", "ruby", "scala", "matlab",
#     "perl", "dart", "objective-c", "shell scripting", "bash",
#     "powershell", "vba",

#     # ---------------- Web Technologies ----------------
#     "html", "html5", "css", "css3", "javascript", "typescript",
#     "sass", "less", "bootstrap", "tailwind", "tailwind css",
#     "material ui", "webpack", "babel", "jquery",

#     # ---------------- Frontend Frameworks ----------------
#     "react", "react.js", "angular", "vue", "vue.js", "next.js",
#     "nuxt.js", "svelte", "redux", "ember.js",

#     # ---------------- Backend Frameworks ----------------
#     "nodejs", "node.js", "express", "express.js", "flask", "django",
#     "fastapi", "spring", "spring boot", ".net", "asp.net", "junit",
#     "laravel", "ruby on rails", "nestjs", "graphql", "grpc",
#     "hibernate", "maven", "gradle", "jpa",

#     # ---------------- Mobile Development ----------------
#     "android", "ios", "flutter", "react native", "xamarin",
#     "swiftui", "jetpack compose",

#     # ---------------- Databases ----------------
#     "mysql", "postgresql", "oracle", "oracle sql", "mongodb",
#     "redis", "cassandra", "dynamodb", "sqlite", "mariadb",
#     "firebase", "elasticsearch", "neo4j", "supabase",

#     # ---------------- Cloud & DevOps ----------------
#     "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
#     "terraform", "ansible", "jenkins", "ci/cd", "gitlab",
#     "gitlab ci", "bitbucket", "heroku", "vercel", "netlify",
#     "microservices", "linux", "nginx", "apache",

#     # ---------------- Version Control & Tools ----------------
#     "git", "github", "jira", "postman", "vs code", "intellij idea",
#     "jupyter notebook", "google colab", "mlflow", "google sheets",
#     "google data studio", "figma", "adobe xd", "confluence", "trello",

#     # ---------------- Testing ----------------
#     "selenium", "cypress", "jest", "mocha", "pytest", "unit testing",
#     "api testing", "test automation", "junit", "postman",

#     # ---------------- Machine Learning & AI ----------------
#     "machine learning", "deep learning", "tensorflow", "keras",
#     "pytorch", "scikit-learn", "xgboost", "opencv", "nlp",
#     "computer vision", "hugging face", "transformers", "langchain",
#     "openai api", "llm", "generative ai", "cnn", "rnn", "lstm",
#     "bert", "spark", "pyspark", "hadoop", "airflow", "reinforcement learning",

#     # ---------------- Data Analysis & Visualization ----------------
#     "pandas", "numpy", "scipy", "matplotlib", "seaborn", "power bi",
#     "tableau", "excel", "pivot tables", "vlookup", "power query", "dax",
#     "plotly", "d3.js", "looker", "google analytics",

#     # ---------------- Concepts ----------------
#     "statistics", "probability", "feature engineering", "model evaluation",
#     "supervised learning", "unsupervised learning", "a/b testing",
#     "data cleaning", "descriptive statistics", "dashboarding",
#     "kpi reporting", "oop", "object oriented programming",
#     "data structures", "algorithms", "data structures and algorithms",
#     "sdlc", "design patterns", "agile", "scrum", "kanban",
#     "system design", "distributed systems", "data warehousing",
#     "etl", "data modeling",

#     # ---------------- Security / Blockchain ----------------
#     "cybersecurity", "penetration testing", "ethical hacking",
#     "blockchain", "solidity", "web3", "cryptography",

#     # ---------------- Soft Skills ----------------
#     "communication", "leadership", "teamwork", "problem solving",
#     "time management", "critical thinking", "project management",
#     "collaboration", "adaptability", "presentation skills",

#     # ---------------- Project & Program Management ----------------
#     "risk management", "stakeholder management", "waterfall",
#     "ms project", "microsoft project", "team management",
#     "conflict resolution", "vendor management", "resource allocation",
#     "budgeting", "cost management", "quality management",
#     "change management", "sprint planning", "user stories",
#     "roadmap planning", "product roadmap", "gantt chart", "okrs",
#     "requirements gathering", "status reporting", "trello", "confluence",

#     # ---------------- Others ----------------
#     "rest api", "streamlit", "ui/ux", "responsive design", "seo"
# ]

# # ==========================================================
# # Job Role / Title Dictionary
# # ==========================================================

# ROLE_TITLES = [
#     "data scientist intern", "software engineer intern", "data analyst intern",
#     "machine learning engineer intern", "research intern",
#     "data scientist", "data analyst", "data engineer", "ml engineer",
#     "ai engineer", "software engineer", "software developer",
#     "backend developer", "frontend developer", "full stack developer",
#     "full stack engineer", "web developer", "mobile developer",
#     "machine learning engineer", "business analyst", "business intelligence analyst",
#     "quantitative analyst", "product manager", "project manager",
#     "program manager", "devops engineer", "cloud engineer", "site reliability engineer",
#     "qa engineer", "test engineer", "systems engineer",
#     "system administrator", "database administrator", "research scientist",
#     "research assistant", "teaching assistant", "consultant", "analyst",
#     "intern"
# ]

# # ==========================================================
# # Skill → Role mapping (used by /gap-report)
# # ==========================================================
# # ⚠️ IMPORTANT: these dict keys must exactly match the role names your
# # model actually predicts — i.e. the values in label_encoder.classes_.
# # Run `print(label_encoder.classes_)` once and adjust the keys below
# # to match exactly (including casing), or /gap-report will 404 for
# # any predicted role that isn't listed here.
# #
# # Skill names use .title() casing to match what extract_skills() below
# # produces (e.g. "rest api" -> "Rest Api"), so alignment percentages
# # come out correct instead of showing every skill as "missing".

# ROLE_SKILLS = {
#     "Full Stack Developer": [
#         "Html", "Css", "Javascript", "React", "Node.Js", "Python", "Sql",
#         "Rest Api", "Git", "Mongodb"
#     ],
#     "Java Developer": [
#         "Java", "Oop", "Sql", "Spring Boot", "Spring", "Hibernate",
#         "Maven", "Git", "Rest Api", "Junit"
#     ],
#     "Full Stack Java Developer": [
#         "Java", "Spring Boot", "Spring", "Hibernate", "Maven", "Html",
#         "Css", "Javascript", "React", "Sql", "Mysql", "Rest Api", "Git", "Oop"
#     ],
#     "Python Developer": [
#         "Python", "Oop", "Sql", "Git", "Rest Api", "Flask", "Fastapi",
#         "Django", "Pytest"
#     ],
#     "Frontend Developer": [
#         "Html", "Css", "Javascript", "Typescript", "React", "Angular",
#         "Vue", "Tailwind Css", "Git", "Responsive Design"
#     ],
#     "Backend Developer": [
#         "Python", "Java", "Node.Js", "Sql", "Rest Api", "Graphql", "Git",
#         "Docker", "Mongodb", "Microservices"
#     ],
#     "Web Developer": [
#         "Html", "Css", "Javascript", "React", "Node.Js", "Git",
#         "Responsive Design", "Bootstrap", "Rest Api"
#     ],
#     "Mobile Developer": [
#         "Kotlin", "Swift", "Flutter", "React Native", "Android", "Ios", "Git"
#     ],
#     "Data Scientist": [
#         "Python", "Sql", "Machine Learning", "Pandas", "Numpy",
#         "Scikit-Learn", "Statistics", "Matplotlib", "Deep Learning"
#     ],
#     "Data Analyst": [
#         "Sql", "Excel", "Power Bi", "Tableau", "Statistics",
#         "Data Cleaning", "Pandas", "Python", "Plotly"
#     ],
#     "Data Engineer": [
#         "Python", "Sql", "Docker", "Aws", "Mongodb", "Mysql",
#         "Postgresql", "Spark", "Airflow", "Etl"
#     ],
#     "Machine Learning Engineer": [
#         "Python", "Machine Learning", "Deep Learning", "Tensorflow",
#         "Pytorch", "Keras", "Sql", "Docker", "Mlflow"
#     ],
#     "Ai Engineer": [
#         "Python", "Machine Learning", "Deep Learning", "Tensorflow",
#         "Pytorch", "Nlp", "Opencv", "Transformers", "Llm"
#     ],
#     "Software Engineer": [
#         "Java", "Python", "C++", "Data Structures", "Algorithms", "Git",
#         "Oop", "System Design"
#     ],
#     "Software Developer": [
#         "Java", "Python", "Sql", "Git", "Oop", "Data Structures",
#         "Design Patterns"
#     ],
#     "Business Analyst": [
#         "Sql", "Excel", "Power Bi", "Tableau", "Statistics", "Data Cleaning"
#     ],
#     "Business Intelligence Analyst": [
#         "Sql", "Power Bi", "Tableau", "Excel", "Dax", "Power Query"
#     ],
#     "Quantitative Analyst": [
#         "Python", "Sql", "Statistics", "Probability", "Excel"
#     ],
#     "Product Manager": [
#         "Agile", "Scrum", "Jira", "Sql", "Excel", "Project Management"
#     ],
#     "Project Manager": [
#         "Agile", "Scrum", "Jira", "Sdlc", "Project Management",
#         "Communication", "Leadership", "Risk Management","Java Core", "Swing",
#         "Stakeholder Management", "Team Management", "Waterfall",
#         "Sprint Planning", "Budgeting", "Change Management",
#         "Conflict Resolution", "Trello", "Confluence",
#         "Requirements Gathering", "Status Reporting"
#     ],
#     "Program Manager": [
#         "Agile", "Scrum", "Jira", "Sdlc", "Excel", "Leadership"
#     ],
#     "Devops Engineer": [
#         "Docker", "Kubernetes", "Aws", "Azure", "Terraform", "Ansible",
#         "Jenkins", "Ci/Cd", "Git", "Linux"
#     ],
#     "Cloud Engineer": [
#         "Aws", "Azure", "Gcp", "Docker", "Kubernetes", "Terraform", "Linux"
#     ],
#     "Site Reliability Engineer": [
#         "Docker", "Kubernetes", "Aws", "Azure", "Git", "Linux", "Ci/Cd"
#     ],
#     "Qa Engineer": [
#         "Junit", "Postman", "Selenium", "Cypress", "Sql", "Git",
#         "Test Automation", "Jira"
#     ],
#     "Test Engineer": [
#         "Junit", "Postman", "Selenium", "Cypress", "Sql", "Git",
#         "Api Testing", "Jira"
#     ],
#     "Systems Engineer": [
#         "Aws", "Azure", "Docker", "Git", "Sql", "Linux"
#     ],
#     "System Administrator": [
#         "Aws", "Azure", "Git", "Sql", "Linux", "Bash"
#     ],
#     "Database Administrator": [
#         "Sql", "Mysql", "Postgresql", "Oracle", "Oracle Sql", "Mongodb",
#         "Data Modeling"
#     ],
#     "Research Scientist": [
#         "Python", "Machine Learning", "Statistics", "Nlp", "Deep Learning"
#     ],
#     "Research Assistant": [
#         "Python", "Statistics", "Data Cleaning"
#     ],
#     "Consultant": [
#         "Excel", "Sql", "Statistics", "Power Bi", "Communication"
#     ],
#     "Analyst": [
#         "Sql", "Excel", "Statistics", "Data Cleaning"
#     ],
#     "Ui/Ux Designer": [
#         "Figma", "Adobe Xd", "Ui/Ux", "Responsive Design", "Html", "Css"
#     ],
#     "Blockchain Developer": [
#         "Solidity", "Web3", "Blockchain", "Javascript", "Cryptography"
#     ],
#     "Cybersecurity Analyst": [
#         "Cybersecurity", "Penetration Testing", "Ethical Hacking", "Linux",
#         "Networking"
#     ],
# }

# SKILL_SUGGESTIONS = {
#     "Java": "Learn Java syntax, collections, exception handling and multithreading.",
#     "Oop": "Practice inheritance, polymorphism, abstraction and encapsulation.",
#     "Spring Boot": "Build REST APIs using Spring Boot.",
#     "Spring": "Learn the Spring Framework fundamentals.",
#     "Hibernate": "Learn Hibernate/JPA for ORM-based database integration.",
#     "Jpa": "Learn the Java Persistence API for ORM mapping.",
#     "Maven": "Learn Maven dependency management and build lifecycle.",
#     "Gradle": "Learn Gradle build scripts and dependency management.",
#     "Sql": "Practice joins, subqueries, normalization and database design.",
#     "Git": "Practice Git branching, merging and collaborative workflows.",
#     "Rest Api": "Build and consume REST APIs.",
#     "Graphql": "Learn to design and query GraphQL APIs.",
#     "Python": "Strengthen Python programming and advanced concepts.",
#     "Html": "Practice semantic HTML and accessibility.",
#     "Css": "Learn Flexbox, Grid and responsive layouts.",
#     "Tailwind Css": "Practice building layouts with Tailwind's utility classes.",
#     "Bootstrap": "Practice responsive layouts using Bootstrap components.",
#     "Javascript": "Strengthen ES6+, DOM manipulation and asynchronous JavaScript.",
#     "Typescript": "Learn static typing, interfaces and generics in TypeScript.",
#     "React": "Build React applications using components and hooks.",
#     "Node.Js": "Build backend services and APIs with Node.js.",
#     "Angular": "Learn Angular components, services and routing.",
#     "Vue": "Practice building small apps with Vue.js.",
#     "Flask": "Build lightweight REST APIs using Flask.",
#     "Fastapi": "Build production-ready REST APIs using FastAPI.",
#     "Django": "Create a complete web application using Django.",
#     "Pytest": "Practice writing test suites with pytest.",
#     "Docker": "Learn containerization and Docker deployment.",
#     "Kubernetes": "Learn container orchestration basics.",
#     "Terraform": "Practice infrastructure-as-code with Terraform.",
#     "Ansible": "Learn configuration management and automation with Ansible.",
#     "Jenkins": "Set up CI/CD pipelines using Jenkins.",
#     "Ci/Cd": "Learn to build automated CI/CD pipelines.",
#     "Linux": "Get comfortable with the Linux command line and shell.",
#     "Bash": "Practice writing shell scripts in Bash.",
#     "Mongodb": "Practice CRUD operations and schema design in MongoDB.",
#     "Mysql": "Practice SQL queries, joins and indexing in MySQL.",
#     "Postgresql": "Practice SQL queries and schema design in PostgreSQL.",
#     "Oracle": "Learn Oracle database fundamentals.",
#     "Oracle Sql": "Practice PL/SQL and Oracle-specific SQL features.",
#     "Data Modeling": "Practice designing normalized, scalable data models.",
#     "Microservices": "Learn to design and deploy microservice architectures.",
#     "Machine Learning": "Practice supervised and unsupervised ML projects.",
#     "Deep Learning": "Learn neural networks using TensorFlow or PyTorch.",
#     "Tensorflow": "Build and train deep learning models with TensorFlow.",
#     "Pytorch": "Build and train deep learning models with PyTorch.",
#     "Keras": "Practice model-building with the Keras API.",
#     "Mlflow": "Track experiments and manage ML models with MLflow.",
#     "Pandas": "Practice data wrangling and cleaning with Pandas.",
#     "Numpy": "Get comfortable with array operations in NumPy.",
#     "Scikit-Learn": "Practice building ML pipelines with scikit-learn.",
#     "Statistics": "Review descriptive statistics and probability.",
#     "Probability": "Review probability theory fundamentals.",
#     "Matplotlib": "Practice data visualization with Matplotlib.",
#     "Plotly": "Build interactive visualizations with Plotly.",
#     "Nlp": "Learn text preprocessing and NLP model basics.",
#     "Opencv": "Practice image processing with OpenCV.",
#     "Transformers": "Learn transformer architectures for NLP/vision tasks.",
#     "Llm": "Explore prompting and fine-tuning large language models.",
#     "Spark": "Learn distributed data processing with Apache Spark.",
#     "Airflow": "Practice orchestrating data pipelines with Airflow.",
#     "Etl": "Practice building extract-transform-load data pipelines.",
#     "C++": "Strengthen core C++ syntax and STL usage.",
#     "Data Structures": "Practice arrays, trees, graphs, and linked lists.",
#     "Algorithms": "Practice sorting, searching, and complexity analysis.",
#     "System Design": "Practice designing scalable backend systems.",
#     "Design Patterns": "Learn common OOP design patterns and when to use them.",
#     "Excel": "Practice formulas, pivot tables, and VLOOKUP.",
#     "Power Bi": "Build interactive dashboards in Power BI.",
#     "Tableau": "Practice building visualizations in Tableau.",
#     "Data Cleaning": "Practice handling missing/inconsistent data.",
#     "Dax": "Learn DAX formulas for Power BI measures.",
#     "Power Query": "Practice data transformation with Power Query.",
#     "Agile": "Learn Agile principles and iterative development.",
#     "Scrum": "Understand Scrum roles, ceremonies and artifacts.",
#     "Jira": "Practice sprint planning and issue tracking using Jira.",
#     "Sdlc": "Review the software development lifecycle stages.",
#     "Project Management": "Practice planning, scheduling and tracking projects.",
#     "Communication": "Practice clear written and verbal communication with stakeholders.",
#     "Leadership": "Develop decision-making and team-leading skills.",
#     "Aws": "Get familiar with core AWS services (EC2, S3, IAM).",
#     "Azure": "Get familiar with core Azure services.",
#     "Gcp": "Get familiar with core Google Cloud Platform services.",
#     "Junit": "Practice writing unit tests with JUnit.",
#     "Postman": "Practice testing APIs with Postman.",
#     "Selenium": "Practice browser automation and UI testing with Selenium.",
#     "Cypress": "Practice end-to-end testing with Cypress.",
#     "Test Automation": "Build automated test suites for regression coverage.",
#     "Api Testing": "Practice validating REST APIs with tools like Postman.",
#     "Kotlin": "Learn Kotlin fundamentals for Android development.",
#     "Swift": "Learn Swift fundamentals for iOS development.",
#     "Flutter": "Build cross-platform mobile apps with Flutter.",
#     "React Native": "Build cross-platform mobile apps with React Native.",
#     "Android": "Learn Android app development fundamentals.",
#     "Ios": "Learn iOS app development fundamentals.",
#     "Responsive Design": "Practice building mobile-first, responsive layouts.",
#     "Figma": "Practice designing UI mockups and prototypes in Figma.",
#     "Adobe Xd": "Practice designing UI mockups and prototypes in Adobe XD.",
#     "Ui/Ux": "Study usability principles and user-centered design.",
#     "Solidity": "Learn smart contract development with Solidity.",
#     "Web3": "Explore decentralized app development fundamentals.",
#     "Blockchain": "Learn blockchain fundamentals and consensus mechanisms.",
#     "Cryptography": "Review core cryptographic principles and algorithms.",
#     "Cybersecurity": "Study core security principles and common attack vectors.",
#     "Penetration Testing": "Practice ethical penetration testing techniques.",
#     "Ethical Hacking": "Learn ethical hacking methodologies and tools.",
#     "Networking": "Review core networking concepts (TCP/IP, DNS, firewalls).",
#     "Risk Management": "Learn how to identify, assess and mitigate project risks.",
#     "Stakeholder Management": "Practice aligning expectations across stakeholders and sponsors.",
#     "Waterfall": "Understand the Waterfall methodology and when to apply it.",
#     "Ms Project": "Practice scheduling and tracking projects in MS Project.",
#     "Microsoft Project": "Practice scheduling and tracking projects in Microsoft Project.",
#     "Team Management": "Practice delegating, motivating and coordinating a project team.",
#     "Conflict Resolution": "Develop techniques for resolving team and stakeholder conflicts.",
#     "Vendor Management": "Learn to manage vendor contracts, SLAs and deliverables.",
#     "Resource Allocation": "Practice planning and balancing team workload across tasks.",
#     "Budgeting": "Practice estimating and tracking project budgets.",
#     "Cost Management": "Learn to monitor and control project costs against budget.",
#     "Quality Management": "Learn quality assurance and control practices for projects.",
#     "Change Management": "Learn to manage scope changes and organizational change.",
#     "Sprint Planning": "Practice planning and estimating sprints in Agile teams.",
#     "User Stories": "Practice writing clear, testable user stories.",
#     "Roadmap Planning": "Practice building and prioritizing product/project roadmaps.",
#     "Product Roadmap": "Practice building and communicating a product roadmap.",
#     "Gantt Chart": "Practice building Gantt charts to visualize project timelines.",
#     "Okrs": "Learn to set and track Objectives and Key Results.",
#     "Requirements Gathering": "Practice eliciting and documenting stakeholder requirements.",
#     "Status Reporting": "Practice writing clear project status updates for stakeholders.",
#     "Trello": "Practice organizing tasks and workflows on Trello boards.",
#     "Confluence": "Practice documenting project plans and specs in Confluence.",
# }

# # ==========================================================
# # Extract Resume Text
# # ==========================================================

# def extract_text(pdf_file):
#     text = ""
#     pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     for page in pdf:
#         text += page.get_text()
#     pdf.close()
#     text = re.sub(r"\s+", " ", text)
#     return text.lower().strip()

# # ==========================================================
# # Extract Skills
# # ==========================================================

# def extract_skills(text):
#     skills = []
#     for skill in SKILLS:
#         if re.search(r"\b" + re.escape(skill) + r"\b", text):
#             skills.append(skill.title())
#     return sorted(list(set(skills)))

# # ==========================================================
# # Extract Roles / Job Titles
# # ==========================================================

# def extract_roles(text):
#     roles = []
#     for role in ROLE_TITLES:
#         if re.search(r"\b" + re.escape(role) + r"\b", text):
#             roles.append(role.title())
#     roles = sorted(set(roles), key=len, reverse=True)
#     filtered = []
#     for r in roles:
#         if not any(r.lower() in other.lower() and r != other for other in filtered):
#             filtered.append(r)
#     return sorted(filtered)

# # ==========================================================
# # Extract Education Entities (degrees + institutions)
# # ==========================================================

# def extract_education_entities(text):
#     entities = set()

#     degree_pattern = re.compile(
#         r"\b(b\.?\s?tech|m\.?\s?tech|b\.?\s?e\.?|m\.?\s?s\.?|b\.?\s?s\.?|mca|bca|phd|"
#         r"bachelor(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?|"
#         r"master(?:'s)?(?:\s+of\s+[a-z]+(?:\s+[a-z]+)?)?)\b",
#         re.IGNORECASE
#     )
#     for m in degree_pattern.finditer(text):
#         cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
#         if len(cleaned) > 1:
#             entities.add(cleaned.title())

#     uni_pattern = re.compile(
#         r"\b(?:university|institute(?:\s+of\s+technology)?|college)\s+of\s+[a-z]+(?:,\s*[a-z]+)?|"
#         r"\b[a-z]+(?:\s+[a-z]+){0,3}\s+(?:university|institute of technology|college)\b",
#         re.IGNORECASE
#     )
#     for m in uni_pattern.finditer(text):
#         cleaned = re.sub(r"\s+", " ", m.group(0)).strip()
#         if len(cleaned) > 4:
#             entities.add(cleaned.title())

#     field_pattern = re.compile(
#         r"\b(computer science|data science|information technology|electronics|"
#         r"electrical engineering|mechanical engineering|mathematics|statistics)\b",
#         re.IGNORECASE
#     )
#     for m in field_pattern.finditer(text):
#         entities.add(m.group(0).strip().title())

#     return sorted(entities, key=len, reverse=True)

# # ==========================================================
# # Extract Summary
# # ==========================================================

# def extract_summary(text):
#     sentences = re.split(r"[.!?]", text)
#     summary = []
#     for sentence in sentences:
#         sentence = sentence.strip()
#         if len(sentence.split()) > 8:
#             summary.append(sentence)
#         if len(summary) == 3:
#             break
#     if summary:
#         return ". ".join(summary)
#     return "Summary not available."

# # ==========================================================
# # Extract Experience
# # ==========================================================

# def extract_experience(text):
#     pattern = re.compile(
#         r"experience(.*?)(education|skills|projects|certifications|references|$)",
#         re.IGNORECASE | re.DOTALL
#     )
#     match = pattern.search(text)
#     if match:
#         exp = match.group(1)
#         exp = re.sub(r"\s+", " ", exp)
#         return exp[:700]
#     years = re.findall(r"\d+\+?\s+years?", text)
#     if years:
#         return ", ".join(years)
#     return "Experience not found."

# # ==========================================================
# # Extract Education
# # ==========================================================

# def extract_education(text):
#     pattern = re.compile(
#         r"education(.*?)(experience|skills|projects|certifications|references|$)",
#         re.IGNORECASE | re.DOTALL
#     )
#     match = pattern.search(text)
#     if match:
#         edu = match.group(1)
#         edu = re.sub(r"\s+", " ", edu)
#         return edu[:600]
#     keywords = [
#         "b.tech", "b.e", "bachelor", "master", "m.tech", "mca", "bca",
#         "phd", "computer science"
#     ]
#     found = []
#     for word in keywords:
#         if word in text:
#             found.append(word.upper())
#     if found:
#         return ", ".join(found)
#     return "Education not found."

# # ==========================================================
# # Top-N Predictions
# # ==========================================================

# def get_predictions(X, clf, top_n=5):
#     probabilities = clf.predict_proba(X)[0]
#     top_idx = np.argsort(probabilities)[::-1][:top_n]
#     recs, conf = [], []
#     for idx in top_idx:
#         role = label_encoder.inverse_transform([idx])[0]
#         recs.append(role)
#         conf.append(round(float(probabilities[idx] * 100), 2))
#     return recs, conf

# # ==========================================================
# # Prediction Endpoint
# # ==========================================================

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         print("🔥🔥🔥 NEW APP.PY /predict CALLED 🔥🔥🔥")
#         if "resume" not in request.files:
#             return jsonify({"error": "Resume file not found."}), 400

#         file = request.files["resume"]

#         if file.filename == "":
#             return jsonify({"error": "No file selected."}), 400

#         text = extract_text(file)

#         summary = extract_summary(text)
#         experience = extract_experience(text)
#         education = extract_education(text)
#         skills = extract_skills(text)
#         roles = extract_roles(text)
#         education_entities = extract_education_entities(text)

#         X = vectorizer.transform([text])
#         X = feature_selector.transform(X)

#         recommendations, confidence = get_predictions(X, model, top_n=5)

#         try:
#             tsne_points = compute_tsne(text)
#         except Exception:
#             # FIX: print a full traceback instead of just str(err), so the
#             # real failing line is visible in the server console the next
#             # time this is triggered from the frontend.
#             print("⚠️  t-SNE step failed — full traceback below:")
#             traceback.print_exc()
#             tsne_points = []

#         return jsonify({
#             "summary": summary,
#             "experience": experience,
#             "education": education,
#             "skills": skills,
#             "roles": roles,
#             "education_entities": education_entities,
#             "recommendations": recommendations,
#             "confidence": confidence,
#             "model_comparison": MODEL_METRICS,
#             "tsne": tsne_points
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# # ==========================================================
# # Skill Gap Report Endpoint
# # ==========================================================

# @app.route("/gap-report", methods=["POST"])
# def gap_report():
#     try:
#         data = request.get_json(force=True) or {}
#         role = data.get("job_role")
#         skills = data.get("skills", [])

#         required_skills = ROLE_SKILLS.get(role)
#         resolved_role = role

#         # FIX: the model can predict role names (e.g. "Full Stack Java
#         # Developer") that don't have an exact ROLE_SKILLS entry yet.
#         # Instead of hard-failing with 404, fall back to the closest
#         # known role name so the skill gap still renders. This also
#         # future-proofs against any new label_encoder classes added
#         # later without a matching ROLE_SKILLS key.
#         if required_skills is None and role:
#             close = difflib.get_close_matches(
#                 role, ROLE_SKILLS.keys(), n=1, cutoff=0.55
#             )
#             if close:
#                 resolved_role = close[0]
#                 required_skills = ROLE_SKILLS[resolved_role]

#         if required_skills is None:
#             return jsonify({"error": f"No skill mapping found for '{role}'."}), 404

#         user_skills_normalized = {s.strip().lower() for s in skills}

#         matched_skills = [
#             s for s in required_skills if s.lower() in user_skills_normalized
#         ]
#         missing_skills = [
#             s for s in required_skills if s.lower() not in user_skills_normalized
#         ]

#         total = len(required_skills)
#         alignment = round(len(matched_skills) / total * 100, 2) if total else 0

#         suggestions = [
#             {
#                 "skill": s,
#                 "suggestion": SKILL_SUGGESTIONS.get(s, f"Improve your knowledge of {s}.")
#             }
#             for s in missing_skills
#         ]

#         return jsonify({
#             "job_role": role,
#             "your_skills": matched_skills,
#             "missing_skills": missing_skills,
#             "skill_alignment": alignment,
#             "suggestions": suggestions
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# # ==========================================================
# # Home Route
# # ==========================================================

# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "status": "success",
#         "message": "AI Career Recommendation API Running",
#         "model": "Logistic Regression"
#     })

# # ==========================================================
# # Health Check
# # ==========================================================

# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"})

# # ==========================================================
# # Run Server
# # ==========================================================
# # FIX: this used to call uvicorn.run("api:app", ...), which launched a
# # completely different FastAPI app defined in api.py instead of this
# # Flask app — so every route above (including the correctly-implemented
# # file-upload /predict) was never actually served. Run this Flask app
# # directly instead.

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=8000, debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import joblib
import numpy as np
import re
import os
import json
import traceback
import difflib

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
# ⚠️ ASSUMPTION: career_embeddings.npy holds one embedding vector per row
# of your training resume dataset.
#
# FIX (memory): this used to try sentence-transformers (torch-based)
# first and fall back to fastembed. On Render's free 512MB instance,
# loading torch + a transformer model on top of your sklearn models was
# regularly exceeding available memory, causing Gunicorn's worker to be
# OOM-killed mid-request — which the Render proxy reports to the browser
# as a 502 Bad Gateway (and can *look* like a CORS error in devtools,
# since a killed connection never sends CORS headers either).
#
# fastembed runs on ONNX Runtime instead of torch and uses a fraction of
# the memory, at the cost of not being byte-identical to a true
# sentence-transformers MiniLM embedding. Good enough for the t-SNE
# visualization; if you need exact parity later, upgrade your Render
# plan to get more RAM and swap sentence-transformers back in.
EMBED_BACKEND = None

try:
    import fastembed  # noqa: F401
    EMBED_BACKEND = "fastembed"
    print("✅ Using fastembed (ONNX Runtime, low memory — no torch)")
except Exception as e:
    print(f"⚠️  fastembed unavailable ({type(e).__name__}: {e})")
    print("⚠️  t-SNE panel will stay empty. Run: pip install fastembed")


_sbert_model = None  # lazy-loaded, see get_sbert()


def get_sbert():
    """Lazy-load the embedding model so the Flask server still boots even
    if fastembed isn't available yet. Returns an object with an
    .encode(list_of_strings) -> np.ndarray method."""
    global _sbert_model
    if _sbert_model is not None:
        return _sbert_model

    if EMBED_BACKEND != "fastembed":
        raise RuntimeError(
            "No embedding backend available — run: pip install fastembed"
        )

    from fastembed import TextEmbedding

    # fastembed's closest widely-available equivalent to all-MiniLM-L6-v2
    # (also 384-dim). Not numerically identical to true MiniLM, but fine
    # for visualization purposes.
    fe_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    class _FastEmbedWrapper:
        def encode(self, texts):
            return np.array(list(fe_model.embed(texts)))

    _sbert_model = _FastEmbedWrapper()
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
    """Embed the résumé, drop it into the same space as the training
    embeddings, and run t-SNE so the frontend can scatter-plot it against
    the dataset. Returns [] if the embeddings file / embedding backend
    aren't available so the rest of the response still works."""
    if career_embeddings is None:
        print("⚠️  compute_tsne skipped — career_embeddings is None (file missing or failed to load).")
        return []

    if career_embeddings.ndim != 2 or career_embeddings.shape[0] < 2:
        print(f"⚠️  compute_tsne skipped — career_embeddings has unusable shape {career_embeddings.shape}.")
        return []

    if EMBED_BACKEND != "fastembed":
        print("⚠️  compute_tsne skipped — fastembed not available.")
        return []

    from sklearn.manifold import TSNE

    sbert = get_sbert()
    resume_vec = np.asarray(sbert.encode([resume_text]), dtype=np.float64)

    # FIX: guard against an embedding-dimension mismatch (e.g. embedding
    # model producing a different dimension than career_embeddings.npy) —
    # this used to blow up inside np.vstack with an unhelpful shape error.
    if resume_vec.shape[1] != career_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: résumé vector is {resume_vec.shape[1]}-dim "
            f"but career_embeddings.npy is {career_embeddings.shape[1]}-dim. "
            f"Regenerate career_embeddings.npy using the same fastembed model."
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
# or as a nested per-model mapping:
#   {"Random Forest": {"training_accuracy": 0.96, "testing_accuracy": 0.92,
#                       "balanced_accuracy": 0.86}, ...}
# Either shape works — see the flattening step below.
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

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
    # ---------------- Languages ----------------
    "python", "java", "c", "c++", "c#", "sql", "r", "go", "golang",
    "rust", "kotlin", "swift", "php", "ruby", "scala", "matlab",
    "perl", "dart", "objective-c", "shell scripting", "bash",
    "powershell", "vba",

    # ---------------- Web Technologies ----------------
    "html", "html5", "css", "css3", "javascript", "typescript",
    "sass", "less", "bootstrap", "tailwind", "tailwind css",
    "material ui", "webpack", "babel", "jquery",

    # ---------------- Frontend Frameworks ----------------
    "react", "react.js", "angular", "vue", "vue.js", "next.js",
    "nuxt.js", "svelte", "redux", "ember.js",

    # ---------------- Backend Frameworks ----------------
    "nodejs", "node.js", "express", "express.js", "flask", "django",
    "fastapi", "spring", "spring boot", ".net", "asp.net", "junit",
    "laravel", "ruby on rails", "nestjs", "graphql", "grpc",
    "hibernate", "maven", "gradle", "jpa",

    # ---------------- Mobile Development ----------------
    "android", "ios", "flutter", "react native", "xamarin",
    "swiftui", "jetpack compose",

    # ---------------- Databases ----------------
    "mysql", "postgresql", "oracle", "oracle sql", "mongodb",
    "redis", "cassandra", "dynamodb", "sqlite", "mariadb",
    "firebase", "elasticsearch", "neo4j", "supabase",

    # ---------------- Cloud & DevOps ----------------
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "jenkins", "ci/cd", "gitlab",
    "gitlab ci", "bitbucket", "heroku", "vercel", "netlify",
    "microservices", "linux", "nginx", "apache",

    # ---------------- Version Control & Tools ----------------
    "git", "github", "jira", "postman", "vs code", "intellij idea",
    "jupyter notebook", "google colab", "mlflow", "google sheets",
    "google data studio", "figma", "adobe xd", "confluence", "trello",

    # ---------------- Testing ----------------
    "selenium", "cypress", "jest", "mocha", "pytest", "unit testing",
    "api testing", "test automation", "junit", "postman",

    # ---------------- Machine Learning & AI ----------------
    "machine learning", "deep learning", "tensorflow", "keras",
    "pytorch", "scikit-learn", "xgboost", "opencv", "nlp",
    "computer vision", "hugging face", "transformers", "langchain",
    "openai api", "llm", "generative ai", "cnn", "rnn", "lstm",
    "bert", "spark", "pyspark", "hadoop", "airflow", "reinforcement learning",

    # ---------------- Data Analysis & Visualization ----------------
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "power bi",
    "tableau", "excel", "pivot tables", "vlookup", "power query", "dax",
    "plotly", "d3.js", "looker", "google analytics",

    # ---------------- Concepts ----------------
    "statistics", "probability", "feature engineering", "model evaluation",
    "supervised learning", "unsupervised learning", "a/b testing",
    "data cleaning", "descriptive statistics", "dashboarding",
    "kpi reporting", "oop", "object oriented programming",
    "data structures", "algorithms", "data structures and algorithms",
    "sdlc", "design patterns", "agile", "scrum", "kanban",
    "system design", "distributed systems", "data warehousing",
    "etl", "data modeling",

    # ---------------- Security / Blockchain ----------------
    "cybersecurity", "penetration testing", "ethical hacking",
    "blockchain", "solidity", "web3", "cryptography",

    # ---------------- Soft Skills ----------------
    "communication", "leadership", "teamwork", "problem solving",
    "time management", "critical thinking", "project management",
    "collaboration", "adaptability", "presentation skills",

    # ---------------- Project & Program Management ----------------
    "risk management", "stakeholder management", "waterfall",
    "ms project", "microsoft project", "team management",
    "conflict resolution", "vendor management", "resource allocation",
    "budgeting", "cost management", "quality management",
    "change management", "sprint planning", "user stories",
    "roadmap planning", "product roadmap", "gantt chart", "okrs",
    "requirements gathering", "status reporting", "trello", "confluence",

    # ---------------- Others ----------------
    "rest api", "streamlit", "ui/ux", "responsive design", "seo"
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
# Skill → Role mapping (used by /gap-report)
# ==========================================================
# ⚠️ IMPORTANT: these dict keys must exactly match the role names your
# model actually predicts — i.e. the values in label_encoder.classes_.
# Run `print(label_encoder.classes_)` once and adjust the keys below
# to match exactly (including casing), or /gap-report will 404 for
# any predicted role that isn't listed here.
#
# Skill names use .title() casing to match what extract_skills() below
# produces (e.g. "rest api" -> "Rest Api"), so alignment percentages
# come out correct instead of showing every skill as "missing".

ROLE_SKILLS = {
    "Full Stack Developer": [
        "Html", "Css", "Javascript", "React", "Node.Js", "Python", "Sql",
        "Rest Api", "Git", "Mongodb"
    ],
    "Java Developer": [
        "Java", "Oop", "Sql", "Spring Boot", "Spring", "Hibernate",
        "Maven", "Git", "Rest Api", "Junit"
    ],
    "Full Stack Java Developer": [
        "Java", "Spring Boot", "Spring", "Hibernate", "Maven", "Html",
        "Css", "Javascript", "React", "Sql", "Mysql", "Rest Api", "Git", "Oop"
    ],
    "Python Developer": [
        "Python", "Oop", "Sql", "Git", "Rest Api", "Flask", "Fastapi",
        "Django", "Pytest"
    ],
    "Frontend Developer": [
        "Html", "Css", "Javascript", "Typescript", "React", "Angular",
        "Vue", "Tailwind Css", "Git", "Responsive Design"
    ],
    "Backend Developer": [
        "Python", "Java", "Node.Js", "Sql", "Rest Api", "Graphql", "Git",
        "Docker", "Mongodb", "Microservices"
    ],
    "Web Developer": [
        "Html", "Css", "Javascript", "React", "Node.Js", "Git",
        "Responsive Design", "Bootstrap", "Rest Api"
    ],
    "Mobile Developer": [
        "Kotlin", "Swift", "Flutter", "React Native", "Android", "Ios", "Git"
    ],
    "Data Scientist": [
        "Python", "Sql", "Machine Learning", "Pandas", "Numpy",
        "Scikit-Learn", "Statistics", "Matplotlib", "Deep Learning"
    ],
    "Data Analyst": [
        "Sql", "Excel", "Power Bi", "Tableau", "Statistics",
        "Data Cleaning", "Pandas", "Python", "Plotly"
    ],
    "Data Engineer": [
        "Python", "Sql", "Docker", "Aws", "Mongodb", "Mysql",
        "Postgresql", "Spark", "Airflow", "Etl"
    ],
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "Deep Learning", "Tensorflow",
        "Pytorch", "Keras", "Sql", "Docker", "Mlflow"
    ],
    "Ai Engineer": [
        "Python", "Machine Learning", "Deep Learning", "Tensorflow",
        "Pytorch", "Nlp", "Opencv", "Transformers", "Llm"
    ],
    "Software Engineer": [
        "Java", "Python", "C++", "Data Structures", "Algorithms", "Git",
        "Oop", "System Design"
    ],
    "Software Developer": [
        "Java", "Python", "Sql", "Git", "Oop", "Data Structures",
        "Design Patterns"
    ],
    "Business Analyst": [
        "Sql", "Excel", "Power Bi", "Tableau", "Statistics", "Data Cleaning"
    ],
    "Business Intelligence Analyst": [
        "Sql", "Power Bi", "Tableau", "Excel", "Dax", "Power Query"
    ],
    "Quantitative Analyst": [
        "Python", "Sql", "Statistics", "Probability", "Excel"
    ],
    "Product Manager": [
        "Agile", "Scrum", "Jira", "Sql", "Excel", "Project Management"
    ],
    "Project Manager": [
        "Agile", "Scrum", "Jira", "Sdlc", "Project Management",
        "Communication", "Leadership", "Risk Management", "Java Core", "Swing",
        "Stakeholder Management", "Team Management", "Waterfall",
        "Sprint Planning", "Budgeting", "Change Management",
        "Conflict Resolution", "Trello", "Confluence",
        "Requirements Gathering", "Status Reporting"
    ],
    "Program Manager": [
        "Agile", "Scrum", "Jira", "Sdlc", "Excel", "Leadership"
    ],
    "Devops Engineer": [
        "Docker", "Kubernetes", "Aws", "Azure", "Terraform", "Ansible",
        "Jenkins", "Ci/Cd", "Git", "Linux"
    ],
    "Cloud Engineer": [
        "Aws", "Azure", "Gcp", "Docker", "Kubernetes", "Terraform", "Linux"
    ],
    "Site Reliability Engineer": [
        "Docker", "Kubernetes", "Aws", "Azure", "Git", "Linux", "Ci/Cd"
    ],
    "Qa Engineer": [
        "Junit", "Postman", "Selenium", "Cypress", "Sql", "Git",
        "Test Automation", "Jira"
    ],
    "Test Engineer": [
        "Junit", "Postman", "Selenium", "Cypress", "Sql", "Git",
        "Api Testing", "Jira"
    ],
    "Systems Engineer": [
        "Aws", "Azure", "Docker", "Git", "Sql", "Linux"
    ],
    "System Administrator": [
        "Aws", "Azure", "Git", "Sql", "Linux", "Bash"
    ],
    "Database Administrator": [
        "Sql", "Mysql", "Postgresql", "Oracle", "Oracle Sql", "Mongodb",
        "Data Modeling"
    ],
    "Research Scientist": [
        "Python", "Machine Learning", "Statistics", "Nlp", "Deep Learning"
    ],
    "Research Assistant": [
        "Python", "Statistics", "Data Cleaning"
    ],
    "Consultant": [
        "Excel", "Sql", "Statistics", "Power Bi", "Communication"
    ],
    "Analyst": [
        "Sql", "Excel", "Statistics", "Data Cleaning"
    ],
    "Ui/Ux Designer": [
        "Figma", "Adobe Xd", "Ui/Ux", "Responsive Design", "Html", "Css"
    ],
    "Blockchain Developer": [
        "Solidity", "Web3", "Blockchain", "Javascript", "Cryptography"
    ],
    "Cybersecurity Analyst": [
        "Cybersecurity", "Penetration Testing", "Ethical Hacking", "Linux",
        "Networking"
    ],
}

SKILL_SUGGESTIONS = {
    "Java": "Learn Java syntax, collections, exception handling and multithreading.",
    "Oop": "Practice inheritance, polymorphism, abstraction and encapsulation.",
    "Spring Boot": "Build REST APIs using Spring Boot.",
    "Spring": "Learn the Spring Framework fundamentals.",
    "Hibernate": "Learn Hibernate/JPA for ORM-based database integration.",
    "Jpa": "Learn the Java Persistence API for ORM mapping.",
    "Maven": "Learn Maven dependency management and build lifecycle.",
    "Gradle": "Learn Gradle build scripts and dependency management.",
    "Sql": "Practice joins, subqueries, normalization and database design.",
    "Git": "Practice Git branching, merging and collaborative workflows.",
    "Rest Api": "Build and consume REST APIs.",
    "Graphql": "Learn to design and query GraphQL APIs.",
    "Python": "Strengthen Python programming and advanced concepts.",
    "Html": "Practice semantic HTML and accessibility.",
    "Css": "Learn Flexbox, Grid and responsive layouts.",
    "Tailwind Css": "Practice building layouts with Tailwind's utility classes.",
    "Bootstrap": "Practice responsive layouts using Bootstrap components.",
    "Javascript": "Strengthen ES6+, DOM manipulation and asynchronous JavaScript.",
    "Typescript": "Learn static typing, interfaces and generics in TypeScript.",
    "React": "Build React applications using components and hooks.",
    "Node.Js": "Build backend services and APIs with Node.js.",
    "Angular": "Learn Angular components, services and routing.",
    "Vue": "Practice building small apps with Vue.js.",
    "Flask": "Build lightweight REST APIs using Flask.",
    "Fastapi": "Build production-ready REST APIs using FastAPI.",
    "Django": "Create a complete web application using Django.",
    "Pytest": "Practice writing test suites with pytest.",
    "Docker": "Learn containerization and Docker deployment.",
    "Kubernetes": "Learn container orchestration basics.",
    "Terraform": "Practice infrastructure-as-code with Terraform.",
    "Ansible": "Learn configuration management and automation with Ansible.",
    "Jenkins": "Set up CI/CD pipelines using Jenkins.",
    "Ci/Cd": "Learn to build automated CI/CD pipelines.",
    "Linux": "Get comfortable with the Linux command line and shell.",
    "Bash": "Practice writing shell scripts in Bash.",
    "Mongodb": "Practice CRUD operations and schema design in MongoDB.",
    "Mysql": "Practice SQL queries, joins and indexing in MySQL.",
    "Postgresql": "Practice SQL queries and schema design in PostgreSQL.",
    "Oracle": "Learn Oracle database fundamentals.",
    "Oracle Sql": "Practice PL/SQL and Oracle-specific SQL features.",
    "Data Modeling": "Practice designing normalized, scalable data models.",
    "Microservices": "Learn to design and deploy microservice architectures.",
    "Machine Learning": "Practice supervised and unsupervised ML projects.",
    "Deep Learning": "Learn neural networks using TensorFlow or PyTorch.",
    "Tensorflow": "Build and train deep learning models with TensorFlow.",
    "Pytorch": "Build and train deep learning models with PyTorch.",
    "Keras": "Practice model-building with the Keras API.",
    "Mlflow": "Track experiments and manage ML models with MLflow.",
    "Pandas": "Practice data wrangling and cleaning with Pandas.",
    "Numpy": "Get comfortable with array operations in NumPy.",
    "Scikit-Learn": "Practice building ML pipelines with scikit-learn.",
    "Statistics": "Review descriptive statistics and probability.",
    "Probability": "Review probability theory fundamentals.",
    "Matplotlib": "Practice data visualization with Matplotlib.",
    "Plotly": "Build interactive visualizations with Plotly.",
    "Nlp": "Learn text preprocessing and NLP model basics.",
    "Opencv": "Practice image processing with OpenCV.",
    "Transformers": "Learn transformer architectures for NLP/vision tasks.",
    "Llm": "Explore prompting and fine-tuning large language models.",
    "Spark": "Learn distributed data processing with Apache Spark.",
    "Airflow": "Practice orchestrating data pipelines with Airflow.",
    "Etl": "Practice building extract-transform-load data pipelines.",
    "C++": "Strengthen core C++ syntax and STL usage.",
    "Data Structures": "Practice arrays, trees, graphs, and linked lists.",
    "Algorithms": "Practice sorting, searching, and complexity analysis.",
    "System Design": "Practice designing scalable backend systems.",
    "Design Patterns": "Learn common OOP design patterns and when to use them.",
    "Excel": "Practice formulas, pivot tables, and VLOOKUP.",
    "Power Bi": "Build interactive dashboards in Power BI.",
    "Tableau": "Practice building visualizations in Tableau.",
    "Data Cleaning": "Practice handling missing/inconsistent data.",
    "Dax": "Learn DAX formulas for Power BI measures.",
    "Power Query": "Practice data transformation with Power Query.",
    "Agile": "Learn Agile principles and iterative development.",
    "Scrum": "Understand Scrum roles, ceremonies and artifacts.",
    "Jira": "Practice sprint planning and issue tracking using Jira.",
    "Sdlc": "Review the software development lifecycle stages.",
    "Project Management": "Practice planning, scheduling and tracking projects.",
    "Communication": "Practice clear written and verbal communication with stakeholders.",
    "Leadership": "Develop decision-making and team-leading skills.",
    "Aws": "Get familiar with core AWS services (EC2, S3, IAM).",
    "Azure": "Get familiar with core Azure services.",
    "Gcp": "Get familiar with core Google Cloud Platform services.",
    "Junit": "Practice writing unit tests with JUnit.",
    "Postman": "Practice testing APIs with Postman.",
    "Selenium": "Practice browser automation and UI testing with Selenium.",
    "Cypress": "Practice end-to-end testing with Cypress.",
    "Test Automation": "Build automated test suites for regression coverage.",
    "Api Testing": "Practice validating REST APIs with tools like Postman.",
    "Kotlin": "Learn Kotlin fundamentals for Android development.",
    "Swift": "Learn Swift fundamentals for iOS development.",
    "Flutter": "Build cross-platform mobile apps with Flutter.",
    "React Native": "Build cross-platform mobile apps with React Native.",
    "Android": "Learn Android app development fundamentals.",
    "Ios": "Learn iOS app development fundamentals.",
    "Responsive Design": "Practice building mobile-first, responsive layouts.",
    "Figma": "Practice designing UI mockups and prototypes in Figma.",
    "Adobe Xd": "Practice designing UI mockups and prototypes in Adobe XD.",
    "Ui/Ux": "Study usability principles and user-centered design.",
    "Solidity": "Learn smart contract development with Solidity.",
    "Web3": "Explore decentralized app development fundamentals.",
    "Blockchain": "Learn blockchain fundamentals and consensus mechanisms.",
    "Cryptography": "Review core cryptographic principles and algorithms.",
    "Cybersecurity": "Study core security principles and common attack vectors.",
    "Penetration Testing": "Practice ethical penetration testing techniques.",
    "Ethical Hacking": "Learn ethical hacking methodologies and tools.",
    "Networking": "Review core networking concepts (TCP/IP, DNS, firewalls).",
    "Risk Management": "Learn how to identify, assess and mitigate project risks.",
    "Stakeholder Management": "Practice aligning expectations across stakeholders and sponsors.",
    "Waterfall": "Understand the Waterfall methodology and when to apply it.",
    "Ms Project": "Practice scheduling and tracking projects in MS Project.",
    "Microsoft Project": "Practice scheduling and tracking projects in Microsoft Project.",
    "Team Management": "Practice delegating, motivating and coordinating a project team.",
    "Conflict Resolution": "Develop techniques for resolving team and stakeholder conflicts.",
    "Vendor Management": "Learn to manage vendor contracts, SLAs and deliverables.",
    "Resource Allocation": "Practice planning and balancing team workload across tasks.",
    "Budgeting": "Practice estimating and tracking project budgets.",
    "Cost Management": "Learn to monitor and control project costs against budget.",
    "Quality Management": "Learn quality assurance and control practices for projects.",
    "Change Management": "Learn to manage scope changes and organizational change.",
    "Sprint Planning": "Practice planning and estimating sprints in Agile teams.",
    "User Stories": "Practice writing clear, testable user stories.",
    "Roadmap Planning": "Practice building and prioritizing product/project roadmaps.",
    "Product Roadmap": "Practice building and communicating a product roadmap.",
    "Gantt Chart": "Practice building Gantt charts to visualize project timelines.",
    "Okrs": "Learn to set and track Objectives and Key Results.",
    "Requirements Gathering": "Practice eliciting and documenting stakeholder requirements.",
    "Status Reporting": "Practice writing clear project status updates for stakeholders.",
    "Trello": "Practice organizing tasks and workflows on Trello boards.",
    "Confluence": "Practice documenting project plans and specs in Confluence.",
}

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
            # Print a full traceback instead of just str(err), so the real
            # failing line is visible in the server console the next time
            # this is triggered from the frontend.
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
# Skill Gap Report Endpoint
# ==========================================================

@app.route("/gap-report", methods=["POST"])
def gap_report():
    try:
        data = request.get_json(force=True) or {}
        role = data.get("job_role")
        skills = data.get("skills", [])

        required_skills = ROLE_SKILLS.get(role)
        resolved_role = role

        # FIX: the model can predict role names (e.g. "Full Stack Java
        # Developer") that don't have an exact ROLE_SKILLS entry yet.
        # Instead of hard-failing with 404, fall back to the closest
        # known role name so the skill gap still renders. This also
        # future-proofs against any new label_encoder classes added
        # later without a matching ROLE_SKILLS key.
        if required_skills is None and role:
            close = difflib.get_close_matches(
                role, ROLE_SKILLS.keys(), n=1, cutoff=0.55
            )
            if close:
                resolved_role = close[0]
                required_skills = ROLE_SKILLS[resolved_role]

        if required_skills is None:
            return jsonify({"error": f"No skill mapping found for '{role}'."}), 404

        user_skills_normalized = {s.strip().lower() for s in skills}

        matched_skills = [
            s for s in required_skills if s.lower() in user_skills_normalized
        ]
        missing_skills = [
            s for s in required_skills if s.lower() not in user_skills_normalized
        ]

        total = len(required_skills)
        alignment = round(len(matched_skills) / total * 100, 2) if total else 0

        suggestions = [
            {
                "skill": s,
                "suggestion": SKILL_SUGGESTIONS.get(s, f"Improve your knowledge of {s}.")
            }
            for s in missing_skills
        ]

        return jsonify({
            "job_role": role,
            "your_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_alignment": alignment,
            "suggestions": suggestions
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
# FIX: bind to 0.0.0.0 (not 127.0.0.1) and read PORT from the
# environment, since Render assigns the port dynamically and its proxy
# cannot reach a server bound only to localhost. This block only runs
# when you start the app directly with `python app.py` — if you deploy
# with a Procfile using gunicorn, gunicorn's own --bind flag controls
# this instead, and this block is skipped entirely.

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)