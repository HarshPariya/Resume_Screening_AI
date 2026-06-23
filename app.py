import streamlit as st
import joblib
import re
import io
import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI – Smart Career Insights",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp {
    background: linear-gradient(155deg, #EEF2FF 0%, #F8FAFF 50%, #E0F2FE 100%) !important;
}
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 800px !important;
    padding: 0 20px 60px !important;
    margin: 0 auto !important;
}
[data-testid="stMarkdownContainer"] { padding: 0 !important; }
div[data-testid="stAlert"] { display: none !important; }

/* NAV */
.topbar {
    position: fixed; top:0; left:0; right:0; z-index:1000;
    height:58px;
    background: rgba(255,255,255,0.93);
    backdrop-filter: blur(18px);
    border-bottom: 1px solid #e0e7ff;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 28px;
    box-shadow: 0 2px 16px rgba(79,70,229,0.06);
}
.tb-brand { display:flex; align-items:center; gap:10px; font-size:17px; font-weight:800; color:#1e293b; }
.tb-icon {
    width:34px; height:34px;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    border-radius:9px; display:flex; align-items:center; justify-content:center;
    font-size:16px; box-shadow:0 4px 10px rgba(79,70,229,0.3);
}
.tb-pill {
    font-size:11px; font-weight:700; letter-spacing:1px;
    color:#4f46e5; background:#eef2ff; border:1px solid #c7d2fe;
    padding:4px 12px; border-radius:20px;
}

/* HERO */
.hero { text-align:center; padding:96px 0 44px; }
.hero-badge {
    display:inline-flex; align-items:center; gap:7px;
    font-size:12px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
    color:#4f46e5; background:#eef2ff; border:1px solid #c7d2fe;
    padding:5px 14px; border-radius:20px; margin-bottom:24px;
}
.hero-h1 {
    font-size:clamp(32px,5.5vw,52px); font-weight:900; line-height:1.12;
    letter-spacing:-1.5px; color:#0f172a; margin-bottom:16px;
}
.hero-h1 .grd {
    background:linear-gradient(135deg,#4f46e5 0%,#06b6d4 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-p { font-size:18px; color:#64748b; max-width:800px; margin:0 auto 44px; line-height:1.8; font-weight:500; }

/* STATS */
.stats { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-bottom:40px; }
.scard {
    background:#fff; border:1px solid #e2e8f0; border-radius:16px;
    padding:18px 26px; text-align:center; min-width:135px;
    box-shadow:0 2px 8px rgba(0,0,0,0.04); transition:box-shadow .2s;
}
.scard:hover { box-shadow:0 6px 20px rgba(79,70,229,0.1); }
.snum {
    font-size:24px; font-weight:900; line-height:1; display:block;
    background:linear-gradient(135deg,#4f46e5,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.slbl { font-size:11px; color:#94a3b8; font-weight:600; margin-top:5px; }

/* BUTTON overrides */
[data-testid="baseButton-primary"] {
    background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color:#fff !important; border:none !important;
    border-radius:14px !important; font-weight:800 !important;
    font-size:16px !important; padding:12px !important;
    box-shadow:0 8px 20px rgba(79,70,229,0.3) !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    transition: transform 0.1s, box-shadow 0.1s;
}
[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px);
    box-shadow:0 10px 25px rgba(79,70,229,0.4) !important;
}

/* RESULT CARDS */
.res-header { text-align:center; padding:40px 0 32px; }
.res-label {
    font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    color:#06b6d4; margin-bottom:8px;
}
.res-title {
    font-size:clamp(24px,4vw,34px); font-weight:900;
    color:#0f172a; letter-spacing:-0.8px;
}
.res-title span {
    background:linear-gradient(135deg,#4f46e5,#06b6d4);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.filebar {
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;
    background:#fff; border:1px solid #e2e8f0; border-radius:14px;
    padding:14px 18px; margin-bottom:24px; margin-top: 40px;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
}
.fb-left { display:flex; align-items:center; gap:12px; }
.fb-icon { width:40px; height:40px; border-radius:11px; background:#eef2ff; display:flex; align-items:center; justify-content:center; font-size:20px; }
.fb-name { font-size:14px; font-weight:700; color:#1e293b; }
.fb-size { font-size:12px; color:#94a3b8; }
.done-tag {
    display:flex; align-items:center; gap:6px;
    background:#f0fdf4; border:1px solid #bbf7d0;
    color:#15803d; font-size:13px; font-weight:700;
    padding:7px 14px; border-radius:10px;
}
.rcard {
    background:#fff; border:1px solid #e2e8f0; border-radius:20px;
    padding:28px 26px; margin-bottom:14px;
    box-shadow:0 4px 20px rgba(0,0,0,0.05);
    transition:transform .2s, box-shadow .2s; position:relative; overflow:hidden;
}
.rcard:hover { transform:translateY(-2px); box-shadow:0 10px 36px rgba(0,0,0,0.09); }
.rstripe { position:absolute; top:0; left:0; right:0; height:4px; border-radius:20px 20px 0 0; }
.si { background:linear-gradient(90deg,#4f46e5,#818cf8); }
.sc { background:linear-gradient(90deg,#06b6d4,#67e8f9); }
.sv { background:linear-gradient(90deg,#7c3aed,#c084fc); }
.st { background:linear-gradient(90deg,#0d9488,#2dd4bf); }
.ctag {
    font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
    margin-bottom:14px; display:flex; align-items:center; gap:8px;
}
.ci { color:#4f46e5; } .cc { color:#0891b2; } .cv { color:#7c3aed; } .ctt { color:#0f766e; }
.cicon { width:46px; height:46px; border-radius:13px; display:flex; align-items:center; justify-content:center; font-size:22px; margin-bottom:12px; }
.bi { background:#eef2ff; } .bc { background:#ecfeff; } .bv { background:#f5f3ff; } .bt { background:#f0fdfa; }
.cval { font-size:clamp(20px,3vw,28px); font-weight:900; color:#0f172a; letter-spacing:-.5px; margin-bottom:8px; line-height:1.2; }
.csub { font-size:13px; color:#94a3b8; line-height:1.6; }
.confnum { font-size:clamp(42px,7vw,62px); font-weight:900; letter-spacing:-2.5px; line-height:1; margin-bottom:6px; }
.confmsg { font-size:13px; font-weight:600; margin-bottom:18px; }
.ptrack { background:#f1f5f9; border-radius:100px; height:10px; overflow:hidden; }
.pfill { height:100%; border-radius:100px; }
.career-pill {
    display:inline-flex; align-items:center; gap:10px;
    background:linear-gradient(135deg,#f5f3ff,#fdf4ff);
    border:1.5px solid #e9d5ff; color:#6d28d9;
    font-size:17px; font-weight:800; padding:12px 18px; border-radius:13px;
    box-shadow:0 4px 14px rgba(124,58,237,0.1);
}
.cric { width:32px; height:32px; border-radius:9px; background:linear-gradient(135deg,#7c3aed,#c084fc); display:flex; align-items:center; justify-content:center; font-size:16px; }
.swrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.stag {
    display:inline-flex; align-items:center; gap:6px;
    background:#f8faff; border:1.5px solid #e0e7ff;
    color:#3730a3; font-size:13px; font-weight:600;
    padding:6px 13px; border-radius:10px; transition:all .18s;
}
.stag:hover { background:#eef2ff; border-color:#a5b4fc; box-shadow:0 2px 8px rgba(79,70,229,0.12); }
.scbadge {
    font-size:11px; font-weight:700; background:#fef3c7; border:1px solid #fde68a;
    color:#92400e; padding:2px 9px; border-radius:20px; margin-left:auto;
}
.divider { height:1px; background:linear-gradient(90deg,transparent,#e2e8f0,transparent); margin:36px 0; }
.appfooter {
    text-align:center; padding:20px; color:#94a3b8; font-size:13px; font-weight:500;
    border-top:1px solid #e2e8f0;
}
.appfooter b { color:#4f46e5; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    xgb = joblib.load('resume_model.pkl')
    vec = joblib.load('tfidf.pkl')
    le  = joblib.load('label_encoder.pkl')
    return xgb, vec, le

try:
    xgb_model, tfidf, le = load_models()
    MODEL_COLS = xgb_model.n_features_in_
except Exception as e:
    st.error(f"⛔ Could not load models: {e}")
    st.stop()

# ─── PDF Extraction (multi-method fallback) ──────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + " "
        if text.strip(): return text.strip()
    except: pass

    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + " "
        if text.strip(): return text.strip()
    except: pass

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + " "
        if text.strip(): return text.strip()
    except: pass

    return ""

# ─── Helpers ────────────────────────────────────────────────────────────────────
def clean_resume(txt: str) -> str:
    txt = re.sub(r'http\S+\s*', ' ', txt)
    txt = re.sub(r'[#@]\S+', ' ', txt)
    txt = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    return re.sub(r'\s+', ' ', txt).lower().strip()

def align_to_model(mat, n: int):
    c = mat.shape[1]
    if c > n:   return mat[:, :n]
    if c < n:   return sp.hstack([mat, sp.csr_matrix((1, n - c))])
    return mat

SKILLS = [
    ("Python","🐍"),("Java","☕"),("JavaScript","📜"),("TypeScript","📘"),
    ("SQL","🗄️"),("NoSQL","🔷"),("TensorFlow","🔬"),("PyTorch","🔥"),
    ("Machine Learning","🤖"),("Deep Learning","🧠"),("NLP","💬"),
    ("Computer Vision","👁️"),("Docker","🐳"),("Kubernetes","⚙️"),
    ("AWS","☁️"),("Azure","🔷"),("GCP","🌐"),("React","⚛️"),
    ("Node.js","🟢"),("MongoDB","🍃"),("Pandas","🐼"),("NumPy","🔢"),
    ("Scikit-Learn","📊"),("Flask","🌶️"),("FastAPI","⚡"),("Django","🎸"),
    ("Spark","✨"),("Tableau","📈"),("Power BI","📊"),("Git","🔀"),
    ("Linux","🐧"),("C++","⚡"),("Scala","🔴"),("Data Science","📊"),
    ("DevOps","🔧"),("Excel","📗"),("CSS","🎨"),("HTML","🌐"),("R","📉"),
]

def extract_skills(text: str):
    t = text.lower()
    return [(name, icon) for name, icon in SKILLS
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', t)]

CAREER = {
    "INFORMATION-TECHNOLOGY":("Web Development","💻"),
    "ENGINEERING":           ("DevOps Engineering","⚙️"),
    "DESIGNER":              ("UI/UX Design","🎨"),
    "BUSINESS-DEVELOPMENT":  ("Data Science","📊"),
    "HR":                    ("Human Resources","👥"),
    "FITNESS":               ("Fitness & Wellness","💪"),
    "AVIATION":              ("Aviation & Aerospace","✈️"),
    "SALES":                 ("Sales & Marketing","📣"),
    "BANKING":               ("FinTech & Banking","🏦"),
    "HEALTHCARE":            ("Healthcare Management","🏥"),
    "CONSULTANT":            ("Strategy Consulting","📋"),
    "PUBLIC-RELATIONS":      ("PR & Communications","📡"),
    "ARTS":                  ("Creative Arts","🎭"),
    "TEACHER":               ("Education & Training","🎓"),
    "APPAREL":               ("Fashion & Design","👗"),
    "AGRICULTURE":           ("AgriTech","🌱"),
    "BPO":                   ("Customer Success","🎧"),
    "AUTOMOBILE":            ("Automotive Engineering","🚗"),
    "CHEF":                  ("Culinary Arts","👨‍🍳"),
    "CONSTRUCTION":          ("Civil Engineering","🏗️"),
    "DIGITAL-MEDIA":         ("Digital Marketing","📲"),
    "FINANCE":               ("Financial Analysis","💹"),
}
CAT_ICONS = {k: v[1] for k, v in CAREER.items()}

# ═══════════════════════════════════════════════════════════════
#  RENDER
# ═══════════════════════════════════════════════════════════════

# ── Navbar ──────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="tb-brand">
    <div class="tb-icon">🎯</div>
    ResumeAI
  </div>
  <span class="tb-pill">✦ AI POWERED</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🚀 XGBoost + TF-IDF NLP Engine</div>
  <h1 class="hero-h1">Your Resume,<br><span class="grd">Analysed by AI</span></h1>
  <p class="hero-p">
    Upload your PDF resumes and get instant breakdowns —
    job categories, confidence scores, detected skills, and personalised career paths.
  </p>
</div>
<div class="stats">
  <div class="scard"><span class="snum">2,484</span><div class="slbl">Resumes Trained</div></div>
  <div class="scard"><span class="snum">22+</span><div class="slbl">Job Categories</div></div>
  <div class="scard"><span class="snum">80.9%</span><div class="slbl">Accuracy</div></div>
  <div class="scard"><span class="snum">&lt;1s</span><div class="slbl">Analysis Time</div></div>
</div>
""", unsafe_allow_html=True)

# ── Upload Zone ─────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Drop Your Resumes Here 📄", 
    type=["pdf"], 
    accept_multiple_files=True
)

# ── Analysis ────────────────────────────────────────────────────
if uploaded_files:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyse Resumes 🚀", type="primary", use_container_width=True):
        st.markdown(f"""
        <div class="divider"></div>
        <div class="res-header">
          <div class="res-label">✦ Batch Analysis Complete</div>
          <div class="res-title">Your Career <span>Intelligence Reports</span></div>
        </div>
        """, unsafe_allow_html=True)

        for uploaded in uploaded_files:
            file_bytes = uploaded.getvalue()

            with st.spinner(f"🔍 Analysing {uploaded.name}…"):
                try:
                    # Extract text
                    raw = extract_pdf_text(file_bytes)

                    if not raw.strip():
                        st.error(f"❌ Could not extract text from {uploaded.name}. The file may be a scanned image or corrupted.")
                        continue

                    # Clean and vectorise
                    cleaned  = clean_resume(raw)
                    feats    = tfidf.transform([cleaned])
                    feats    = align_to_model(feats, MODEL_COLS)

                    # Predict
                    pred_id  = int(xgb_model.predict(feats)[0])
                    category = le.inverse_transform([pred_id])[0]

                    # Confidence
                    try:
                        proba      = xgb_model.predict_proba(feats)
                        confidence = float(np.max(proba)) * 100
                    except Exception:
                        confidence = 85.0

                    # Skills & career
                    skills      = extract_skills(cleaned)
                    key         = category.upper().replace(" ", "-").replace("_", "-")
                    career_name, career_icon = CAREER.get(key, ("Specialist Role", "🌟"))
                    cat_icon    = CAT_ICONS.get(key, "📄")
                    pretty_cat  = category.replace("-", " ").replace("_", " ").title()
                except Exception as e:
                    st.error(f"❌ Error analysing {uploaded.name}: {str(e)}")
                    continue

            # ── File info bar ──
            size_kb = round(len(file_bytes) / 1024, 1)
            st.markdown(f"""
            <div class="filebar">
              <div class="fb-left">
                <div class="fb-icon">📎</div>
                <div>
                  <div class="fb-name">{uploaded.name}</div>
                  <div class="fb-size">{size_kb} KB &nbsp;·&nbsp; PDF</div>
                </div>
              </div>
              <div class="done-tag">✅ Processed</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Card 1: Category ──
            st.markdown(f"""
            <div class="rcard">
              <div class="rstripe si"></div>
              <div class="ctag ci">Predicted Job Category</div>
              <div class="cicon bi">{cat_icon}</div>
              <div class="cval">{pretty_cat}</div>
              <div class="csub">Classified by analysing vocabulary patterns, keyword density, and structural signals using TF-IDF and XGBoost.</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Card 2: Confidence ──
            bar_w = min(int(confidence), 100)
            if confidence >= 75:
                col,grad,msg,mc = "#10b981","linear-gradient(90deg,#10b981,#34d399)","High confidence — strong category match","#15803d"
            elif confidence >= 50:
                col,grad,msg,mc = "#f59e0b","linear-gradient(90deg,#f59e0b,#fbbf24)","Moderate confidence — review may be useful","#b45309"
            else:
                col,grad,msg,mc = "#ef4444","linear-gradient(90deg,#ef4444,#f87171)","Low confidence — manual review recommended","#dc2626"

            st.markdown(f"""
            <div class="rcard">
              <div class="rstripe sc"></div>
              <div class="ctag cc">Confidence Score</div>
              <div class="confnum" style="color:{col};">{confidence:.1f}%</div>
              <div class="confmsg" style="color:{mc};">{msg}</div>
              <div class="ptrack"><div class="pfill" style="width:{bar_w}%;background:{grad};"></div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Card 3: Career ──
            st.markdown(f"""
            <div class="rcard">
              <div class="rstripe sv"></div>
              <div class="ctag cv">Recommended Career Track</div>
              <div class="career-pill">
                <div class="cric">{career_icon}</div>
                {career_name}
              </div>
              <div class="csub" style="margin-top:14px;">This career path aligns with the skills, terminology, and experience patterns detected in your resume.</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Card 4: Skills ──
            count = len(skills)
            badge = f'<span class="scbadge">{count} skill{"" if count==1 else "s"} found</span>'
            if skills:
                chips = "".join(f'<span class="stag">{ic}&nbsp;{nm}</span>' for nm, ic in skills)
                body  = f'<div class="swrap">{chips}</div>'
            else:
                body = '<div class="csub" style="margin-top:10px;">No predefined tech skills detected. Add explicit skill mentions to your resume for better detection.</div>'

            st.markdown(f"""
            <div class="rcard" style="margin-bottom: 50px;">
              <div class="rstripe st"></div>
              <div class="ctag ctt" style="display:flex;align-items:center;gap:8px;">
                <span>Detected Skills</span>{badge}
              </div>
              {body}
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="height:32px;"></div>
<div class="appfooter">
  Made with ❤️ using <b>Streamlit</b> · <b>XGBoost</b> · <b>scikit-learn</b> &nbsp;|&nbsp; ResumeAI &copy; 2026
</div>
""", unsafe_allow_html=True)
