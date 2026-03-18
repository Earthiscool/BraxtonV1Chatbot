import os
import csv
import json
import time
import io
from datetime import datetime
import streamlit as st
import chromadb
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY         = os.environ.get("GROQ_API_KEY", "")
CHROMA_DIR           = "./chroma_db"
DOCS_DIR             = "./downloaded_docs"
FEEDBACK_FILE        = "./feedback_log.csv"
CORRECTIONS_FILE     = "./corrections.json"
SYNC_STATE_FILE      = "./sync_state.json"
import tempfile
_sa = os.environ.get("SERVICE_ACCOUNT_JSON")
if _sa:
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _tmp.write(_sa)
    _tmp.flush()
    SERVICE_ACCOUNT_FILE = _tmp.name
else:
    SERVICE_ACCOUNT_FILE = "service_account.json"
print(f"DEBUG: SERVICE_ACCOUNT_FILE = {SERVICE_ACCOUNT_FILE}, exists = {os.path.exists(SERVICE_ACCOUNT_FILE)}, SA_JSON set = {bool(os.environ.get('SERVICE_ACCOUNT_JSON'))}")


FOLDER_ID            = os.environ.get("DRIVE_FOLDER_ID", "")
ADMIN_PASSWORD       = os.environ.get("ADMIN_PASSWORD", "")
SCOPES               = ["https://www.googleapis.com/auth/drive.readonly"]
SHEET_ID          = os.environ.get("SHEET_ID", "1IBJzo1GOv8KgwmtKJRBmbgphOoI8HhSVaJXlIXPRv4E")
# ── Corrections ───────────────────────────────────────────────────────────────
def load_corrections():
    # Try Google Sheets first
    try:
        import gspread
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.worksheet("Corrections")
        rows = ws.get_all_records()
        return {row["question"]: row["answer"] for row in rows if row.get("question")}
    except Exception as e:
        print(f"CORRECTIONS READ ERROR: {e}")

    # Fallback to local JSON
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "r") as f:
            return json.load(f)
    return {}
def save_corrections(corrections):
    # ── 1. Save to local JSON ─────────────────────────────────────────────
    try:
        with open(CORRECTIONS_FILE, "w") as f:
            json.dump(corrections, f, indent=2)
    except Exception as e:
        print(f"CORRECTIONS JSON ERROR: {e}")

    # ── 2. Save to Google Sheets (separate tab) ───────────────────────────
    try:
        import gspread
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)

        # Get or create a Corrections tab
        try:
            ws = sh.worksheet("Corrections")
        except Exception:
            ws = sh.add_worksheet(title="Corrections", rows=500, cols=3)
            ws.insert_row(["question", "answer", "last_updated"], 1)

        # Clear and rewrite all corrections
        ws.clear()
        ws.insert_row(["question", "answer", "last_updated"], 1)
        for q, a in corrections.items():
            ws.append_row([q, a, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    except Exception as e:
        print(f"CORRECTIONS SHEETS ERROR: {e}")

def check_corrections(question):
    corrections = load_corrections()
    q = question.lower().strip()
    for key, value in corrections.items():
        if key.lower().strip() in q or q in key.lower().strip():
            return value
    return None

# ── Feedback ──────────────────────────────────────────────────────────────────
def log_feedback(question, answer, rating, comment="", auto_score=None):
    import re
    score = auto_score or {}

    # Strip HTML tags from answer
    clean_answer = re.sub(r'<[^>]+>', '', answer).strip()

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question, clean_answer, rating, comment,
        score.get("relevance", ""),
        score.get("groundedness", ""),
        score.get("completeness", ""),
        score.get("overall", ""),
    ]

    # ── 1. Save to local CSV ──────────────────────────────────────────────
    try:
        exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "question", "answer", "rating",
                                 "comment", "relevance", "groundedness",
                                 "completeness", "overall"])
            writer.writerow(row)
    except Exception as e:
        print(f"CSV ERROR: {e}")

    # ── 2. Save to Google Sheets ──────────────────────────────────────────
    try:
        import gspread
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.sheet1
        # Add headers if sheet is empty
        if not ws.get_all_values():
            ws.insert_row(["timestamp", "question", "answer", "rating",
                           "comment", "relevance", "groundedness",
                           "completeness", "overall"], 1)
        ws.append_row(row)
    except Exception as e:
        import traceback
        print(f"SHEETS ERROR: {e}")
        print(traceback.format_exc())
def load_feedback():
    # Try Google Sheets first
    try:
        import gspread
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.sheet1
        rows = ws.get_all_records()
        # Ensure all rows have score fields
        for row in rows:
            for col in ["relevance", "groundedness", "completeness", "overall"]:
                if col not in row:
                    row[col] = ""
        return rows
    except Exception as e:
        print(f"SHEETS READ ERROR: {e}")

    # Fallback to local CSV
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in ["relevance", "groundedness", "completeness", "overall"]:
            if col not in row:
                row[col] = ""
    return rows
# ── Auto-scoring ──────────────────────────────────────────────────────────────
def score_response(question, answer, sources):
    try:
        from groq import Groq as GroqClient
        client = GroqClient(api_key=GROQ_API_KEY)
        scoring_prompt = f"""Rate this chatbot response. Reply ONLY with valid JSON, nothing else.

Question: {question}
Answer: {answer}
Sources: {sources}

Rate each 1-5:
- relevance: did it answer the actual question?
- groundedness: is it based on real document content?
- completeness: is the answer complete?
- overall: overall quality score

Reply format exactly: {{"relevance": X, "groundedness": X, "completeness": X, "overall": X}}"""

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": scoring_prompt}],
            max_tokens=80
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return None

# ── Safety ────────────────────────────────────────────────────────────────────
SAFETY_KEYWORDS = ["emergency", "injury", "hurt", "accident", "fire", "theft",
                   "robbery", "assault", "police", "ambulance", "bleeding"]

def check_safety(prompt):
    return any(w in prompt.lower() for w in SAFETY_KEYWORDS)

# ── Prompt ────────────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    """You are Braxton Assistant for Braxton retail employees.
Answer using ONLY the documents below. Be concise. Use numbered steps for procedures.
If not found say: "I don't have that information. Please check with your manager."

DOCUMENTS:
{context_str}

QUESTION: {query_str}
ANSWER:"""
)

# ── Index ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm         = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    chroma_client     = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("client_docs")
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Braxton Assistant",
    page_icon="🦅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="collapsedControl"],[data-testid="stSidebar"],
#MainMenu, footer { display: none !important; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .main .block-container, section.main {
    background-color: #F5F1EC !important;
    font-family: 'DM Sans', sans-serif;
}

.main .block-container {
    max-width: 720px !important;
    padding: 0 1.25rem 7rem !important;
    margin: 0 auto !important;
}

.admin-header-negative { color: #DC2626 !important; font-family: 'DM Serif Display', serif; margin-bottom: 10px; }
.admin-header-active   { color: #166534 !important; font-family: 'DM Serif Display', serif; margin-bottom: 10px; }
.admin-header-scores   { color: #1D4ED8 !important; font-family: 'DM Serif Display', serif; margin-bottom: 10px; }

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
[data-testid="stExpander"] strong,
.stTextArea label, .stTextInput label, .stCaption { color: #1A1410 !important; }

[data-testid="stBottom"]     { background-color: #F5F1EC !important; }
[data-testid="stBottom"] > * { background-color: #F5F1EC !important; }

[data-testid="stChatInput"] {
    background: #FFFFFF !important; border: 1.5px solid #D5CFC8 !important;
    border-radius: 14px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #B87830 !important; box-shadow: 0 0 0 3px rgba(184,120,48,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    color: #1A1410 !important; background: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #A09080 !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: #B87830 !important; border-radius: 9px !important; border: none !important;
}

.bx-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1.75rem 0 0.75rem;
    border-bottom: 2px solid #E2DDD6; margin-bottom: 1.1rem;
}
.bx-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #C8913E, #8B5E20);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 22px; flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(140,80,20,0.2);
}
.bx-title h1 {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem;
    font-weight: 400; color: #1A1410; margin: 0; letter-spacing: -0.02em;
}
.bx-title p {
    font-size: 0.72rem; color: #9A8E80; margin: 2px 0 0;
    letter-spacing: 0.05em; text-transform: uppercase; font-weight: 600;
}

.bx-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1rem; }
.bx-pill {
    background: #FFFFFF; border: 1px solid #DDD8D0;
    border-radius: 100px; padding: 4px 12px;
    font-size: 0.72rem; color: #6A5E50; font-weight: 500;
}

[data-testid="stChatMessage"] {
    background: transparent !important; padding: 0.08rem 0 !important; gap: 10px !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.92rem !important; line-height: 1.65 !important;
    color: #1A1410 !important; margin: 0.3rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    background: #B87830 !important; border: none !important;
    border-radius: 18px 3px 18px 18px !important;
    padding: 10px 15px !important; max-width: 68% !important; margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] p { color: #FFFFFF !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    background: #FFFFFF !important; border: 1px solid #E2DDD6 !important;
    border-radius: 3px 18px 18px 18px !important;
    padding: 12px 16px !important; max-width: 80% !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 30px !important; height: 30px !important;
    border-radius: 9px !important; flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #C8913E, #7A5020) !important;
}

.bx-welcome {
    background: #FFF8EE; border: 1px solid #EDD9B0;
    border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: 1rem;
}
.bx-welcome p {
    color: #6A5030 !important; font-size: 0.87rem !important;
    line-height: 1.6 !important; margin: 0 !important;
}
.bx-welcome strong { color: #9A6820 !important; }

.bx-sources {
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid #EDE9E3; font-size: 0.7rem;
}
.bx-sources span {
    display: inline-block; background: #F5F2EE; border: 1px solid #E0DAD2;
    border-radius: 5px; padding: 2px 7px; margin: 2px 2px 2px 0; color: #8A7A6A;
}

.bx-correction-badge {
    display: inline-block; background: #EFF7EF; border: 1px solid #AADAAA;
    border-radius: 5px; padding: 2px 9px; font-size: 0.7rem; color: #3A7A3A; margin-bottom: 6px;
}
.bx-alert {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 3px solid #DC2626; border-radius: 10px;
    padding: 11px 15px; font-size: 0.87rem; color: #991B1B;
}

.stButton > button {
    background: #FFFFFF !important; border: 1.5px solid #C8B8A8 !important;
    color: #2A1E14 !important; border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 500 !important; padding: 7px 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    cursor: pointer !important; transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #FFF3E0 !important; border-color: #B87830 !important;
    color: #7A4A10 !important; box-shadow: 0 2px 6px rgba(184,120,48,0.2) !important;
}

.admin-metric {
    background: #FFF8EE; border: 1px solid #EDD9B0;
    border-radius: 10px; padding: 0.8rem 1rem; text-align: center;
}
.admin-metric-label {
    font-size: 0.72rem; color: #9A7A40;
    text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 4px;
}
.admin-metric-value { font-size: 1.8rem; font-weight: 700; color: #2A1E14; line-height: 1; }

.score-pill {
    display: inline-block; border-radius: 6px; padding: 2px 8px;
    font-size: 0.72rem; font-weight: 600; margin: 2px;
}
.score-good  { background: #DCFCE7; color: #166534; }
.score-mid   { background: #FEF9C3; color: #854D0E; }
.score-bad   { background: #FEE2E2; color: #991B1B; }

.stTextInput > div > div > input {
    background: #FFFFFF !important; border: 1.5px solid #D0C8C0 !important;
    border-radius: 8px !important; color: #2A1E14 !important;
    font-family: 'DM Sans', sans-serif !important; padding: 8px 12px !important;
}

.bx-rated { font-size: 0.75rem; color: #9A8A7A; margin: 4px 0 0; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #EDE9E3; }
::-webkit-scrollbar-thumb { background: #C0B0A0; border-radius: 4px; }

@media (max-width: 768px) {
    .main .block-container { padding: 0 0.9rem 6rem !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] { max-width: 85% !important; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] { max-width: 90% !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "messages":         [],
    "feedback_given":   {},
    "admin_mode":       False,
    "show_admin_login": False,
    "pending_feedback": None,
    "pending_score":    None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Score pill helper ─────────────────────────────────────────────────────────
def score_pill(label, value):
    if not value:
        return ""
    v = int(float(value))
    cls = "score-good" if v >= 4 else ("score-mid" if v >= 3 else "score-bad")
    return f'<span class="score-pill {cls}">{label}: {v}/5</span>'

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.admin_mode:
    col_title, col_back = st.columns([4, 1])
    with col_title:
        st.markdown("""
        <div style="padding:1.75rem 0 0.75rem;">
            <h2 style="font-family:'DM Serif Display',serif;font-size:1.5rem;
                font-weight:400;color:#1A1410;margin:0;">⚙️ Admin Panel</h2>
            <p style="font-size:0.72rem;color:#9A8E80;margin:2px 0 0;
                letter-spacing:0.05em;text-transform:uppercase;font-weight:600;">
                Feedback, Scores & Corrections</p>
        </div>""", unsafe_allow_html=True)
    with col_back:
        st.markdown("<div style='padding-top:2rem;'>", unsafe_allow_html=True)
        if st.button("← Back to chat"):
            st.session_state.admin_mode = False
            st.rerun()
        if st.button("🧪 Test Sheets Connection"):
            try:
                import gspread

                creds = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE,
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )
                gc = gspread.authorize(creds)
                sh = gc.open_by_key(SHEET_ID)
                ws = sh.sheet1
                ws.append_row(["TEST", "test question", "test answer", "test", "", "", "", "", ""])
                st.success(f"✅ Connected! Sheet: {sh.title}, Tab: {ws.title}")
            except Exception as e:
                import traceback

                st.error(f"❌ Failed: {e}")
                st.code(traceback.format_exc())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:2px solid #E2DDD6;margin:0 0 1.2rem;'>",
                unsafe_allow_html=True)

    feedback_data = load_feedback()
    corrections   = load_corrections()
    total         = len(feedback_data)
    thumbs_up     = sum(1 for f in feedback_data if f.get("rating") == "👍")
    sat           = round((thumbs_up / total * 100)) if total > 0 else 0
    scored        = [f for f in feedback_data if f.get("overall")]
    avg_overall   = round(sum(float(f["overall"]) for f in scored) / len(scored), 1) if scored else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="admin-metric"><div class="admin-metric-label">Total Feedback</div><div class="admin-metric-value">{total}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="admin-metric"><div class="admin-metric-label">Satisfaction</div><div class="admin-metric-value">{sat}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="admin-metric"><div class="admin-metric-label">Avg AI Score</div><div class="admin-metric-value">{avg_overall}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="admin-metric"><div class="admin-metric-label">Corrections</div><div class="admin-metric-value">{len(corrections)}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.info("To sync new documents, run `python ingest.py` in your terminal then push chroma_db to GitHub.")
    st.markdown("---")

    st.markdown('<h3 class="admin-header-scores">📊 Auto-Score Summary</h3>', unsafe_allow_html=True)
    if not scored:
        st.info("No auto-scores yet — scores are generated silently on each response.")
    else:
        avg_rel  = round(sum(float(f.get("relevance",  0)) for f in scored if f.get("relevance"))  / len(scored), 1)
        avg_gnd  = round(sum(float(f.get("groundedness",0)) for f in scored if f.get("groundedness"))/ len(scored), 1)
        avg_comp = round(sum(float(f.get("completeness",0)) for f in scored if f.get("completeness"))/ len(scored), 1)
        st.markdown(
            f'{score_pill("Relevance", avg_rel)} {score_pill("Groundedness", avg_gnd)} '
            f'{score_pill("Completeness", avg_comp)} {score_pill("Overall", avg_overall)}',
            unsafe_allow_html=True
        )
        st.caption(f"Based on {len(scored)} scored responses")
        low = sorted([f for f in scored if float(f.get("overall", 5)) <= 3],
                     key=lambda x: float(x.get("overall", 5)))[:5]
        if low:
            st.markdown("**Lowest scoring responses:**")
            for item in low:
                with st.expander(f"⚠️ Score {item.get('overall')}/5 — {item['question'][:60]}"):
                    st.markdown(f"**Q:** {item['question']}")
                    st.markdown(f"**A:** {item['answer'][:300]}...")
                    st.markdown(
                        score_pill("Relevance", item.get("relevance")) +
                        score_pill("Groundedness", item.get("groundedness")) +
                        score_pill("Completeness", item.get("completeness")),
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<h3 class="admin-header-negative">👎 Negative Feedback</h3>', unsafe_allow_html=True)
    bad = [f for f in feedback_data if f.get("rating") == "👎"]
    if not bad:
        st.info("No negative feedback yet — great sign!")
    else:
        for i, item in enumerate(reversed(bad[-20:])):
            with st.expander(f"❌  {item['question'][:80]}  —  {item['timestamp']}"):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**AI Answer:** {item['answer'][:400]}...")
                if item.get("comment"):
                    st.markdown(f"**Employee comment:** {item['comment']}")
                if item.get("overall"):
                    st.markdown(
                        score_pill("Relevance",    item.get("relevance")) +
                        score_pill("Groundedness", item.get("groundedness")) +
                        score_pill("Completeness", item.get("completeness")) +
                        score_pill("Overall",      item.get("overall")),
                        unsafe_allow_html=True)
                ct = st.text_area("Type the correct answer:", key=f"ct_{i}",
                    placeholder="This will override the AI next time this question is asked.")
                if st.button("💾 Save Correction", key=f"sv_{i}"):
                    if ct.strip():
                        corrections[item["question"]] = ct.strip()
                        save_corrections(corrections)
                        st.success("✓ Saved! AI will use this answer next time.")

    st.markdown("---")
    st.markdown('<h3 class="admin-header-active">✅ Active Corrections</h3>', unsafe_allow_html=True)
    corrections = load_corrections()
    if not corrections:
        st.info("No corrections saved yet.")
    else:
        for q, a in list(corrections.items()):
            with st.expander(f"✅  {q[:80]}"):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                if st.button("🗑️ Delete", key=f"dl_{q[:15]}"):
                    del corrections[q]
                    save_corrections(corrections)
                    st.rerun()

    st.markdown("---")
    if feedback_data:
        st.markdown("### 📥 Download Feedback Log")
        import io as _io

        output = _io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "question", "answer", "rating",
                         "comment", "relevance", "groundedness", "completeness", "overall"])
        for row in feedback_data:
            writer.writerow([
                row.get("timestamp", ""),
                row.get("question", ""),
                row.get("answer", ""),
                row.get("rating", ""),
                row.get("comment", ""),
                row.get("relevance", ""),
                row.get("groundedness", ""),
                row.get("completeness", ""),
                row.get("overall", ""),
            ])
        st.download_button("📥 Download CSV", output.getvalue(),
                           file_name="braxton_feedback.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT UI
# ══════════════════════════════════════════════════════════════════════════════
else:
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown("""
        <div class="bx-header">
            <div class="bx-logo">🦅</div>
            <div class="bx-title">
                <h1>Braxton Assistant</h1>
                <p>Internal Operations Guide</p>
            </div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='padding-top:1.9rem;text-align:right;'>", unsafe_allow_html=True)
        if st.button("⚙️ Admin"):
            st.session_state.show_admin_login = not st.session_state.show_admin_login
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.show_admin_login and not st.session_state.admin_mode:
        with st.container():
            st.markdown("""<div style="background:#FFFFFF;border:1.5px solid #D5CFC8;
                border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.8rem;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);">""", unsafe_allow_html=True)
            st.markdown("**Admin Login**")
            pwd = st.text_input("Password", type="password", key="admin_pwd",
                label_visibility="collapsed", placeholder="Enter admin password…")
            al1, al2 = st.columns([1, 3])
            with al1:
                if st.button("Login →", key="do_login"):
                    if pwd == ADMIN_PASSWORD:
                        st.session_state.admin_mode       = True
                        st.session_state.show_admin_login = False
                        st.rerun()
                    else:
                        st.error("Wrong password")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="bx-pills">
        <span class="bx-pill">Opening & Closing</span>
        <span class="bx-pill">POS Troubleshooting</span>
        <span class="bx-pill">Inventory & Receiving</span>
        <span class="bx-pill">New Hire Onboarding</span>
        <span class="bx-pill">Company Policies</span>
    </div>""", unsafe_allow_html=True)

    index = load_index()

    _, clr = st.columns([6, 1])
    with clr:
        if st.button("🗑️ Clear"):
            st.session_state.messages       = []
            st.session_state.feedback_given = {}
            st.session_state.pending_score  = None
            st.rerun()

    if not st.session_state.messages:
        st.markdown("""
        <div class="bx-welcome"><p>
            Ask me anything about <strong>store procedures</strong>,
            <strong>POS system issues</strong>, <strong>inventory</strong>,
            or <strong>company policies</strong>. I answer directly from
            Braxton's official documents. You can ask follow-up questions too.
        </p></div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Feedback buttons — render immediately after last assistant message
    if st.session_state.messages:
        last     = st.session_state.messages[-1]
        last_idx = len(st.session_state.messages) - 1
        msg_key  = f"msg_{last_idx}"

        if last["role"] == "assistant" and not last.get("is_safety"):
            already = st.session_state.feedback_given.get(msg_key)
            if already:
                st.markdown(f'<p class="bx-rated">You rated this {already} — thanks!</p>',
                            unsafe_allow_html=True)
            else:
                fb1, fb2, _ = st.columns([2, 2, 7])
                with fb1:
                    if st.button("👍 Helpful", key=f"up_{last_idx}"):
                        q = st.session_state.messages[last_idx-1]["content"] if last_idx > 0 else ""
                        log_feedback(q, last["content"], "👍",
                                     auto_score=st.session_state.get("pending_score"))
                        st.session_state.feedback_given[msg_key] = "👍"
                        st.session_state.pending_score = None
                        st.rerun()
                with fb2:
                    if st.button("👎 Not helpful", key=f"dn_{last_idx}"):
                        st.session_state.pending_feedback = {
                            "msg_key":  msg_key,
                            "question": st.session_state.messages[last_idx-1]["content"] if last_idx > 0 else "",
                            "answer":   last["content"],
                        }
                        st.rerun()

    if st.session_state.pending_feedback:
        pf = st.session_state.pending_feedback
        with st.container():
            st.markdown("""<div style="background:#FFFFFF;border:1.5px solid #D5CFC8;
                border-radius:12px;padding:1rem 1.25rem;margin:0.5rem 0;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">""", unsafe_allow_html=True)
            st.markdown("**What was wrong with this answer?** *(optional)*")
            comment = st.text_input("c", placeholder="e.g. Missing steps, wrong info…",
                label_visibility="collapsed", key="fb_comment")
            fc1, fc2, _ = st.columns([1, 1, 5])
            with fc1:
                if st.button("Submit", key="fb_sub"):
                    log_feedback(pf["question"], pf["answer"], "👎", comment,
                                 auto_score=st.session_state.get("pending_score"))
                    st.session_state.feedback_given[pf["msg_key"]] = "👎"
                    st.session_state.pending_feedback = None
                    st.session_state.pending_score    = None
                    st.rerun()
            with fc2:
                if st.button("Skip", key="fb_skip"):
                    log_feedback(pf["question"], pf["answer"], "👎",
                                 auto_score=st.session_state.get("pending_score"))
                    st.session_state.feedback_given[pf["msg_key"]] = "👎"
                    st.session_state.pending_feedback = None
                    st.session_state.pending_score    = None
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about procedures, POS issues, policies…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. Safety
            if check_safety(prompt):
                ans = """<div class="bx-alert">
                    ⚠️ <strong>Contact your manager immediately.</strong><br>
                    Refer to the <em>BAW Incident Report</em> document.
                </div>"""
                st.markdown(ans, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ans, "is_safety": True})
                st.rerun()

            # 2. Corrections
            elif (correction := check_corrections(prompt)):
                ans = f'<div class="bx-correction-badge">✓ Verified answer</div>\n\n{correction}'
                st.markdown(ans, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()

            # 3. AI
            else:
                with st.spinner("Searching…"):
                    start_time = time.time()

                    followup_words = ["what about", "and then", "after that",
                                      "how about", "what if", "tell me more", "explain more",
                                      "what is it", "what does it", "how does it", "why does it",
                                      " it ", " it?", " it.", "close it", "open it",
                                      "do it", "use it", "what is that", "how do i do that",
                                      "what about that", "and that", "when do i do it"]
                    prompt_padded = f" {prompt.lower()} "
                    is_followup = (
                        any(w in prompt_padded for w in followup_words)
                        and len(st.session_state.messages) > 2
                    )

                    if is_followup:
                        last_user_msg = ""
                        for msg in reversed(st.session_state.messages[:-1]):
                            if msg["role"] == "user":
                                last_user_msg = msg["content"]
                                break
                        enriched_prompt = f"{prompt} (context: previous topic was '{last_user_msg}')"
                    else:
                        enriched_prompt = prompt

                    qe       = index.as_query_engine(similarity_top_k=4, text_qa_template=QA_PROMPT)
                    response = qe.query(enriched_prompt)
                    answer   = str(response).strip()

                    if not answer or answer in ["None", "Empty Response"]:
                        answer = "I don't have that information. Please check with your manager."

                    sources_list = []
                    sources_html = ""
                    if response and hasattr(response, "source_nodes") and response.source_nodes:
                        sources_list = list(set([
                            node.metadata.get("file_name", "")
                            .replace(".txt", "").replace(".csv", "")
                            for node in response.source_nodes
                            if node.metadata.get("file_name")
                        ]))
                        if sources_list:
                            badges = "".join([f"<span>{s}</span>" for s in sources_list])
                            sources_html = f'<div class="bx-sources">📄 {badges}</div>'

                    auto_score = score_response(prompt, answer, ", ".join(sources_list))
                    st.session_state.pending_score = auto_score

                elapsed = round(time.time() - start_time, 2)
                st.markdown(answer)
                if sources_html:
                    st.markdown(sources_html, unsafe_allow_html=True)
                st.markdown(
                    f'<p style="font-size:0.7rem;color:#B0A090;margin-top:4px;">⏱ {elapsed}s</p>',
                    unsafe_allow_html=True)

                full = answer + (f"\n\n{sources_html}" if sources_html else "")
                st.session_state.messages.append({"role": "assistant", "content": full})
                st.rerun()