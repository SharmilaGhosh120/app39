import os
import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import logging
import time
import uuid
import re
import bcrypt

# Lightweight imports for fallback embeddings/search
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try optional heavy imports (faiss, sentence-transformers). Provide safe fallbacks.
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SBERT_AVAILABLE = False

# Basic configuration and logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv('DB_PATH', 'kyra.db')
KYRA_API_URL = os.getenv('KYRA_API_URL', 'http://kyra.kyras.in:8000/student-query')  # used as fallback

# --------------------- Embedding & Vector Store Manager ---------------------
class VectorStore:
    def __init__(self, dimension=384):
        self.doc_texts = []
        self.dimension = dimension
        self.use_faiss = False
        self.faiss_index = None
        self.embeddings = None  # numpy array (n_docs, dim) used in non-faiss fallback
        self.tfidf = None  # sklearn vectorizer fallback
        self.sbert = None

        if SBERT_AVAILABLE:
            try:
                self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = self.sbert.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning("SentenceTransformer load failed, will use TF-IDF fallback. %s", str(e))
                self.sbert = None

        if FAISS_AVAILABLE and SBERT_AVAILABLE and self.sbert is not None:
            try:
                self.faiss_index = faiss.IndexFlatL2(self.dimension)
                self.use_faiss = True
                logger.info("Using FAISS + SBERT for vector store.")
            except Exception as e:
                logger.warning("Failed to initialize FAISS. Falling back. %s", str(e))
                self.faiss_index = None
                self.use_faiss = False

    def add_docs(self, docs):
        docs = [d for d in docs if isinstance(d, str) and d.strip()]
        if not docs:
            return
        self.doc_texts.extend(docs)
        if self.sbert is not None:
            try:
                emb = np.array(self.sbert.encode(docs, convert_to_tensor=False)).astype('float32')
            except Exception as e:
                logger.warning("SBERT encoding failed; falling back to TF-IDF. %s", str(e))
                emb = None
                self.sbert = None

            if emb is not None and self.use_faiss and self.faiss_index is not None:
                try:
                    self.faiss_index.add(emb)
                except Exception as e:
                    logger.warning("Adding to FAISS failed: %s", str(e))
                    self.use_faiss = False
                    # fall through to embeddings fallback
            if not self.use_faiss:
                if self.embeddings is None:
                    self.embeddings = emb
                else:
                    self.embeddings = np.vstack([self.embeddings, emb])
        else:
            # TF-IDF fallback: rebuild TF-IDF for all docs
            self.tfidf = TfidfVectorizer(stop_words='english').fit(self.doc_texts)
            self.embeddings = self.tfidf.transform(self.doc_texts).toarray().astype('float32')

    def search(self, query, top_k=5):
        if len(self.doc_texts) == 0:
            return []
        if self.sbert is not None and (self.use_faiss and self.faiss_index is not None):
            q_emb = np.array(self.sbert.encode([query], convert_to_tensor=False)).astype('float32')
            D, I = self.faiss_index.search(q_emb, min(top_k, self.faiss_index.ntotal))
            indices = [int(i) for i in I[0] if i != -1]
            return [self.doc_texts[i] for i in indices]
        elif self.embeddings is not None:
            if self.sbert is not None:
                q_emb = np.array(self.sbert.encode([query], convert_to_tensor=False)).astype('float32')
            else:
                q_emb = self.tfidf.transform([query]).toarray().astype('float32')
            sims = cosine_similarity(q_emb, self.embeddings)[0]
            top_idx = sims.argsort()[::-1][:top_k]
            return [self.doc_texts[i] for i in top_idx if i < len(self.doc_texts)]
        else:
            return []

# --------------------- Initialize Vector Store with some docs ---------------------
vector_store = VectorStore()
initial_docs = [
    "To write an internship resume, highlight your skills, projects, and achievements relevant to the role.",
    "Best final-year AI projects include AI chatbots, recommendation systems, and computer vision apps.",
    "To prepare for an interview, research the company, practice common questions, and prepare your own questions.",
    "Skills for cybersecurity include networking, programming, threat analysis, and cryptography."
]
vector_store.add_docs(initial_docs)

# --------------------- Authentication Helpers ---------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False

def is_valid_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

# --------------------- Database functions ---------------------
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    name TEXT,
                    role TEXT,
                    password_hash TEXT
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    email TEXT,
                    name TEXT,
                    project_title TEXT,
                    question TEXT,
                    response TEXT,
                    retrieved_docs TEXT,
                    timestamp TEXT,
                    feedback_rating INTEGER
                )
            ''')
            default_admin = ('admin@college.edu', 'Jane Admin', 'admin', hash_password('default123'))
            c.execute('INSERT OR IGNORE INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)', default_admin)
            conn.commit()
    except sqlite3.Error as e:
        logger.error("Database initialization error: %s", str(e))
        raise

def register_user(email, name, role, password):
    if not is_valid_email(email):
        return False, "Invalid email format."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)', (email, name, role, hash_password(password)))
            conn.commit()
        return True, "Registered successfully."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    except Exception as e:
        logger.error("Error registering user: %s", str(e))
        return False, "Registration failed."

def authenticate(email, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('SELECT password_hash, name, role FROM users WHERE email = ?', (email,))
            row = c.fetchone()
            if not row:
                return None, "Email not found."
            stored_hash, name, role = row[0], row[1], row[2]
            if check_password(password, stored_hash):
                return {"email": email, "name": name, "role": role}, None
            else:
                return None, "Incorrect password."
    except Exception as e:
        logger.error("Authentication error: %s", str(e))
        return None, "Authentication failed."

def save_query_record(email, name, project_title, question, response, retrieved_docs):
    try:
        qid = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO queries (query_id, email, name, project_title, question, response, retrieved_docs, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                      (qid, email, name, project_title, question, response, retrieved_docs, timestamp))
            conn.commit()
    except Exception as e:
        logger.error("Failed to save query record: %s", str(e))

# --------------------- RAG / Query response pipeline ---------------------
def kyra_response(email, query_text):
    # simple rule-based example response
    if query_text.strip().lower() == "skills needed for cybersecurity?":
        return ("Key skills for cybersecurity: Networking, OS, security tools, programming (Python), threat analysis, incident response, cryptography."), "rule-based"

    # try vector retrieval
    try:
        retrieved = vector_store.search(query_text, top_k=5)
        if retrieved:
            context = " ".join(retrieved)
            # simple answer compose
            answer = f"Based on retrieved documents: {context}"
            return answer, context
    except Exception as e:
        logger.error("Vector retrieval failed: %s", str(e))

    # fallback to external API (if reachable)
    try:
        payload = {"student_id": email, "query": query_text}
        resp = requests.post(KYRA_API_URL, json=payload, timeout=4)
        if resp.status_code == 200:
            return resp.json().get("response", "No response from Ky'ra API."), "api"
    except Exception as e:
        logger.info("Kyra API unavailable or failed: %s", str(e))

    return "Sorry — I couldn't find a confident answer. Try rephrasing your question.", ""

# --------------------- Streamlit UI ---------------------
def main():
    st.set_page_config(page_title="RAG-Enhanced Technical Q&A - Ky'ra", layout="wide")
    st.title("RAG-Enhanced Technical Q&A System (Ky'ra)")
    st.caption("Robust app with FAISS+SBERT if available, otherwise TF-IDF fallback.")

    init_db()

    menu = ["Home", "Register", "Login"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Please login or register from the sidebar to ask questions.")

    elif choice == "Register":
        st.subheader("Create an account")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        role = st.selectbox("Role", ["student", "mentor", "admin"])
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        if st.button("Register"):
            if not name or not email or not password:
                st.warning("Please fill all fields.")
            elif password != password2:
                st.warning("Passwords do not match.")
            else:
                ok, msg = register_user(email.strip().lower(), name.strip(), role, password)
                if ok:
                    st.success(msg + " You can now login.")
                else:
                    st.error(msg)

    elif choice == "Login":
        st.subheader("Login to Ky'ra")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user, err = authenticate(email.strip().lower(), password)
            if err:
                st.error(err)
            else:
                st.success(f"Welcome, {user['name']} ({user['role']})")
                # Session state for logged-in user
                st.session_state['user'] = user

    # If logged in, show Q&A interface
    if 'user' in st.session_state:
        user = st.session_state['user']
        st.sidebar.info(f"Logged in as: {user['name']} ({user.get('email','N/A')})")
        st.header("Ask a technical question")
        question = st.text_area("Your question", height=120)
        if st.button("Submit question"):
            if not question or question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding an answer..."):
                    response, retrieved = kyra_response(user.get('email', 'anonymous'), question)
                    st.markdown("**Response:**")
                    st.write(response)
                    if retrieved:
                        st.markdown("**Retrieved documents / context:**")
                        if isinstance(retrieved, (list, tuple)):
                            for d in retrieved:
                                st.write("- " + d)
                        else:
                            st.write(retrieved)
                    # Save record
                    save_query_record(user.get('email', 'anonymous'), user.get('name', ''), "", question, response, str(retrieved))

        st.markdown("---")
        st.subheader("Add documents to knowledge base (optional)")
        new_doc = st.text_area("Paste document text to add to KB", height=120)
        if st.button("Add document to KB"):
            if new_doc and new_doc.strip():
                vector_store.add_docs([new_doc.strip()])
                st.success("Added to vector store. New documents will be used for retrieval.")

        st.markdown("---")
        st.subheader("Your recent queries")
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query("SELECT timestamp, question, response FROM queries WHERE email = ? ORDER BY timestamp DESC LIMIT 10", conn, params=(user.get('email', 'anonymous'),))
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("No previous queries found.")
        except Exception as e:
            st.error("Failed to load query history.")

if __name__ == '__main__':
    main()
