import streamlit as st
from docx import Document
import PyPDF2
import re
from openai import OpenAI
import numpy as np
import sys
import io

# --- 0. ENCODING ХАМГААЛАЛТ ---
# Сервер дээр кирилл үсэг уншихад ASCII алдаа гарахаас сэргийлнэ
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 1. ТОХИРГОО ---
try:
    # Streamlit Secrets-ээс Key-г унших
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("🔑 OpenAI API Key тохируулагдаагүй байна. Settings -> Secrets хэсэгт OPENAI_API_KEY нэрээр нэмнэ үү.")

# --- BACKGROUND CSS ---
# --- 2. BACKGROUND CSS ---
st.set_page_config(page_title="Way Academy RAG", layout="wide", page_icon="🎓")
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
 
/* ── Үндсэн хуудас ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}
 
[data-testid="stHeader"] {
    background: transparent !important;
}
 
/* ── Хөдөлгөөнт томьёонуудын canvas ── */
.formula-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
 
.formula {
    position: absolute;
    font-family: 'Space Mono', monospace;
    color: rgba(99, 179, 237, 0.12);
    font-size: clamp(10px, 1.2vw, 16px);
    white-space: nowrap;
    animation: floatUp linear infinite;
    user-select: none;
}
 
@keyframes floatUp {
    0%   { transform: translateY(110vh) rotate(-5deg); opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { transform: translateY(-10vh) rotate(5deg); opacity: 0; }
}
 
/* ── Gradient mesh background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 20%, rgba(37, 99, 235, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 80%, rgba(124, 58, 237, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(16, 185, 129, 0.04) 0%, transparent 70%),
        #0a0e1a;
    pointer-events: none;
    z-index: 0;
}
 
/* ── Grid overlay ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}
 
/* ── Контент давхарга ── */
.block-container {
    position: relative;
    z-index: 1;
    max-width: 1100px !important;
    padding-top: 2rem !important;
}
 
/* ── Гарчиг ── */
h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: clamp(1.8rem, 3vw, 2.8rem) !important;
    background: linear-gradient(135deg, #63b3ed 0%, #9f7aea 50%, #68d391 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}
 
h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #63b3ed !important;
}
 
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 20, 35, 0.95) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
    backdrop-filter: blur(20px);
}
 
[data-testid="stSidebar"] * {
    color: #e8eaf0 !important;
}
 
/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(99,179,237,0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s;
}
 
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.6) !important;
}
 
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}
 
.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
 
.stButton > button:hover::after { opacity: 1; }
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.4) !important;
}
 
/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.3s;
}
 
.stTextInput > div > div > input:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}
 
/* ── Expander (илэрц карт) ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    transition: border-color 0.3s, transform 0.2s;
}
 
[data-testid="stExpander"]:hover {
    border-color: rgba(99,179,237,0.35) !important;
    transform: translateX(3px);
}
 
/* ── Radio, slider ── */
.stRadio label, .stSlider label { color: #a0aec0 !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #e8eaf0 !important; }
 
/* ── Divider ── */
hr {
    border-color: rgba(99,179,237,0.15) !important;
    margin: 1.5rem 0 !important;
}
 
/* ── Success/Error messages ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
}
 
/* ── Spinner ── */
.stSpinner > div { border-top-color: #63b3ed !important; }
 
/* ── Number input ── */
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}
 
/* ── Caption ── */
.stCaption { color: #718096 !important; font-size: 0.78rem !important; }
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb {
    background: rgba(99,179,237,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.5); }
 
/* ── Pulse animation for embedding button ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,179,237,0.4); }
    50%       { box-shadow: 0 0 20px 6px rgba(99,179,237,0.2); }
}
 
.pulse-btn .stButton > button {
    animation: pulse-glow 2.5s infinite;
}
</style>
 
<!-- ── Хөдөлгөөнт томьёонуудын давхарга ── -->
<div class="formula-canvas" id="formulaCanvas"></div>
 
<script>
const formulas = [
    "cos_sim(a,b) = a·b / ‖a‖‖b‖",
    "embedding(x) ∈ ℝ¹⁵³⁶",
    "argmax score(q, dᵢ)",
    "RAG = Retrieve + Generate",
    "‖v‖ = √(Σvᵢ²)",
    "P(answer | context, query)",
    "TF-IDF(t,d) = tf × log(N/df)",
    "sim(q, c) > threshold",
    "∑ wᵢ · xᵢ = ŷ",
    "chunk_size = 512 tokens",
    "top_k = argmax(cosine(q, D))",
    "GPT(prompt) → completion",
    "vec(word) ∈ Rⁿ",
    "∇L = ∂Loss/∂θ",
    "attention(Q,K,V) = softmax(QKᵀ/√d)V",
    "context_window = 128k",
    "Σ p(xᵢ) = 1",
];
 
const canvas = document.getElementById('formulaCanvas');
const count = 18;
 
for (let i = 0; i < count; i++) {
    const el = document.createElement('div');
    el.className = 'formula';
    el.textContent = formulas[i % formulas.length];
    el.style.left = (Math.random() * 95) + 'vw';
    el.style.animationDuration = (22 + Math.random() * 20) + 's';
    el.style.animationDelay = (-Math.random() * 30) + 's';
    el.style.fontSize = (10 + Math.random() * 8) + 'px';
    el.style.opacity = (0.06 + Math.random() * 0.1);
    canvas.appendChild(el);
}
</script>
""", unsafe_allow_html=True)


def get_embedding(text, model="text-embedding-3-small"):
    # Текстийг цэвэрлэх ба Unicode-д найдвартай шилжүүлэх
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')
    
    text = text.replace("\n", " ").strip()
    
    if not text:
        return [0.0] * 1536 # Хоосон текст байвал тэг вектор буцаана
        
    try:
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        st.error(f"Embedding хийхэд алдаа гарлаа (API Key эсвэл Төлбөр шалгана уу): {e}")
        raise e

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 2. УНИВЕРСАЛ ФАЙЛ УНШИГЧ ---
def read_single_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    try:
        if file_type == 'docx':
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        elif file_type == 'txt':
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"❌ '{uploaded_file.name}' файлыг уншихад алдаа: {e}")
    return text

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Way Academy RAG", layout="wide")
st.title("🤖 Advanced Multi-File RAG Tool")

# Sidebar - Тохиргоо
with st.sidebar:
    st.header("⚙️ Конфигураци")
    chunk_method = st.radio("Chunk хийх арга:", ("Өгүүлбэрээр", "Параграфаар"))
    top_k = st.number_input("Хэдэн илэрц харуулах вэ?", 1, 10, 5)
    st.divider()
    if st.button("Шинээр эхлэх (Reset)"):
        st.session_state.clear()
        st.rerun()

# Файл оруулах хэсэг
uploaded_files = st.file_uploader(
    "Файлуудаа оруулна уу (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    # 1. Текст боловсруулах (Chunking)
    if "chunks" not in st.session_state:
        all_chunks = []
        all_sources = []
        
        with st.spinner("Файлуудыг уншиж байна..."):
            for uploaded_file in uploaded_files:
                content = read_single_file(uploaded_file)
                
                if chunk_method == "Параграфаар":
                    file_chunks = [p.strip() for p in content.split('\n') if p.strip()]
                else:
                    # Кирилл цэг болон асуултын тэмдэгээр салгах
                    file_chunks = re.split(r'(?<=[.!?]) +', content)
                    file_chunks = [s.strip() for s in file_chunks if s.strip()]
                
                for c in file_chunks:
                    all_chunks.append(c)
                    all_sources.append(uploaded_file.name)
            
            st.session_state.chunks = all_chunks
            st.session_state.sources = all_sources
            st.success(f"✅ Нийт {len(all_chunks)} хэсэг мэдээлэл бэлэн боллоо.")

    # 2. Embedding (Вектор сан үүсгэх)
    if "embeddings" not in st.session_state and "chunks" in st.session_state:
        if st.button("🚀 Мэдлэгийн сан үүсгэх (Embedding)"):
            with st.spinner("Вектор сан үүсгэж байна... (API Key шалгаж байна)"):
                try:
                    embs = [get_embedding(c) for c in st.session_state.chunks]
                    st.session_state.embeddings = embs
                    st.success("✅ Вектор сан амжилттай үүслээ!")
                except Exception as e:
                    # Алдааг дэлгэрэнгүй харуулах
                    st.error(f"Embedding хийхэд алдаа гарлаа. Таны API Key эсвэл төлбөрийн үлдэгдэл хүрэлцэхгүй байж магадгүй.")

    # 3. Хайлт ба Асуулт хариулт
    if "embeddings" in st.session_state:
        st.divider()
        query = st.text_input("📝 Документаас асуух асуултаа бичнэ үү:", placeholder="Жишээ нь: Гэрээний хугацаа хэзээ дуусах вэ?")
        
        if query:
            with st.spinner("Хамгийн ойрхон хэсгүүдийг хайж байна..."):
                query_emb = get_embedding(query)
                scores = []
                for i, emb in enumerate(st.session_state.embeddings):
                    score = cosine_similarity(query_emb, emb)
                    scores.append((score, st.session_state.chunks[i], st.session_state.sources[i]))
                
                # Оноогоор нь эрэмбэлэх (Хамгийн өндөр нь дээрээ)
                scores.sort(key=lambda x: x[0], reverse=True)
                
                st.subheader(f"🔍 Олдсон {top_k} илэрц:")
                for i in range(min(top_k, len(scores))):
                    score, text, source = scores[i]
                    with st.expander(f"Илэрц #{i+1} | Ижил тал: {score:.2%} | Эх сурвалж: {source}"):
                        st.write(text)
                        st.caption(f"Файл: {source}")
