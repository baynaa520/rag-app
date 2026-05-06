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
def add_bg_math_pattern():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            background-image: url("https://www.transparenttextures.com/patterns/carbon-fibre.png");
        }
        
        /* Ар талд томьёо хөвж байгаа мэт эффект өгөх */
        .stApp::before {
            content: "E=mc²  f(x)=∫x²dx  ∇×B=μ₀J  ∑n=1  π≈3.14  Hψ=Eψ  sin²θ+cos²θ=1  a²+b²=c²";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.05;  /* Томьёонууд хэт тод байж болохгүй */
            font-size: 25px;
            color: #ffffff;
            line-height: 150px;
            word-spacing: 200px;
            font-family: 'Courier New', Courier, monospace;
            pointer-events: none;
            overflow: hidden;
            animation: move-bg 60s linear infinite;
        }

        @keyframes move-bg {
            from { transform: translateY(0); }
            to { transform: translateY(-500px); }
        }
        
        /* Sidebar-ийг илүү гоё болгох */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Функцээ дуудах
add_bg_math_pattern()


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
