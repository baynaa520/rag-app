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
def add_website_style_bg():
    st.markdown(
        """
        <style>
        /* 1. Вэбсайт шиг Mesh Gradient Background */
        .stApp {
            background: linear-gradient(125deg, #0e1117 0%, #161b22 50%, #0e1117 100%);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
            overflow: hidden;
        }

        @keyframes gradientAnimation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* 2. Гэрэлтсэн дугуй дүрснүүд (Floating Orbs) */
        .stApp::before {
            content: "";
            position: fixed;
            top: -10%;
            left: -10%;
            width: 40%;
            height: 40%;
            background: radial-gradient(circle, rgba(79, 172, 254, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
            z-index: -1;
            filter: blur(50px);
            animation: orbMove 20s infinite alternate;
        }

        .stApp::after {
            content: "";
            position: fixed;
            bottom: -10%;
            right: -10%;
            width: 50%;
            height: 50%;
            background: radial-gradient(circle, rgba(0, 242, 254, 0.1) 0%, rgba(0, 0, 0, 0) 70%);
            z-index: -1;
            filter: blur(60px);
            animation: orbMove 25s infinite alternate-reverse;
        }

        @keyframes orbMove {
            from { transform: translate(0, 0); }
            to { transform: translate(100px, 100px); }
        }

        /* 3. Card стиль (Контент хэсгийг вэбсайт шиг цэгцтэй болгох) */
        div.stButton > button {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
            transition: 0.3s;
            box-shadow: 0px 4px 15px rgba(0, 242, 254, 0.3);
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0px 6px 20px rgba(0, 242, 254, 0.5);
        }

        /* 4. Текстүүдийг вэбсайт шиг фонттой болгох */
        h1 {
            font-family: 'Inter', sans-serif;
            background: -webkit-linear-gradient(#eee, #333);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }

        /* 5. Sidebar-ийг шилэн (Glassmorphism) болгох */
        [data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.8) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_website_style_bg()

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
