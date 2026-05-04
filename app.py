import streamlit as st
from docx import Document
import PyPDF2
import re
from openai import OpenAI
import numpy as np
import io

# --- 1. ТОХИРГОО ---
# Streamlit Secrets-ээс Key-г унших (Аюулгүй арга)
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("🔑 OpenAI API Key тохируулагдаагүй байна. Settings -> Secrets хэсэгт нэмнэ үү.")

def get_embedding(text, model="text-embedding-3-small"):
    # Кирилл үсэг болон тусгай тэмдэгтүүдийг цэвэрлэх/бэлдэх
    text = text.replace("\n", " ").strip()
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 2. УНИВЕРСАЛ ФАЙЛ УНШИГЧ (UTF-8 дэмждэг) ---
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
            # Encoding алдаанаас сэргийлж 'ignore' ашиглав
            text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"❌ '{uploaded_file.name}' файлыг уншихад алдаа: {e}")
    return text

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Way Academy RAG", layout="wide")
st.title("🤖 Advanced Multi-File RAG Tool")

with st.sidebar:
    st.header("⚙️ Конфигураци")
    chunk_method = st.radio("Splatting арга:", ("Өгүүлбэрээр", "Параграфаар"))
    top_k = st.number_input("Хэдэн илэрц харуулах вэ?", 1, 10, 5)
    if st.button("Шинээр эхлэх (Reset)"):
        st.session_state.clear()
        st.rerun()

# Файл оруулах (Олон файл зөвшөөрнө)
uploaded_files = st.file_uploader(
    "Файлуудаа оруулна уу (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    # 1. Текст боловсруулах
    if "chunks" not in st.session_state:
        all_chunks = []
        all_sources = []
        
        with st.spinner("Файлуудыг уншиж байна..."):
            for uploaded_file in uploaded_files:
                content = read_single_file(uploaded_file)
                
                # Chunking
                if chunk_method == "Параграфаар":
                    file_chunks = [p.strip() for p in content.split('\n') if p.strip()]
                else:
                    # Кирилл цэгийг тооцсон Regex
                    file_chunks = re.split(r'(?<=[.!?]) +', content)
                    file_chunks = [s.strip() for s in file_chunks if s.strip()]
                
                for c in file_chunks:
                    all_chunks.append(c)
                    all_sources.append(uploaded_file.name)
            
            st.session_state.chunks = all_chunks
            st.session_state.sources = all_sources
            st.success(f"✅ Нийт {len(all_chunks)} хэсэг мэдээлэл бэлэн боллоо.")

    # 2. Embedding (Вектор сан)
    if "embeddings" not in st.session_state and "chunks" in st.session_state:
        if st.button("🚀 Мэдлэгийн сан үүсгэх"):
            with st.spinner("Вектор сан үүсгэж байна... Түр хүлээнэ үү."):
                try:
                    embs = [get_embedding(c) for c in st.session_state.chunks]
                    st.session_state.embeddings = embs
                    st.success("✅ Вектор сан амжилттай үүслээ!")
                except Exception as e:
                    st.error(f"Embedding алдаа: {e}")

    # 3. Хайлт ба Асуулт
    if "embeddings" in st.session_state:
        st.divider()
        query = st.text_input("📝 Асуултаа бичнэ үү:", placeholder="Документаас хайх...")
        
        if query:
            with st.spinner("Хайж байна..."):
                query_emb = get_embedding(query)
                scores = []
                for i, emb in enumerate(st.session_state.embeddings):
                    score = cosine_similarity(query_emb, emb)
                    scores.append((score, st.session_state.chunks[i], st.session_state.sources[i]))
                
                # Хамгийн өндөр оноотойг эрэмбэлэх
                scores.sort(key=lambda x: x[0], reverse=True)
                
                st.subheader(f"🔍 Олдсон {top_k} илэрц:")
                for i in range(min(top_k, len(scores))):
                    score, text, source = scores[i]
                    with st.expander(f"Илэрц #{i+1} | Ижил тал: {score:.2%} | Эх сурвалж: {source}"):
                        st.write(text)
