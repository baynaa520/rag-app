import streamlit as st
from docx import Document
import PyPDF2
import io
import re
from openai import OpenAI
import numpy as np

# --- 1. ТОХИРГОО ---
OPENAI_API_KEY = "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"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 2. ФАЙЛ УНШИХ ФУНКЦ ---
def read_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type == 'docx':
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_type == 'pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif file_type == 'txt':
        return uploaded_file.getvalue().decode("utf-8")
    return None

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Way Academy - All-in-One RAG", layout="wide")
st.title("🤖 Advanced RAG Tool")

# Sidebar - Тохиргоо
with st.sidebar:
    st.header("⚙️ Конфигураци")
    chunk_method = st.radio("Chunk хийх арга:", ("Өгүүлбэрээр", "Параграфаар"))
    top_k = st.number_input("Хэдэн илэрц (chunk) харуулах вэ?", min_value=1, max_value=10, value=5)
    st.divider()
    if st.button("Санах ой цэвэрлэх"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# Файл оруулах хэсэг
uploaded_file = st.file_uploader("Файлаа оруулна уу (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    # 1. Файл унших
    if "full_text" not in st.session_state:
        with st.spinner("Файлыг уншиж байна..."):
            st.session_state.full_text = read_file(uploaded_file)
            st.success(f"✅ {uploaded_file.name} амжилттай уншигдлаа.")

    # 2. Chunking хийх
    if chunk_method == "Параграфаар":
        chunks = [p.strip() for p in st.session_state.full_text.split('\n') if p.strip()]
    else:
        chunks = re.split(r'(?<=[.!?]) +', st.session_state.full_text)
        chunks = [s.strip() for s in chunks if s.strip()]

    st.info(f"📄 Нийт {len(chunks)} хэсэг (chunk) үүссэн байна.")

    # 3. Embedding үүсгэх товч
    if "embeddings" not in st.session_state:
        if st.button("🎯 Embedding үүсгэж эхлэх"):
            with st.spinner("OpenAI руу илгээж байна... (Түр хүлээнэ үү)"):
                try:
                    embs = [get_embedding(c) for c in chunks]
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embs
                    st.success("✅ Вектор сан бэлэн боллоо!")
                except Exception as e:
                    st.error(f"Алдаа гарлаа: {e}")

    # 4. АСУУЛТ АСУУХ ХЭСЭГ (Одоо гарч ирнэ)
    if "embeddings" in st.session_state:
        st.divider()
        st.subheader("❓ Документаас асуух")
        query = st.text_input("Асуултаа энд бичнэ үү:", placeholder="Deerh file dotroos hvssenee asuugaarai?")

        if query:
            with st.spinner("Хайж байна..."):
                query_emb = get_embedding(query)
                
                # Similarity тооцоолох
                scores = []
                for i, emb in enumerate(st.session_state.embeddings):
                    score = cosine_similarity(query_emb, emb)
                    scores.append((score, st.session_state.chunks[i]))
                
                # Оноогоор эрэмбэлэх
                scores.sort(key=lambda x: x[0], reverse=True)
                
                # Үр дүнг харуулах
                st.markdown(f"### 🔍 Хамгийн ойрхон {top_k} илэрц:")
                for i in range(min(top_k, len(scores))):
                    score, text = scores[i]
                    with st.expander(f"Илэрц #{i+1} | Ижил тал: {score:.2%}"):
                        st.write(text)
