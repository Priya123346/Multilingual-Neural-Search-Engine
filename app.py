"""
Streamlit UI for Multilingual Neural Search Engine
Allows users to upload documents and search through them in any language.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
from langdetect import detect
import tempfile
import os
import time

# Page config
st.set_page_config(
    page_title="Multilingual Neural Search Engine",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Multilingual Neural Search Engine")
st.markdown("Upload documents and search through them in **any language**!")

# Initialize session state
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'indexed_texts' not in st.session_state:
    st.session_state.indexed_texts = []
if 'index_built' not in st.session_state:
    st.session_state.index_built = False

# Load encoder automatically on first run
@st.cache_resource
def load_encoder():
    """Load the multilingual encoder once and cache it"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

if st.session_state.encoder is None:
    with st.spinner("🔄 Loading multilingual model (first time only)..."):
        st.session_state.encoder = load_encoder()
        st.success("✅ Model loaded and ready!")

# Helper functions
def detect_language(text):
    """Detect the language of input text"""
    try:
        return detect(text)
    except:
        return "unknown"

class Multilingual_FAISS:
    """FAISS index with cosine similarity for semantic search"""
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.index = faiss.IndexFlatIP(dimensions)
        self.vectors = {}
        self.counter = 0
        self.texts = []
    
    def add_batch(self, texts, vectors):
        """Add multiple texts and embeddings at once"""
        vectors_normalized = normalize(vectors, norm='l2')
        vectors_normalized = vectors_normalized.astype('float32')
        
        self.index.add(vectors_normalized)
        for i, (text, vec) in enumerate(zip(texts, vectors_normalized)):
            self.vectors[self.counter] = (text, vec)
            self.texts.append(text)
            self.counter += 1
    
    def search(self, v, k=5, threshold=0.2):
        """Search for similar items"""
        v_normalized = normalize(v.reshape(1, -1), norm='l2')[0]
        v_normalized = v_normalized.reshape(1, -1).astype('float32')
        
        distances, item_idx = self.index.search(v_normalized, k)
        
        results = []
        for d, i in zip(distances[0], item_idx[0]):
            if i == -1:
                break
            similarity = float(d) * 100
            if similarity >= threshold * 100:
                text, vec = self.vectors[i]
                results.append((text, similarity))        
        return results

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    top_k = st.slider("Number of Results:", 1, 20, 5)
    threshold = st.slider("Similarity Threshold (%):", 0, 100, 20) / 100.0

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Upload & Index")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV or TXT file:",
        type=["csv", "txt"],
        help="CSV: must have 'question' or 'text' column. TXT: one question per line."
    )
    
    if uploaded_file:
        st.info(f"📄 File: {uploaded_file.name}")
        
        # Parse file
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file, encoding='latin1')
                
                # Try to find text column
                text_col = None
                for col in ['question', 'question1', 'text', 'content', 'description']:
                    if col in df.columns:
                        text_col = col
                        break
                
                if text_col is None:
                    st.warning(f"⚠️ No recognized column. Available columns: {list(df.columns)}")
                    text_col = st.selectbox("Select column to index:", df.columns)
                
                texts = df[text_col].dropna().astype(str).values.tolist()
            
            else:  # TXT file
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.success(f"✅ Loaded {len(texts)} documents")
            
            # Build index (chunked, with progress) - faster and non-blocking for large files
            if st.button("🚀 Build Index", key="build_index"):
                n = len(texts)
                # choose a safe batch size depending on dataset size
                batch_size = 64 if n < 20000 else 32
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                faiss_index = None
                start_time = time.time()

                try:
                    for start_idx in range(0, n, batch_size):
                        end_idx = min(n, start_idx + batch_size)
                        batch = texts[start_idx:end_idx]

                        # Encode this batch
                        embeddings = st.session_state.encoder.encode(
                            batch,
                            batch_size=batch_size,
                            convert_to_numpy=True
                        )
                        # ensure proper dtype
                        embeddings = embeddings.astype('float32')

                        # initialize FAISS index on first batch
                        if faiss_index is None:
                            dimension = embeddings.shape[1]
                            faiss_index = Multilingual_FAISS(dimension)

                        # add batch to index
                        faiss_index.add_batch(batch, embeddings)

                        # update progress
                        progress = (end_idx) / float(n)
                        progress_bar.progress(progress)
                        status_text.text(f"Indexed {end_idx} / {n} documents")

                    duration = time.time() - start_time

                    # Store in session
                    st.session_state.faiss_index = faiss_index
                    st.session_state.indexed_texts = texts
                    st.session_state.index_built = True

                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Index built! Indexed {n} documents in {duration:.1f}s")

                except Exception as e:
                    st.error(f"❌ Error building index: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with col2:
    st.subheader("🔍 Search")
    
    if not st.session_state.index_built:
        st.info("👈 Upload a file and build an index first!")
    else:
        st.success(f"✅ Index ready with {len(st.session_state.indexed_texts)} documents")
        
        # Search input
        query = st.text_input(
            "Enter your search query (in any language):",
            placeholder="e.g., 'How to learn Python?' or 'Python ని ఎలా నేర్చుకోవాలి?' or 'Python को कैसे सीखें?'"
        )
        
        if query:
            # Detect language
            detected_lang = detect_language(query)
            lang_map = {
                'en': '🇬🇧 English',
                'es': '🇪🇸 Spanish',
                'fr': '🇫🇷 French',
                'de': '🇩🇪 German',
                'hi': '🇮🇳 Hindi',
                'te': '🇮🇳 Telugu',
                'zh-cn': '🇨🇳 Chinese',
                'ar': '🇸🇦 Arabic',
                'ja': '🇯🇵 Japanese',
            }
            lang_display = lang_map.get(detected_lang, f"🌐 {detected_lang.upper()}")
            
            col_lang, col_results = st.columns([1, 3])
            with col_lang:
                st.metric("Detected Language", lang_display)
            
            # Perform search
            with st.spinner("Searching..."):
                # Ensure we get a numpy array of shape (dim,)
                try:
                    query_vec = st.session_state.encoder.encode(query, convert_to_numpy=True)
                except TypeError:
                    # fallback if the encoder doesn't accept convert_to_numpy
                    query_vec = st.session_state.encoder.encode(query)

                # If the encoder returned a (1, dim) array, squeeze to (dim,)
                try:
                    import numpy as _np
                    if hasattr(query_vec, 'ndim') and query_vec.ndim == 2 and query_vec.shape[0] == 1:
                        query_vec = query_vec[0]
                    query_vec = _np.asarray(query_vec, dtype='float32')
                except Exception:
                    pass

                results = st.session_state.faiss_index.search(
                    query_vec,
                    k=top_k,
                    threshold=threshold
                )

                # Debug: show number of results returned (helps diagnose UI issues)
                st.write(f"(debug) results returned: {len(results)}")
            
            # Display results
            if results:
                st.subheader(f"📊 Results ({len(results)} found)")
                for rank, (text, similarity) in enumerate(results, 1):
                    with st.container():
                        col_rank, col_sim, col_text = st.columns([0.5, 1, 4])
                        with col_rank:
                            st.write(f"**#{rank}**")
                        with col_sim:
                            st.metric("Similarity", f"{similarity:.2f}%")
                        with col_text:
                            st.write(text)
                        st.divider()
            else:
                st.warning(f"⚠️ No results found above {threshold*100:.0f}% similarity threshold")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🌍 Multilingual Neural Search Engine | Powered by Sentence-Transformers & FAISS</p>
    <p><small>Supports 100+ languages • Semantic search • No translation needed</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
