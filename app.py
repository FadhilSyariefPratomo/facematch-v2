import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceMatch · Neo-Brutalism",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS Neo-Brutalism & Layout ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Space+Grotesk:wght@500;700;900&display=swap');

html, body, [class*="css"] { 
    font-family: 'Space Grotesk', sans-serif; 
    color: #000000;
}

h1, h2, h3, h4, p { color: #000000 !important; }

:root {
    --border-thick: 4px solid #000;
    --border-thin: 2px solid #000;
    --shadow-solid: 6px 6px 0px #000;
    --shadow-hover: 8px 8px 0px #000;
    --color-yellow: #FFD500;
    --color-blue: #4D90FE;
    --color-pink: #FF6B6B;
    --color-green: #4ADE80;
    --color-purple: #A78BFA;
    --color-orange: #F97316;
    --color-white: #FFFFFF;
}

/* Hero Section */
.hero {
    background-color: var(--color-yellow);
    border: var(--border-thick);
    padding: 2.5rem 2rem; 
    margin-bottom: 2rem; 
    position: relative; 
    box-shadow: var(--shadow-solid);
}
.badge { display: inline-block; background: #000; color: var(--color-yellow); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; padding: 6px 12px; border: var(--border-thin); margin-bottom: 1rem; }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 900; text-transform: uppercase; color: #000; margin: 0 0 0.5rem; line-height: 1.1; letter-spacing: -1px; }
.hero-sub { color: #000; font-family: 'IBM Plex Mono', monospace; font-size: 1rem; font-weight: 600; margin: 0; line-height: 1.5; }
.hero-stats { display: flex; gap: 1.5rem; margin-top: 1.5rem; flex-wrap: wrap; border-top: var(--border-thick); padding-top: 1rem; }
.hero-stat { color: #000; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; }
.hero-stat strong { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; display: block; font-weight: 900; }

/* Upload Zone & Previews */
.upload-zone { background: var(--color-white); border: 4px dashed #000; padding: 2rem 1rem; text-align: center; box-shadow: var(--shadow-solid); transition: all 0.2s ease-in-out; margin-bottom: 1.5rem; cursor: pointer; }
.upload-zone:hover { transform: translate(-4px, -4px); box-shadow: var(--shadow-hover); background: #F8F8F8; border-style: solid; }
.upload-label { font-weight: 900; color: #000; font-size: 1.2rem; margin: 0.5rem 0 0.2rem; text-transform: uppercase; }
.upload-sub { font-family: 'IBM Plex Mono', monospace; color: #000; font-size: 0.85rem; font-weight: 600; margin: 0; }
.upload-icon { font-size: 3rem; }

.preview-header { border: var(--border-thick); padding: 0.8rem; text-align: center; font-family: 'IBM Plex Mono', monospace; font-weight: 900; margin-top: 1rem; margin-bottom: 0.8rem; box-shadow: 4px 4px 0px #000; text-transform: uppercase; color: #000; }

/* Verdict Box */
.verdict-box { border: var(--border-thick); padding: 2rem; text-align: center; margin-top: 1rem; box-shadow: var(--shadow-solid); transition: background-color 0.3s ease; margin-bottom: 2.5rem; }
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1; }
.result-title { font-size: 1.8rem; font-weight: 900; margin: 0.2rem 0; color: #000; text-transform: uppercase; }
.result-sub { font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #000; font-size: 0.95rem; margin: 0; }
.similarity-bar-wrap { margin: 1.5rem 0 0.5rem; border-top: var(--border-thick); padding-top: 1rem; }
.similarity-label { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.9rem; color: #000; text-transform: uppercase; margin-bottom: 4px; }
.similarity-pct { font-size: 3.5rem; font-weight: 900; color: #000; line-height: 1; text-shadow: 2px 2px 0px #FFF; margin: 0; }

/* Interactive Cards */
.interactive-card { background: var(--color-white); border: var(--border-thick); box-shadow: 4px 4px 0px #000; padding: 1.2rem; text-align: center; transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.2s ease; margin-bottom: 1.5rem !important; cursor: default; }
.interactive-card:hover { transform: scale(1.02) translate(-2px, -2px); box-shadow: 8px 8px 0px #000; }
.metric-val { font-size: 2.2rem; font-weight: 900; margin: 10px 0; color: #000; }

/* Feature Cards */
.feature-card { background: var(--color-white); border: var(--border-thick); box-shadow: 4px 4px 0px #000; padding: 1.2rem; margin-bottom: 1.5rem; width: 100%; box-sizing: border-box; }
.feature-item { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; padding: 6px 8px; border-radius: 4px; transition: all 0.2s ease; cursor: crosshair; }
.feature-item:hover { transform: translateX(8px); background: #F4F4F5; box-shadow: -4px 0px 0px #000; }
.feature-name { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.8rem; color: #000; width: 40%; }
.feature-bar-wrap { flex: 1; height: 18px; border: var(--border-thin); background: var(--color-white); margin: 0 12px; position: relative; box-shadow: 2px 2px 0px #000; overflow: hidden; }
.feature-bar { height: 100%; border-right: 2px solid #000; transition: width 1s cubic-bezier(0.22, 1, 0.36, 1); }
.feature-item:hover .feature-bar { filter: brightness(1.1); }
.feature-val { font-family: 'IBM Plex Mono', monospace; font-weight: 900; font-size: 0.85rem; color: #000; min-width: 45px; text-align: right; }

/* Callouts & Buttons */
.callout { background: var(--color-blue); border: var(--border-thick); box-shadow: var(--shadow-solid); padding: 1.2rem; margin: 1.5rem 0; color: #000; font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.9rem; line-height: 1.5; }
.divider { border: none; border-top: 6px solid #000; margin: 2.5rem 0; }
div.stButton > button { background-color: var(--color-blue) !important; color: #FFF !important; text-shadow: 1px 1px 0px #000; border: var(--border-thick) !important; border-radius: 0 !important; box-shadow: var(--shadow-solid) !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 900 !important; font-size: 1.2rem !important; text-transform: uppercase !important; padding: 1rem !important; transition: all 0.1s ease !important; margin-bottom: 1.5rem; }
div.stButton > button:active { transform: translate(4px, 4px) !important; box-shadow: 2px 2px 0px #000 !important; }
div.stButton > button:hover { background-color: var(--color-pink) !important; color: #000 !important; text-shadow: none; }

/* Grid Headers */
.grid-header { font-weight: 900; text-transform: uppercase; font-size: 1.4rem; border-bottom: var(--border-thick); padding-bottom: 0.5rem; margin: 2rem 0 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─── Constants & State ─────────────────────────────────────────────────────────
IMG_SIZE = (100, 100)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pca_model.pkl")

# Threshold Akurat
DEFAULT_COS_THRESHOLD = 0.70
DEFAULT_EUC_THRESHOLD = 15.0

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'vec_old' not in st.session_state:
    st.session_state.vec_old = None
if 'vec_new' not in st.session_state:
    st.session_state.vec_new = None

# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def detect_and_crop(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray) # Standarisasi pencahayaan
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        cropped = gray[y:y+h, x:x+w]
        resized = cv2.resize(cropped, IMG_SIZE)
        return (resized / 255.0).flatten(), True, (x, y, w, h)
    else:
        resized = cv2.resize(gray, IMG_SIZE)
        return (resized / 255.0).flatten(), False, None

def draw_face_box(pil_img, box):
    img_arr = np.array(pil_img.convert("RGB"))
    if box:
        x, y, w, h = box
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), (0, 0, 0), 8)
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), (0, 213, 255), 4) 
        text = "WAJAH"
        cv2.putText(img_arr, text, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 6)
        cv2.putText(img_arr, text, (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 213, 255), 2)
    return Image.fromarray(img_arr)

def extract_features(vec_pca):
    abs_vals = np.abs(vec_pca)
    top_idx = np.argsort(abs_vals)[::-1][:6]
    return {
        "PC Dominan": float(abs_vals[top_idx[0]]),
        "Variasi PC2": float(abs_vals[top_idx[1]]),
        "Variasi PC3": float(abs_vals[top_idx[2]]),
        "Energi Total": float(np.sqrt(np.sum(vec_pca**2))),
        "Kemiringan": float(np.mean(np.abs(vec_pca))),
        "Dispersi": float(np.std(vec_pca)),
    }

def compute_similarity(vec1, vec2, pca, cos_threshold, euc_threshold):
    v1 = pca.transform(vec1.reshape(1, -1))
    v2 = pca.transform(vec2.reshape(1, -1))
    cos_sim = float(cosine_similarity(v1, v2)[0][0])
    euc_dist = float(euclidean(v1[0], v2[0]))
    
    cos_pct = max(0.0, min(100.0, cos_sim * 100))
    euc_pct = max(0.0, min(100.0, (1 - (euc_dist / 40.0)) * 100))
    
    return {
        "cosine_similarity": cos_sim,
        "euclidean_distance": euc_dist,
        "cosine_ok": cos_sim >= cos_threshold,
        "euclidean_ok": euc_dist <= euc_threshold,
        "combined_pct": (cos_pct + euc_pct) / 2,
        "vec_pca_1": v1[0],
        "vec_pca_2": v2[0],
        "feat1": extract_features(v1[0]),
        "feat2": extract_features(v2[0]),
    }

def make_pca_plot(v1, v2):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], linestyle="--", color="#000000", lw=2, zorder=1)
    ax.scatter(v1[0], v1[1], s=350, c="#FFD500", zorder=5, label="FOTO LAMA", edgecolors="#000000", linewidths=3)
    ax.scatter(v2[0], v2[1], s=350, c="#FF6B6B", zorder=5, label="FOTO SEKARANG", edgecolors="#000000", linewidths=3)
    
    mid = ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2)
    dist = np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
    bbox_props = dict(boxstyle="square,pad=0.4", fc="#4ADE80", ec="#000", lw=3)
    ax.annotate(f"DIST: {dist:.1f}", mid, fontsize=9, color="#000", ha="center", weight="bold", bbox=bbox_props, zorder=10)
    
    ax.set_xlabel("KOMPONEN UTAMA 1", color="#000", fontsize=9, weight="bold")
    ax.set_ylabel("KOMPONEN UTAMA 2", color="#000", fontsize=9, weight="bold")
    ax.tick_params(colors="#000", labelsize=9, width=2)
    for sp in ax.spines.values(): 
        sp.set_edgecolor("#000000")
        sp.set_linewidth(3)
        
    legend = ax.legend(facecolor="#FFFFFF", edgecolor="#000", labelcolor="#000", fontsize=8, framealpha=1, loc='best')
    legend.get_frame().set_linewidth(3)
    plt.tight_layout()
    return fig

# ─── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ File `pca_model.pkl` tidak ditemukan. Pastikan file model ada di folder yang sama dengan `app.py`.")
    st.stop()

model_data = load_model()
pca_model = model_data["pca"]

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="badge">ALGORITMA EIGENFACES</div>
    <h1 class="hero-title">FACEMATCH</h1>
    <p class="hero-sub">Bandingkan foto masa kecil dengan foto saat ini.<br>Mesin akan menentukan apakah itu orang yang sama.</p>
    <div class="hero-stats">
        <div class="hero-stat"><strong>{model_data.get('n_train_images', 0):,}</strong>Training</div>
        <div class="hero-stat"><strong>{model_data.get('n_identities', 0):,}</strong>Identitas</div>
        <div class="hero-stat"><strong>k={model_data.get('k', 0)}</strong>Dimensi</div>
        <div class="hero-stat"><strong>{model_data.get('explained_variance', 0)*100:.1f}%</strong>Varians</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Upload Area & PREVIEW REAL-TIME ───────────────────────────────────────────
col_old, col_new = st.columns(2, gap="large")

with col_old:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📼</div>
        <p class="upload-label">FOTO MASA LALU</p>
        <p class="upload-sub">Upload arsip foto masa kecil</p>
    </div>
    """, unsafe_allow_html=True)
    img_old_file = st.file_uploader("Foto Lama", type=["jpg","jpeg","png","webp"], key="old", label_visibility="collapsed")
    
    # Pratinjau Langsung (Real-time Preview)
    if img_old_file:
        # Caching status agar tidak me-run ulang deteksi setiap kali slider bergeser
        if st.session_state.get('old_file_id') != img_old_file.file_id:
            img_old = Image.open(img_old_file)
            vec_old, det_old, box_old = detect_and_crop(img_old)
            st.session_state.vec_old = vec_old
            st.session_state.det_old = det_old
            st.session_state.img_old_box = draw_face_box(img_old, box_old)
            st.session_state.old_file_id = img_old_file.file_id
            st.session_state.analyzed = False # Reset hasil analisis jika foto diganti
            
        header_color = "var(--color-green)" if st.session_state.det_old else "var(--color-pink)"
        header_text = "✓ DETEKSI WAJAH LAMA" if st.session_state.det_old else "❌ WAJAH TIDAK DITEMUKAN"
        st.markdown(f"<div class='preview-header' style='background: {header_color};'>{header_text}</div>", unsafe_allow_html=True)
        st.image(st.session_state.img_old_box, use_container_width=True)

with col_new:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📸</div>
        <p class="upload-label">FOTO SEKARANG</p>
        <p class="upload-sub">Upload foto identitas terbaru</p>
    </div>
    """, unsafe_allow_html=True)
    img_new_file = st.file_uploader("Foto Sekarang", type=["jpg","jpeg","png","webp"], key="new", label_visibility="collapsed")
    
    # Pratinjau Langsung (Real-time Preview)
    if img_new_file:
        if st.session_state.get('new_file_id') != img_new_file.file_id:
            img_new = Image.open(img_new_file)
            vec_new, det_new, box_new = detect_and_crop(img_new)
            st.session_state.vec_new = vec_new
            st.session_state.det_new = det_new
            st.session_state.img_new_box = draw_face_box(img_new, box_new)
            st.session_state.new_file_id = img_new_file.file_id
            st.session_state.analyzed = False
            
        header_color = "var(--color-green)" if st.session_state.det_new else "var(--color-pink)"
        header_text = "✓ DETEKSI WAJAH SEKARANG" if st.session_state.det_new else "❌ WAJAH TIDAK DITEMUKAN"
        st.markdown(f"<div class='preview-header' style='background: {header_color};'>{header_text}</div>", unsafe_allow_html=True)
        st.image(st.session_state.img_new_box, use_container_width=True)

# ─── Settings & Execution ──────────────────────────────────────────────────────
st.markdown("<div class='grid-header'>🎛️ PARAMETER INTERAKTIF</div>", unsafe_allow_html=True)
sc1, sc2 = st.columns(2, gap="large")
with sc1:
    cos_threshold = st.slider("Cosine Similarity (≥)", -1.0, 1.0, DEFAULT_COS_THRESHOLD, 0.01)
with sc2:
    euc_threshold = st.slider("Euclidean Distance (≤)", 1.0, 80.0, DEFAULT_EUC_THRESHOLD, 0.5)

st.markdown("""
<div class="callout" style="margin-top: 0;">
<strong>INFO:</strong> Menggeser slider akan <strong>memperbarui metrik secara real-time</strong> tanpa perlu menekan tombol analisis lagi jika foto sudah terdeteksi.
</div>
""", unsafe_allow_html=True)

if img_old_file and img_new_file:
    if st.button("MULAI ANALISIS WAJAH ⚡", use_container_width=True):
        # Karena deteksi (cropping & matriks) sudah dilakukan secara instan di atas, 
        # tombol ini hanya memicu kalkulasi persentase dan animasi loading UI.
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
            status_text.markdown(f"**Mengekstrak ciri geometris & komputasi PCA... {i+1}%**")
        
        status_text.empty()
        progress_bar.empty()
        st.session_state.analyzed = True

# ─── Layout Visual & Hasil Akhir ───────────────────────────────────────────────
if st.session_state.analyzed:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not st.session_state.det_old or not st.session_state.det_new:
        st.markdown("""
        <div class="callout" style="background:#FFD500;">
        <strong>PERINGATAN:</strong> Wajah gagal dideteksi secara otomatis. Sistem membaca seluruh gambar. Akurasi dapat menurun drastis.
        </div>
        """, unsafe_allow_html=True)

    result = compute_similarity(st.session_state.vec_old, st.session_state.vec_new, pca_model, cos_threshold, euc_threshold)
    
    both_sim  = result["cosine_ok"] and result["euclidean_ok"]
    both_diff = (not result["cosine_ok"]) and (not result["euclidean_ok"])
    
    if both_sim:
        v_class, v_icon, v_title = "var(--color-green)", "✅", "KEMUNGKINAN ORANG YANG SAMA"
        v_sub = "Metrik mendeteksi kemiripan tinggi berdasarkan parameter saat ini."
    elif both_diff:
        v_class, v_icon, v_title = "var(--color-pink)", "❌", "BUKAN ORANG YANG SAMA"
        v_sub = "Tingkat deviasi fitur terlalu jauh melampaui batas toleransi."
    else:
        v_class, v_icon, v_title = "var(--color-yellow)", "⚠️", "HASIL KONFLIK"
        v_sub = "Metode memberikan hasil berbeda. Coba atur ulang threshold."

    # 1. VERDICT UTAMA
    st.markdown(f"""
    <div class="verdict-box" style="background-color: {v_class};">
        <div class="result-icon">{v_icon}</div>
        <h2 class="result-title">{v_title}</h2>
        <p class="result-sub">{v_sub}</p>
        <div class="similarity-bar-wrap">
            <p class="similarity-label">SKOR KEMIRIPAN GLOBAL</p>
            <p class="similarity-pct">{result['combined_pct']:.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. GRID KIRI (GAMBAR KOMPARASI) & KANAN (METRIK + PCA)
    col_vis, col_met = st.columns([1.1, 1], gap="large")
    
    with col_vis:
        st.markdown("<div class='grid-header'>🖼️ GRID MATRIKS INPUT</div>", unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1: 
            st.image(st.session_state.img_old_box, caption="LAMA (KONTUR)", use_container_width=True)
            st.image(st.session_state.vec_old.reshape(IMG_SIZE), caption="GRID GRAYSCALE (100x100)", clamp=True, use_container_width=True)
        with v2: 
            st.image(st.session_state.img_new_box, caption="SEKARANG (KONTUR)", use_container_width=True)
            st.image(st.session_state.vec_new.reshape(IMG_SIZE), caption="GRID GRAYSCALE (100x100)", clamp=True, use_container_width=True)

    with col_met:
        st.markdown("<div class='grid-header'>📊 METRIK THRESHOLD</div>", unsafe_allow_html=True)
        
        c_stat1 = "var(--color-green)" if result['cosine_ok'] else "var(--color-pink)"
        st.markdown(f"""
        <div class="interactive-card" style="border-bottom: 8px solid {c_stat1};">
            <p style="font-family:'IBM Plex Mono'; font-weight:bold; margin:0;">COSINE SIMILARITY</p>
            <p class="metric-val">{result['cosine_similarity']:.4f}</p>
            <p style="font-size:0.8rem; margin:0; font-family:'IBM Plex Mono'; font-weight:bold;">TARGET: ≥ {cos_threshold}</p>
        </div>
        """, unsafe_allow_html=True)
        
        c_stat2 = "var(--color-green)" if result['euclidean_ok'] else "var(--color-pink)"
        st.markdown(f"""
        <div class="interactive-card" style="border-bottom: 8px solid {c_stat2};">
            <p style="font-family:'IBM Plex Mono'; font-weight:bold; margin:0;">EUCLIDEAN DISTANCE</p>
            <p class="metric-val">{result['euclidean_distance']:.4f}</p>
            <p style="font-size:0.8rem; margin:0; font-family:'IBM Plex Mono'; font-weight:bold;">TARGET: ≤ {euc_threshold}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='grid-header' style='margin-top: 1.5rem;'>📐 KOORDINAT PCA</div>", unsafe_allow_html=True)
        fig = make_pca_plot(result["vec_pca_1"], result["vec_pca_2"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 3. GRID FITUR 
    st.markdown("<div class='grid-header'>🧬 ANALISIS FITUR WAJAH</div>", unsafe_allow_html=True)
    col_feat1, col_feat2 = st.columns(2, gap="large")
    
    bar_colors = ["var(--color-blue)", "var(--color-pink)", "var(--color-green)", "var(--color-yellow)", "var(--color-purple)", "var(--color-orange)"]
    
    with col_feat1:
        st.markdown("<p style='font-family:\"IBM Plex Mono\"; font-weight:bold;'>KOMPOSISI FITUR LAMA</p>", unsafe_allow_html=True)
        feats1 = result["feat1"]
        max_val = max(feats1.values()) if feats1 else 1
        feat_html = '<div class="feature-card">'
        for idx, (name, val) in enumerate(feats1.items()):
            pct = min(100, val / max_val * 100) if max_val > 0 else 0
            color = bar_colors[idx % len(bar_colors)]
            feat_html += f"""
            <div class="feature-item" title="Detail nilai {name}: {val:.4f}">
                <span class="feature-name">{name}</span>
                <div class="feature-bar-wrap">
                    <div class="feature-bar" style="width:{pct:.0f}%; background-color: {color};"></div>
                </div>
                <span class="feature-val">{val:.2f}</span>
            </div>"""
        feat_html += "</div>"
        st.markdown(feat_html, unsafe_allow_html=True)

    with col_feat2:
        st.markdown("<p style='font-family:\"IBM Plex Mono\"; font-weight:bold;'>KOMPOSISI FITUR SEKARANG</p>", unsafe_allow_html=True)
        feats2 = result["feat2"]
        max_val2 = max(feats2.values()) if feats2 else 1
        feat_html2 = '<div class="feature-card">'
        for idx, (name, val) in enumerate(feats2.items()):
            pct = min(100, val / max_val2 * 100) if max_val2 > 0 else 0
            color = bar_colors[idx % len(bar_colors)]
            feat_html2 += f"""
            <div class="feature-item" title="Detail nilai {name}: {val:.4f}">
                <span class="feature-name">{name}</span>
                <div class="feature-bar-wrap">
                    <div class="feature-bar" style="width:{pct:.0f}%; background-color: {color};"></div>
                </div>
                <span class="feature-val">{val:.2f}</span>
            </div>"""
        feat_html2 += "</div>"
        st.markdown(feat_html2, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="callout" style="background-color: var(--color-white); margin-top:2rem;">
    <strong>INSTRUKSI OPERASIONAL:</strong><br><br>
    1. Masukkan foto wajah lama di kolom kiri.<br>
    2. Masukkan foto wajah baru di kolom kanan.<br>
    3. Atur parameter threshold jika diperlukan.<br>
    4. Eksekusi tombol <strong>MULAI ANALISIS WAJAH</strong>.<br>
    5. Tunggu hasil komputasi matriks selesai.
    </div>
    """, unsafe_allow_html=True)