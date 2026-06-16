import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceMatch · Foto Lama vs Sekarang",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a2f5e 50%, #0a1628 100%);
    border: 1px solid #1e3a6e; border-radius: 16px;
    padding: 2rem 2rem; margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50px; right: -50px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #2563eb22 0%, transparent 70%); border-radius: 50%;
}
.hero::after {
    content: ''; position: absolute; bottom: -30px; left: 30%;
    width: 150px; height: 150px;
    background: radial-gradient(circle, #7c3aed18 0%, transparent 70%); border-radius: 50%;
}
.badge {
    display: inline-block; background: #1e3a6e; color: #60a5fa;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 4px 10px; border-radius: 20px; margin-bottom: 0.7rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 2rem;
    font-weight: 700; color: #e8f0ff; margin: 0 0 0.4rem; letter-spacing: -0.5px;
}
.hero-sub { color: #7a9fd4; font-size: 0.9rem; margin: 0; line-height: 1.6; }
.hero-stats {
    display: flex; gap: 1.5rem; margin-top: 1.2rem; flex-wrap: wrap;
}
.hero-stat { color: #5f7aa8; font-size: 0.8rem; }
.hero-stat strong { color: #93c5fd; font-size: 1rem; display: block; }

.upload-zone {
    background: #0f1829; border: 1.5px dashed #2a4a7f;
    border-radius: 14px; padding: 1.5rem 1rem; text-align: center;
    transition: border-color 0.2s;
}
.upload-label { font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: #93c5fd; font-size: 1rem; margin: 0.5rem 0 0.2rem; }
.upload-sub { color: #5f7aa8; font-size: 0.78rem; margin: 0; }
.upload-icon { font-size: 2.5rem; }

.result-box {
    border-radius: 14px; padding: 1.8rem; text-align: center; margin-top: 0.5rem;
}
.result-similar { background: linear-gradient(135deg, #052e16, #064e3b); border: 1.5px solid #059669; }
.result-different { background: linear-gradient(135deg, #2d0a0a, #3d1515); border: 1.5px solid #dc2626; }
.result-unsure { background: linear-gradient(135deg, #3d2f0a, #4d3a15); border: 1.5px solid #d97706; }
.result-icon { font-size: 3rem; margin-bottom: 0.4rem; }
.result-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0; }
.result-sub { color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.5; }
.similarity-bar-wrap { margin: 1.2rem 0 0.4rem; }
.similarity-label { font-size: 0.78rem; color: #7a9fd4; margin-bottom: 6px; }
.similarity-pct { font-family: 'Space Grotesk', sans-serif; font-size: 2.5rem; font-weight: 700; color: #e8f0ff; }

.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 1.2rem; }
.metric-box { background: #0f1829; border: 1px solid #1e2d4a; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-label { font-size: 0.75rem; color: #7a9fd4; margin: 0 0 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #e8f0ff; margin: 0; }
.metric-status { font-size: 0.75rem; margin-top: 4px; }
.ok { color: #34d399; } .no { color: #f87171; }

.feature-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 1rem; }
.feature-card { background: #0f1829; border: 1px solid #1e2d4a; border-radius: 10px; padding: 1rem; }
.feature-title { font-size: 0.8rem; font-weight: 600; color: #93c5fd; margin: 0 0 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; }
.feature-item { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.feature-name { font-size: 0.78rem; color: #7a9fd4; }
.feature-bar-wrap { flex: 1; margin: 0 10px; height: 4px; background: #1e2d4a; border-radius: 2px; overflow: hidden; }
.feature-bar { height: 100%; background: linear-gradient(90deg, #2563eb, #7c3aed); border-radius: 2px; }
.feature-val { font-size: 0.75rem; color: #c8d6f0; font-family: monospace; min-width: 40px; text-align: right; }

.callout { background: #0c1f3f; border-left: 3px solid #2563eb; border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem; margin: 1rem 0; color: #93c5fd; font-size: 0.83rem; line-height: 1.6; }
.divider { border: none; border-top: 1px solid #1e2d4a; margin: 1.5rem 0; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (100, 100)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pca_model.pkl")
DEFAULT_COS_THRESHOLD = 0.15
DEFAULT_EUC_THRESHOLD = 30.0

# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def detect_and_crop(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), (59, 130, 246), 3)
        cv2.putText(img_arr, "Wajah Terdeteksi", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (59, 130, 246), 2)
    return Image.fromarray(img_arr)

def extract_features(vec_pca):
    """Ekstrak informasi komponen PCA sebagai 'fitur wajah'."""
    abs_vals = np.abs(vec_pca)
    top_idx = np.argsort(abs_vals)[::-1][:6]
    features = {
        "Intensitas PC Dominan": float(abs_vals[top_idx[0]]),
        "Variasi PC 2": float(abs_vals[top_idx[1]]),
        "Variasi PC 3": float(abs_vals[top_idx[2]]),
        "Energi Total": float(np.sqrt(np.sum(vec_pca**2))),
        "Kemiringan Distribusi": float(np.mean(np.abs(vec_pca))),
        "Dispersi Fitur": float(np.std(vec_pca)),
    }
    return features

def compute_similarity(vec1, vec2, pca, cos_threshold, euc_threshold):
    v1 = pca.transform(vec1.reshape(1, -1))
    v2 = pca.transform(vec2.reshape(1, -1))
    cos_sim = float(cosine_similarity(v1, v2)[0][0])
    euc_dist = float(euclidean(v1[0], v2[0]))
    cos_ok = cos_sim >= cos_threshold
    euc_ok = euc_dist <= euc_threshold
    # Similarity percentage (0-100%)
    cos_pct = max(0.0, min(100.0, (cos_sim + 1) / 2 * 100))
    euc_pct = max(0.0, min(100.0, max(0, 1 - euc_dist / 60) * 100))
    combined_pct = (cos_pct + euc_pct) / 2
    return {
        "cosine_similarity": cos_sim,
        "euclidean_distance": euc_dist,
        "cosine_ok": cos_ok,
        "euclidean_ok": euc_ok,
        "cos_pct": cos_pct,
        "euc_pct": euc_pct,
        "combined_pct": combined_pct,
        "vec_pca_1": v1[0],
        "vec_pca_2": v2[0],
        "feat1": extract_features(v1[0]),
        "feat2": extract_features(v2[0]),
    }

def make_pca_plot(v1, v2):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0f1829")
    ax.scatter(v1[0], v1[1], s=200, c="#3b82f6", zorder=5, label="Foto Lama", edgecolors="#1d4ed8", linewidths=1.5)
    ax.scatter(v2[0], v2[1], s=200, c="#f97316", zorder=5, label="Foto Sekarang", edgecolors="#c2410c", linewidths=1.5)
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], "--", color="#7a9fd4", lw=1.5, alpha=0.6)
    mid = ((v1[0]+v2[0])/2, (v1[1]+v2[1])/2)
    dist = np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
    ax.annotate(f"d={dist:.1f}", mid, fontsize=8, color="#c8d6f0", ha="center")
    ax.set_xlabel("PC 1", color="#7a9fd4", fontsize=9)
    ax.set_ylabel("PC 2", color="#7a9fd4", fontsize=9)
    ax.set_title("Proyeksi di Ruang PCA", color="#93c5fd", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#7a9fd4", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e2d4a")
    ax.legend(facecolor="#0f1829", edgecolor="#1e2d4a", labelcolor="#c8d6f0", fontsize=8)
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
    <div class="badge">PCA · SVD · EIGENFACES · COSINE & EUCLIDEAN</div>
    <p class="hero-title">🧬 FaceMatch — Deteksi Kemiripan Wajah</p>
    <p class="hero-sub">Bandingkan foto lama (masa kecil) dengan foto sekarang untuk mengetahui<br>
    apakah kedua foto tersebut adalah orang yang sama.</p>
    <div class="hero-stats">
        <div class="hero-stat"><strong>{model_data['n_train_images']:,}</strong>Foto Training</div>
        <div class="hero-stat"><strong>{model_data['n_identities']:,}</strong>Identitas</div>
        <div class="hero-stat"><strong>k = {model_data['k']}</strong>Komponen PCA</div>
        <div class="hero-stat"><strong>{model_data['explained_variance']*100:.1f}%</strong>Explained Variance</div>
        <div class="hero-stat"><strong>100×100</strong>Resolusi Gambar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ────────────────────────────────────────────────────────────────────
col_old, col_new = st.columns(2)
with col_old:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🧒</div>
        <p class="upload-label">📷 Foto Masa Lalu</p>
        <p class="upload-sub">Upload foto masa kecil / foto lama Anda</p>
    </div>
    """, unsafe_allow_html=True)
    img_old_file = st.file_uploader("Foto Lama", type=["jpg","jpeg","png","webp"], key="old", label_visibility="collapsed")

with col_new:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🧑</div>
        <p class="upload-label">📷 Foto Sekarang</p>
        <p class="upload-sub">Upload foto terbaru Anda saat ini</p>
    </div>
    """, unsafe_allow_html=True)
    img_new_file = st.file_uploader("Foto Sekarang", type=["jpg","jpeg","png","webp"], key="new", label_visibility="collapsed")

# ─── Settings ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Pengaturan Threshold (opsional)"):
    sc1, sc2 = st.columns(2)
    with sc1:
        cos_threshold = st.slider("Cosine Similarity (≥ mirip)", -1.0, 1.0, DEFAULT_COS_THRESHOLD, 0.01)
    with sc2:
        euc_threshold = st.slider("Euclidean Distance (≤ mirip)", 1.0, 80.0, DEFAULT_EUC_THRESHOLD, 0.5)
    st.markdown(f"""
    <div class="callout">
    ℹ️ Threshold default dikalibrasi dari data LFW: <b>Cosine ≥ {DEFAULT_COS_THRESHOLD}</b> dan <b>Euclidean ≤ {DEFAULT_EUC_THRESHOLD}</b>.
    Untuk foto anak-anak vs dewasa, threshold bisa dilonggarkan karena struktur wajah berubah seiring pertumbuhan.
    </div>
    """, unsafe_allow_html=True)

# ─── Main Logic ────────────────────────────────────────────────────────────────
if img_old_file and img_new_file:
    img_old = Image.open(img_old_file)
    img_new = Image.open(img_new_file)

    vec_old, det_old, box_old = detect_and_crop(img_old)
    vec_new, det_new, box_new = detect_and_crop(img_new)

    # Preview dengan bounding box
    prev1, prev2 = st.columns(2)
    with prev1:
        st.image(draw_face_box(img_old, box_old), caption=f"Foto Lama {'✅ Wajah terdeteksi' if det_old else '⚠️ Wajah tidak terdeteksi'}", use_container_width=True)
    with prev2:
        st.image(draw_face_box(img_new, box_new), caption=f"Foto Sekarang {'✅ Wajah terdeteksi' if det_new else '⚠️ Wajah tidak terdeteksi'}", use_container_width=True)

    if not det_old or not det_new:
        st.markdown("""
        <div class="callout">
        ⚠️ Satu atau kedua foto tidak terdeteksi wajahnya secara otomatis.
        Aplikasi tetap memproses menggunakan seluruh gambar, namun akurasi mungkin berkurang.
        Gunakan foto wajah frontal (menghadap lurus ke kamera) untuk hasil terbaik.
        </div>
        """, unsafe_allow_html=True)

    btn = st.button("⚡ Bandingkan Wajah", type="primary", use_container_width=True)

    if btn:
        with st.spinner("Menganalisis kemiripan wajah..."):
            result = compute_similarity(vec_old, vec_new, pca_model, cos_threshold, euc_threshold)

        cos_sim  = result["cosine_similarity"]
        euc_dist = result["euclidean_distance"]
        cos_ok   = result["cosine_ok"]
        euc_ok   = result["euclidean_ok"]
        combined = result["combined_pct"]

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ─── Verdict ──────────────────────────────────────────────────────────
        both_sim  = cos_ok and euc_ok
        both_diff = (not cos_ok) and (not euc_ok)

        if both_sim:
            verdict_class = "result-similar"
            verdict_icon  = "✅"
            verdict_title = "Kemungkinan Orang yang Sama"
            verdict_color = "#34d399"
            verdict_sub   = "Kedua metode (Cosine Similarity & Euclidean Distance) menunjukkan tingkat kemiripan yang tinggi."
        elif both_diff:
            verdict_class = "result-different"
            verdict_icon  = "❌"
            verdict_title = "Kemungkinan Bukan Orang yang Sama"
            verdict_color = "#f87171"
            verdict_sub   = "Kedua metode menunjukkan tingkat kemiripan yang rendah antara kedua foto."
        else:
            verdict_class = "result-unsure"
            verdict_icon  = "⚖️"
            verdict_title = "Hasil Tidak Konsisten"
            verdict_color = "#fbbf24"
            verdict_sub   = "Kedua metode memberikan hasil yang berbeda. Coba dengan foto yang lebih jelas dan frontal."

        pct_bar = max(0, min(100, combined))
        st.markdown(f"""
        <div class="result-box {verdict_class}">
            <div class="result-icon">{verdict_icon}</div>
            <p class="result-title" style="color:{verdict_color};">{verdict_title}</p>
            <p class="result-sub">{verdict_sub}</p>
            <div class="similarity-bar-wrap">
                <p class="similarity-label">Tingkat Kemiripan Wajah</p>
                <p class="similarity-pct">{pct_bar:.1f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Metric Detail ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <p class="metric-label">Cosine Similarity</p>
                <p class="metric-value">{cos_sim:.4f}</p>
                <p class="metric-status {'ok' if cos_ok else 'no'}">
                    Threshold ≥ {cos_threshold:.2f} &nbsp;·&nbsp; {'Mirip ✅' if cos_ok else 'Tidak Mirip ❌'}
                </p>
            </div>
            <div class="metric-box">
                <p class="metric-label">Euclidean Distance</p>
                <p class="metric-value">{euc_dist:.4f}</p>
                <p class="metric-status {'ok' if euc_ok else 'no'}">
                    Threshold ≤ {euc_threshold:.1f} &nbsp;·&nbsp; {'Mirip ✅' if euc_ok else 'Tidak Mirip ❌'}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Wajah Terdeteksi ─────────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**📊 Analisis Fitur Wajah (Foto Lama)**")
            feats1 = result["feat1"]
            max_val = max(feats1.values()) if feats1 else 1
            feat_html = '<div class="feature-card">'
            for name, val in feats1.items():
                pct = min(100, val / max_val * 100) if max_val > 0 else 0
                feat_html += f"""
                <div class="feature-item">
                    <span class="feature-name">{name}</span>
                    <div class="feature-bar-wrap"><div class="feature-bar" style="width:{pct:.0f}%"></div></div>
                    <span class="feature-val">{val:.2f}</span>
                </div>"""
            feat_html += "</div>"
            st.markdown(feat_html, unsafe_allow_html=True)

        with d2:
            st.markdown("**📊 Analisis Fitur Wajah (Foto Sekarang)**")
            feats2 = result["feat2"]
            max_val2 = max(feats2.values()) if feats2 else 1
            feat_html2 = '<div class="feature-card">'
            for name, val in feats2.items():
                pct = min(100, val / max_val2 * 100) if max_val2 > 0 else 0
                feat_html2 += f"""
                <div class="feature-item">
                    <span class="feature-name">{name}</span>
                    <div class="feature-bar-wrap"><div class="feature-bar" style="width:{pct:.0f}%"></div></div>
                    <span class="feature-val">{val:.2f}</span>
                </div>"""
            feat_html2 += "</div>"
            st.markdown(feat_html2, unsafe_allow_html=True)

        # ─── PCA Projection Plot ──────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        pc1, pc2 = st.columns([1.2, 1])
        with pc1:
            st.markdown("**📐 Proyeksi Wajah di Ruang PCA**")
            fig = make_pca_plot(result["vec_pca_1"], result["vec_pca_2"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with pc2:
            st.markdown("**🖼️ Wajah setelah Preprocessing (100×100)**")
            g1, g2 = st.columns(2)
            with g1:
                st.image(vec_old.reshape(IMG_SIZE), caption="Foto Lama", clamp=True, use_container_width=True)
            with g2:
                st.image(vec_new.reshape(IMG_SIZE), caption="Foto Sekarang", clamp=True, use_container_width=True)

        # ─── Kesimpulan ───────────────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### 📋 Kesimpulan")
        if both_sim:
            st.success(f"✅ **Orang yang Sama** — Cosine Similarity: {cos_sim:.4f} (≥{cos_threshold}) · Euclidean Distance: {euc_dist:.4f} (≤{euc_threshold}) · Kemiripan: **{pct_bar:.1f}%**")
        elif both_diff:
            st.error(f"❌ **Bukan Orang yang Sama** — Cosine Similarity: {cos_sim:.4f} (<{cos_threshold}) · Euclidean Distance: {euc_dist:.4f} (>{euc_threshold}) · Kemiripan: **{pct_bar:.1f}%**")
        else:
            st.warning(f"⚖️ **Hasil Tidak Konsisten** — Cosine: {'✅' if cos_ok else '❌'} {cos_sim:.4f} · Euclidean: {'✅' if euc_ok else '❌'} {euc_dist:.4f} · Kemiripan: **{pct_bar:.1f}%**")

        st.markdown("""
        <div class="callout">
        ⚠️ <strong>Catatan:</strong> Metode PCA/Eigenfaces sensitif terhadap perubahan wajah akibat usia,
        sudut, pencahayaan, dan ekspresi. Untuk foto masa kecil vs dewasa, perubahan struktur wajah
        alami dapat menurunkan nilai kemiripan meskipun merupakan orang yang sama.
        Hasil ini bersifat indikatif dan tidak bisa dijadikan keputusan final.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="callout" style="margin-top:1.5rem; font-size:0.9rem;">
    👆 <strong>Cara menggunakan:</strong><br>
    1. Upload <strong>Foto Masa Lalu</strong> (foto masa kecil / foto lama) di sebelah kiri<br>
    2. Upload <strong>Foto Sekarang</strong> (foto terbaru) di sebelah kanan<br>
    3. Klik tombol <strong>⚡ Bandingkan Wajah</strong><br>
    4. Lihat hasil analisis kemiripan menggunakan PCA/Eigenfaces
    </div>
    """, unsafe_allow_html=True)
