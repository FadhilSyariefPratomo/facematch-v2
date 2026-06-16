# 🧬 FaceMatch — Bandingkan Foto Lama vs Sekarang

Aplikasi web sederhana berbasis Streamlit untuk membandingkan dua foto wajah
(foto lama/masa kecil vs foto sekarang) menggunakan **PCA/Eigenfaces**,
**Cosine Similarity**, dan **Euclidean Distance**.

Model PCA sudah dilatih sebelumnya (`pca_model.pkl`) dari dataset
**LFW (6.562 foto, 1.680 identitas, k=100)** — pengguna **tidak perlu**
melakukan training apa pun.

---

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 🗂️ Struktur File

```
facematch_v2/
├── app.py            ← Aplikasi utama
├── pca_model.pkl     ← Model PCA siap pakai (WAJIB ADA)
├── requirements.txt  ← Dependensi Python
├── packages.txt      ← Dependensi sistem (untuk deploy)
└── README.md         ← Panduan ini
```

⚠️ **Jangan hapus atau pindahkan `pca_model.pkl`** — aplikasi tidak akan
berjalan tanpa file ini.

---

## 🧩 Cara Pakai

1. Upload **Foto Lama** (foto masa kecil/lama)
2. Upload **Foto Sekarang** (foto terbaru)
3. Klik **Bandingkan Wajah**
4. Lihat hasil:
   - **Cosine Similarity** ≥ 0.80 → indikasi mirip
   - **Euclidean Distance** ≤ 15.0 → indikasi mirip
   - Jika **kedua metode sepakat**, sistem memberi kesimpulan akhir
   - Jika hasil berbeda, sistem menandai sebagai "Tidak Konsisten"

Threshold dapat diubah di bagian **Pengaturan Lanjutan**.

---

## ⚠️ Keterbatasan

Metode PCA/Eigenfaces sensitif terhadap:
- Perubahan struktur wajah karena pertumbuhan (anak → dewasa)
- Sudut wajah, pencahayaan, ekspresi
- Wajah yang belum di-*align*

Untuk perbandingan foto anak-anak vs dewasa, hasil bersifat **indikatif**,
bukan keputusan final. Ini sesuai dengan keterbatasan teoritis PCA yang
dijelaskan pada dokumen panduan.

---

## 🛠️ Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru (public)
2. Upload **semua file** di folder ini (termasuk `pca_model.pkl`)
3. Buka [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
4. New app → pilih repo → main file `app.py` → Deploy

`pca_model.pkl` ukurannya ~8 MB, aman untuk diupload ke GitHub.
