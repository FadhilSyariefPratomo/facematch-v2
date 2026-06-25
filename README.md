# 🧬 FaceMatch — Analisis Identitas Wajah Geometris

Aplikasi web interaktif berbasis Streamlit untuk membandingkan tingkat kemiripan dua foto wajah (misal: arsip masa kecil vs. foto saat ini) menggunakan algoritma **Principal Component Analysis (PCA/Eigenfaces)**.

Aplikasi ini menggunakan model yang telah dilatih (*pre-trained*) `pca_model.pkl` dari dataset **LFW (6.562 foto, 1.680 identitas, k=100)**. Pengguna **tidak perlu** melakukan *training* apa pun dan dapat langsung menggunakan aplikasi.

### ✨ Fitur Unggulan (v2.0)
* **Neo-Brutalism UI:** Antarmuka grid yang modern, interaktif, dan responsif.
* **Real-time Preview:** Pratinjau *bounding box* wajah seketika setelah foto diunggah menggunakan *session state caching* agar performa tetap ringan.
* **Robust Low-Res Detection:** Dilengkapi dengan *CLAHE (Contrast Limited Adaptive Histogram Equalization)* dan *Dynamic Upscaling* interpolasi kubik untuk mengoptimalkan pembacaan foto arsip beresolusi rendah.
* **Dual Metrics:** Mengkombinasikan *Cosine Similarity* (sudut orientasi fitur) dan *Euclidean Distance* (deviasi jarak absolut) untuk akurasi yang lebih ketat.

---

## 🚀 Cara Menjalankan Secara Lokal

Disarankan untuk menggunakan *Virtual Environment* agar dependensi pustaka tidak bentrok dengan proyek Python lainnya di komputer Anda.

### 1. Buat Virtual Environment (Opsional tapi sangat disarankan)
Buka terminal di dalam folder proyek, lalu jalankan:
```bash
python -m venv venv
```

### 2. Aktifkan Virtual Environment
* **Windows:**
  ```bash
  venv\Scripts\activate
  ```
* **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```
*(Pastikan muncul indikator `(venv)` di sebelah kiri input terminal Anda).*

### 3. Install Dependensi
Setelah environment aktif, instal semua pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser melalui alamat `http://localhost:8501`.

---

## 🗂️ Struktur File

```text
facematch-v2/
├── app.py            ← Aplikasi utama (Streamlit UI & Logic)
├── pca_model.pkl     ← Model PCA siap pakai (WAJIB ADA)
├── requirements.txt  ← Daftar dependensi pustaka Python
├── packages.txt      ← Dependensi sistem OS (untuk keperluan deploy)
└── README.md         ← Panduan proyek
```

⚠️ **PENTING:** Jangan hapus atau pindahkan `pca_model.pkl` — sistem komputasi matriks tidak akan berjalan tanpa *file* ini.

---

## 🧩 Cara Penggunaan

1. Upload **Foto Masa Lalu** di panel kiri.
2. Upload **Foto Sekarang** di panel kanan.
3. Pastikan indikator pratinjau menunjukkan **✓ DETEKSI WAJAH**.
4. (Opsional) Sesuaikan nilai ambang batas pada **Parameter Interaktif**:
   * **Cosine Similarity** (Target default: ≥ 0.70)
   * **Euclidean Distance** (Target default: ≤ 15.0)
5. Klik tombol **MULAI ANALISIS WAJAH ⚡**.
6. Sistem akan mengekstrak ciri geometris dan memproyeksikannya ke dalam Ruang PCA untuk memberikan *Global Similarity Score*. Jika hasil metrik saling bertentangan, sistem akan melabelinya sebagai "Hasil Konflik".

*(Catatan: Menggeser slider parameter setelah hasil muncul akan memperbarui kalkulasi persentase secara real-time tanpa perlu menekan tombol ulang).*

---

## ⚠️ Keterbatasan Sistem (Metode Holistik)

Metode PCA/Eigenfaces membaca distribusi intensitas piksel secara holistik, sehingga memiliki sensitivitas tinggi terhadap:
* Perubahan ekstrem pada struktur rahang dan tengkorak akibat pertumbuhan usia yang terlalu jauh (misal: balita ke dewasa).
* Pose atau sudut wajah yang tidak menghadap lurus ke depan (*frontal*).
* Perbedaan pencahayaan asimetris yang terlalu kontras antara bayangan terang dan gelap.

Oleh karena itu, hasil persentase kemiripan lintas usia ini bersifat **indikatif statistika**, bukan sebagai keputusan forensik atau identifikasi biometrik final yang sah.

---

## 🛠️ Panduan Deploy ke Streamlit Community Cloud

1. Buat *repository* GitHub baru dengan visibilitas *Public*.
2. Unggah **semua file** (termasuk `pca_model.pkl`) ke dalam *repository* tersebut. Pastikan folder *Virtual Environment* (seperti `venv`) tidak ikut terunggah dengan menggunakan file `.gitignore`.
3. Buka [share.streamlit.io](https://share.streamlit.io) dan masuk menggunakan akun GitHub.
4. Klik **New app** → pilih *repository* → atur *main file path* ke `app.py`.
5. Klik **Deploy** dan tunggu proses instalasi dependensi selesai.