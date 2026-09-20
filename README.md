# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding
**Jaya Jaya Institut** adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan dengan reputasi yang sangat baik. Namun, masih banyak mahasiswa yang tidak menyelesaikan pendidikannya alias **dropout**. Tingginya angka dropout berdampak buruk bagi institusi — mulai dari reputasi, penilaian akreditasi, hingga pendapatan — dan tentu merugikan masa depan mahasiswa itu sendiri.

Jaya Jaya Institut ingin **mendeteksi sedini mungkin mahasiswa yang berpotensi dropout** agar dapat segera diberikan bimbingan khusus, serta memiliki dashboard untuk memahami data dan memonitor performa mahasiswa.

### Permasalahan Bisnis
1. Seberapa besar tingkat dropout di Jaya Jaya Institut?
2. Faktor apa saja (akademik, finansial, demografis, dan profil pendaftaran) yang paling berkaitan dengan dropout?
3. Bagaimana cara mendeteksi mahasiswa yang berisiko dropout sedini mungkin agar dapat diberikan bimbingan khusus?
4. Bagaimana pihak institusi dapat memantau performa mahasiswa dan faktor risiko dropout secara berkala?

### Cakupan Proyek
1. **Data understanding & EDA** terhadap dataset *Students' Performance* (4.424 mahasiswa, 37 kolom) untuk menemukan faktor-faktor utama dropout.
2. **Data preparation**: pelabelan kode kategori untuk dashboard, penentuan target biner (Dropout vs Tidak Dropout), seleksi fitur, train-test split, dan pipeline preprocessing.
3. **Modeling & evaluation**: membandingkan Logistic Regression, Random Forest, dan Gradient Boosting dengan `GridSearchCV`, lalu memilih model terbaik berdasarkan F1-score.
4. **Business dashboard** menggunakan **Metabase** (dengan database PostgreSQL) untuk memonitor performa mahasiswa.
5. **Prototype sistem machine learning** menggunakan **Streamlit** yang di-deploy ke Streamlit Community Cloud.
6. **Kesimpulan dan rekomendasi action items** bagi Jaya Jaya Institut.

### Persiapan

Sumber data: [Students' Performance – Dicoding](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance) (berasal dari UCI Machine Learning Repository: *Predict Students' Dropout and Academic Success*). Salinan dataset disertakan sebagai `data.csv` (dipisahkan dengan `;`).

Struktur proyek:
```
├── model/
│   ├── model.joblib              # pipeline preprocessing + model terbaik
│   └── model_metadata.json       # fitur, metrik evaluasi, dan info model
├── notebook.ipynb                # seluruh proses data science
├── app.py                        # prototype Streamlit
├── data.csv                      # dataset
├── sample_students.csv           # contoh data untuk fitur prediksi batch
├── metabase.db.mv.db             # database Metabase (dashboard)
├── username_dicoding-dashboard.png
├── requirements.txt
└── README.md
```

Setup environment (menggunakan **Anaconda**):
```
conda create --name jaya-jaya-institut python=3.10
conda activate jaya-jaya-institut
pip install -r requirements.txt
```

Setup environment (menggunakan **venv**):
```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Menjalankan notebook:
```
jupyter notebook notebook.ipynb
```
> Notebook dapat dijalankan tanpa database. Sel yang mengirim data ke PostgreSQL akan dilewati otomatis jika database tidak tersedia. Alamat database dapat diatur melalui environment variable `DATABASE_URL` (default: `postgresql://postgres:root123@localhost:5433/jaya_jaya_institut`).

## Business Dashboard

Dashboard **"Jaya Jaya Institut - Student Performance Dashboard"** dibuat menggunakan **Metabase** dengan sumber data tabel `students` di PostgreSQL (hasil pelabelan pada notebook). Dashboard membantu pihak institusi memahami kondisi mahasiswa dan memonitor faktor-faktor penting yang berkaitan dengan dropout.

![Dashboard](username_dicoding-dashboard.png)

Isi dashboard:
| Bagian | Visualisasi | Kegunaan |
|---|---|---|
| Ringkasan | Total mahasiswa, tingkat dropout (32,1%), tingkat kelulusan (49,9%), dan mahasiswa masih terdaftar (17,9%) | Memantau KPI utama |
| Ringkasan | Distribusi status mahasiswa & komposisi status per kelompok usia | Melihat proporsi dropout serta risiko pada mahasiswa usia dewasa |
| Faktor Finansial | Tingkat dropout berdasarkan status pelunasan biaya kuliah, tunggakan (debtor), dan beasiswa | Mengidentifikasi risiko finansial |
| Performa Akademik | Rata-rata mata kuliah lulus & nilai per semester per status, komposisi status berdasarkan approval rate semester 2 | Memantau sinyal akademik sebagai peringatan dini |
| Profil Pendaftaran & Demografi | Tingkat dropout per program studi, jalur pendaftaran, gender, waktu kuliah, dan status pernikahan | Menentukan segmen/program studi yang perlu diprioritaskan |

Dashboard dilengkapi **filter interaktif**: *Program Studi*, *Gender*, *Waktu Kuliah*, dan *Penerima Beasiswa*, sehingga seluruh grafik dapat difokuskan pada segmen tertentu.

**Akses Metabase:**
- Email: `root@mail.com`
- Password: `root123`

**Cara menjalankan dashboard (Docker):**
```
# 1. Buat network dan database PostgreSQL
docker network create jji-net
docker run -d --name jji-postgres --network jji-net -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=root123 -e POSTGRES_DB=jaya_jaya_institut -p 5433:5432 postgres:16-alpine

# 2. Isi tabel "students" dengan menjalankan notebook.ipynb (bagian Data Preparation)

# 3. Jalankan Metabase dan salin database dashboard
docker run -d --name metabase --network jji-net -p 3000:3000 metabase/metabase:v0.50.36
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
docker restart metabase
```
Buka `http://localhost:3000`, login dengan akun di atas, lalu buka dashboard **Jaya Jaya Institut - Student Performance Dashboard**.

Database dashboard diekspor dari container dengan perintah:
```
docker cp metabase:/metabase.db/metabase.db.mv.db ./
```

## Menjalankan Sistem Machine Learning

Prototype **Dropout Early Warning System** dibuat menggunakan Streamlit dan memanfaatkan model **Logistic Regression** (`model/model.joblib`). Fitur aplikasi:
1. **Prediksi Individu** — staf akademik mengisi data pendaftaran, finansial, dan performa semester 1–2 seorang mahasiswa. Aplikasi menampilkan **probabilitas dropout**, **level risiko** (Rendah < 30%, Sedang 30–49%, Tinggi ≥ 50%), **faktor yang paling memengaruhi prediksi**, serta **sinyal risiko beserta rekomendasi tindakan**. Tersedia tombol untuk mengisi contoh mahasiswa berisiko tinggi/rendah.
2. **Prediksi Batch (CSV)** — mengunggah data banyak mahasiswa sekaligus (template tersedia di aplikasi), menampilkan ringkasan jumlah mahasiswa per level risiko, tabel yang diurutkan berdasarkan probabilitas dropout, dan hasil prediksi dapat diunduh.
3. **Tentang Model** — metrik evaluasi, perbandingan model, confusion matrix, dan fitur paling berpengaruh.

Menjalankan prototype secara lokal:
```
pip install -r requirements.txt
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501`.

Link prototype (Streamlit Community Cloud): **https://jaya-jaya-institut-dropout.streamlit.app**

## Conclusion

1. **Tingkat dropout Jaya Jaya Institut tinggi, yaitu 32,1%** (1.421 dari 4.424 mahasiswa). Hampir 1 dari 3 mahasiswa tidak menyelesaikan pendidikannya.
2. **Faktor-faktor utama yang berkaitan dengan dropout:**
   - **Performa akademik semester awal** adalah faktor terkuat. Mahasiswa dropout rata-rata hanya lulus 2,6 mata kuliah di semester 1 dan 1,9 di semester 2 (lulusan: 6,2), dengan rata-rata nilai semester 2 hanya 5,9 dari 20 (lulusan: 12,7). Mahasiswa yang **tidak lulus satu pun mata kuliah di semester 2 memiliki tingkat dropout 84%**, sedangkan yang lulus seluruhnya hanya 6%.
   - **Kondisi finansial**: mahasiswa yang **biaya kuliahnya tidak lunas memiliki tingkat dropout 87%** (vs 25% yang lunas) dan mahasiswa dengan **tunggakan 62%**. Sebaliknya, **penerima beasiswa hanya 12%** dropout (vs 39% non-penerima).
   - **Profil pendaftaran & demografi**: mahasiswa yang mendaftar di **usia > 24 tahun** (dropout > 50%), jalur **Over 23 years old** (55%) dan **Holders of other higher courses** (61%), **kelas malam** (43%), dan **laki-laki** (45%) lebih rentan dropout. Program studi dengan tingkat dropout tertinggi adalah **Equinculture (55%)**, **Informatics Engineering (54%)**, **Management (evening) (51%)**, dan **Basic Education (44%)**, sedangkan **Nursing (15%)** paling rendah.
   - Faktor makroekonomi (pengangguran, inflasi, GDP) dan latar belakang orang tua hampir tidak membedakan status mahasiswa.
3. **Model machine learning** Logistic Regression dengan 19 fitur mampu mendeteksi **83,1% mahasiswa yang akan dropout** (recall) dengan precision 80,5%, **F1-score 0,818**, akurasi 88,1%, dan **ROC-AUC 0,930** pada data uji. Dari 284 mahasiswa dropout di data uji, 236 berhasil terdeteksi. Model ini dipilih dari tiga algoritma (Logistic Regression, Random Forest, Gradient Boosting) berdasarkan F1-score cross-validation.
4. Karena data semester 1–2 sangat informatif, Jaya Jaya Institut dapat **mendeteksi mahasiswa berisiko segera setelah nilai semester keluar**, lalu memantaunya melalui dashboard Metabase dan prototype Streamlit.

### Rekomendasi Action Items
- **Terapkan sistem peringatan dini setiap akhir semester.** Jalankan prediksi batch untuk seluruh mahasiswa aktif menggunakan prototype Streamlit; mahasiswa berlevel risiko **Tinggi** wajib dihubungi dosen wali dalam 1–2 minggu, sedangkan level **Sedang** dipantau secara berkala.
- **Program pendampingan akademik untuk mahasiswa dengan approval rate rendah.** Mahasiswa yang lulus < 50% mata kuliah atau memiliki rata-rata nilai < 10 di semester 1 diberikan tutor sebaya, kelas remedial, dan evaluasi beban SKS sebelum semester 2 dimulai.
- **Intervensi finansial proaktif.** Pantau status pembayaran biaya kuliah setiap bulan; tawarkan skema cicilan, penundaan pembayaran, atau dana darurat kepada mahasiswa yang menunggak sebelum mereka memutuskan berhenti.
- **Perluas dan targetkan beasiswa/bantuan biaya.** Karena penerima beasiswa jauh lebih jarang dropout, alokasikan beasiswa parsial atau beasiswa berbasis kebutuhan bagi mahasiswa berprestasi yang kesulitan finansial.
- **Skema belajar fleksibel bagi mahasiswa usia dewasa & kelas malam.** Sediakan kelas hybrid/daring, rekaman perkuliahan, serta layanan konseling dan administrasi di jam malam untuk mahasiswa jalur *Over 23 years old* dan mahasiswa yang bekerja.
- **Evaluasi program studi berisiko tinggi.** Lakukan kajian kurikulum, orientasi tahun pertama, dan penguatan bimbingan pada Equinculture, Informatics Engineering, Management (evening), dan Basic Education.
- **Gunakan dashboard sebagai alat monitoring rutin.** Pimpinan fakultas meninjau dashboard Metabase setiap bulan dan menetapkan target penurunan tingkat dropout (misalnya dari 32% menjadi < 25%) untuk dievaluasi setiap tahun ajaran.
- **Perkaya data untuk deteksi yang lebih dini.** Kumpulkan data kehadiran, aktivitas LMS, dan nilai tengah semester agar model dapat mendeteksi risiko sebelum semester pertama berakhir, lalu latih ulang model secara berkala dengan data terbaru.
