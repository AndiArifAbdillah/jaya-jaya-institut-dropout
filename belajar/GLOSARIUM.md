# 📖 Glosarium

Kamus istilah yang dipakai di proyek dan materi belajar ini, diurutkan menurut abjad. Angka dalam kurung menunjukkan modul yang membahasnya lebih dalam.

---

**Accuracy (akurasi)**: persentase tebakan yang benar dari semua tebakan. Menyesatkan jika kelas tidak seimbang. (7)

**Agregasi**: meringkas banyak baris menjadi satu angka, misalnya `mean`, `COUNT`, `AVG`. (2, 9)

**Algoritma**: metode yang dipakai komputer untuk belajar dari data, misalnya Logistic Regression. (0, 6)

**Anomali / outlier**: data yang tidak wajar atau jauh berbeda dari kebanyakan data. (0, 3, 4)

**API (Application Programming Interface)**: cara sebuah program berkomunikasi dengan program lain. Dashboard Metabase dibangun lewat API-nya. (0, 9)

**Approval rate**: rasio mata kuliah lulus ÷ mata kuliah diambil. Fitur turunan dengan korelasi terkuat (−0,66). (2, 4)

**AUC (Area Under the ROC Curve)**: peluang model memberi skor lebih tinggi kepada contoh positif dibanding contoh negatif yang dipilih acak. Proyek ini: 0,930. (7)

**Bagging**: melatih banyak model pada sampel acak lalu menggabungkan suaranya. Dasar Random Forest. (6)

**Baseline**: model paling sederhana sebagai pembanding, misalnya "selalu tebak tidak dropout" (akurasi 67,9%). (3, 7)

**Bias (statistik)**: kesalahan sistematis yang membuat hasil selalu condong ke satu arah. (7)

**Bivariat**: analisis yang melihat hubungan dua variabel. (0)

**Boosting**: melatih model secara berurutan, setiap model memperbaiki kesalahan model sebelumnya. Dasar Gradient Boosting. (6)

**Boxplot**: grafik yang menunjukkan median, kuartil, dan outlier. (4)

**Business Intelligence (BI)**: praktik mengolah data menjadi laporan/dashboard untuk keputusan bisnis. (0, 9)

**Cache**: menyimpan hasil perhitungan supaya tidak dihitung ulang (`st.cache_resource`, `st.cache_data`). (8)

**Class imbalance (ketidakseimbangan kelas)**: jumlah contoh antar-kelas tidak seimbang (32% vs 68%). (3, 6)

**`class_weight="balanced"`**: memberi bobot lebih besar pada kelas minoritas saat pelatihan. (6)

**Classification (klasifikasi)**: memprediksi kategori, misalnya dropout / tidak. (0, 6)

**Coefficient (koefisien)**: bobot setiap fitur dalam model linier. (6)

**ColumnTransformer**: menerapkan preprocessing berbeda pada kolom berbeda. (5)

**Confounder (perancu)**: variabel ketiga yang memengaruhi dua variabel lain sehingga keduanya tampak berhubungan. Contoh: usia → kelas malam & dropout. (4)

**Confusion matrix**: tabel 2×2 berisi TP, TN, FP, FN. (7)

**Container (Docker)**: instance dari image yang sedang berjalan. (9)

**Correlation (korelasi)**: ukuran kekuatan hubungan linier dua variabel (−1 sampai 1). (4)

**CRISP-DM**: kerangka kerja 6 tahap proyek data mining/data science. (1)

**Cross-validation (validasi silang)**: membagi data latih menjadi k bagian dan bergantian menguji di setiap bagian. (6)

**CSV (Comma-Separated Values)**: format file tabel teks. Dataset proyek ini memakai pemisah `;`. (2)

**Dashboard**: tampilan satu layar berisi grafik dan KPI untuk memantau kondisi. (0, 9)

**Data drift**: perubahan pola data dari waktu ke waktu sehingga model perlu diperbarui. (10)

**Data leakage**: informasi dari luar data latih (data uji/masa depan) ikut masuk ke pelatihan. (5)

**DataFrame**: struktur tabel di pandas. (2)

**Dataset**: kumpulan data yang tersusun, biasanya berupa tabel. (0)

**Decision tree**: model berupa rangkaian pertanyaan ya/tidak. (6)

**Deployment**: menjadikan model bisa dipakai oleh pengguna nyata. (0, 8)

**Docker**: platform untuk menjalankan software dalam container yang terisolasi. (0, 9)

**Dropout**: mahasiswa yang keluar sebelum menyelesaikan studi. Target proyek ini.

**EDA (Exploratory Data Analysis)**: proses menjelajahi data dengan statistik ringkas dan visualisasi sebelum membuat model. (0, 4)

**Encoding**: mengubah kategori menjadi angka yang bisa diproses model. (5)

**F1-score**: rata-rata harmonik precision dan recall. Proyek ini: 0,818. (7)

**False Negative (FN)**: mahasiswa yang akan dropout tetapi diprediksi aman. Kesalahan paling mahal di proyek ini. (7)

**False Positive (FP)**: mahasiswa aman tetapi diprediksi dropout. (7)

**Feature (fitur)**: variabel input untuk model. (0)

**Feature engineering**: membuat fitur baru dari data yang ada. (0, 5)

**Feature importance**: ukuran seberapa besar pengaruh sebuah fitur terhadap prediksi. (7)

**Feature selection**: memilih subset fitur yang paling berguna. (5)

**Field filter**: filter Metabase yang terhubung ke kolom database (`{{course}}`). (9)

**Generalisasi**: kemampuan model bekerja baik pada data baru. (0, 6)

**GitHub**: layanan penyimpanan kode berbasis Git. (0, 8)

**Gradient Boosting**: algoritma boosting berbasis pohon. (6)

**GridSearchCV**: mencoba semua kombinasi hyperparameter dengan cross-validation. (6)

**`GROUP BY`**: perintah SQL untuk mengelompokkan baris. (9)

**`HAVING`**: menyaring kelompok hasil agregasi di SQL. (9)

**Histogram**: grafik sebaran nilai numerik dalam interval. (4)

**Human-in-the-loop**: keputusan akhir tetap diambil manusia, model hanya membantu. (10)

**Hyperparameter**: pengaturan model yang ditentukan sebelum pelatihan (`C`, `max_depth`). (6)

**Image (Docker)**: cetakan/resep untuk membuat container. (9)

**Inference**: memakai model terlatih untuk memprediksi data baru. (6)

**Insight**: kesimpulan bermakna dari data: temuan + angka + perbandingan + implikasi. (0)

**Intercept**: konstanta dalam model linier (b₀ = −0,181). (6)

**IQR (Interquartile Range)**: Q3 − Q1. (4)

**joblib**: library untuk menyimpan/memuat objek Python (model). (8)

**KPI (Key Performance Indicator)**: angka utama yang dipantau. (0, 9)

**L1 / L2 regularization**: hukuman terhadap koefisien besar. L1 bisa membuat koefisien nol. (6, 11)

**Label**: nilai target yang sudah diketahui. (0)

**Log-loss (cross-entropy)**: fungsi error yang diminimalkan Logistic Regression. (6)

**Log-odds**: ln(p / (1 − p)). Besaran yang dimodelkan secara linier oleh Logistic Regression. (6)

**Logistic Regression**: model klasifikasi linier dengan output probabilitas lewat fungsi sigmoid. (6)

**Mean / median**: rata-rata / nilai tengah. (4)

**Metabase**: alat BI open-source untuk membuat dashboard. (9)

**Metadata**: data tentang data, misalnya `model_metadata.json` (daftar fitur, metrik). (8)

**Model**: hasil algoritma yang sudah belajar dari data. (0)

**Multikolinearitas**: kondisi ketika beberapa fitur saling berkorelasi kuat. (6)

**Multivariat**: analisis yang melibatkan tiga variabel atau lebih. (0)

**Odds**: p / (1 − p). **Odds ratio** = e^koefisien. (6)

**One-Hot Encoding**: satu kolom kategori menjadi banyak kolom 0/1. (5)

**Ordinal**: kategori yang punya urutan. (3)

**Overfitting**: model menghafal data latih sehingga buruk pada data baru. (0, 6)

**pandas**: library Python untuk mengolah data tabel. (2)

**Parameter**: nilai yang dipelajari model dari data (koefisien). (6)

**Permutation importance**: mengukur kepentingan fitur dengan mengacak nilainya dan melihat penurunan skor. (7)

**Pickle**: format penyimpanan objek Python. Terikat versi library. (8)

**Pipeline (scikit-learn)**: rangkaian langkah preprocessing + model dalam satu objek. (5)

**Point-biserial correlation**: korelasi antara variabel numerik dan biner. (4)

**Port mapping**: menghubungkan port laptop ke port container (`5433:5432`). (9)

**PostgreSQL**: sistem database relasional open-source. (9)

**Precision**: dari yang diprediksi positif, berapa yang benar positif. Proyek ini: 0,805. (7)

**Preprocessing**: mengubah format data agar siap dipakai model. (0, 5)

**Probabilitas**: angka 0–1 yang menyatakan keyakinan model. (6)

**Prototype**: versi awal aplikasi untuk menunjukkan konsep. (0, 8)

**Random Forest**: kumpulan banyak decision tree hasil bagging. (6)

**`random_state`**: angka acak tetap agar hasil bisa diulang. (5)

**Recall (sensitivity)**: dari yang benar-benar positif, berapa yang tertangkap. Proyek ini: 0,831. (7)

**Regularisasi**: teknik mencegah overfitting dengan menghukum model yang terlalu kompleks. (6)

**Reproducible**: bisa diulang dengan hasil yang sama. (5)

**ROC curve**: kurva TPR vs FPR untuk semua threshold. (7)

**Scaling / standardisasi**: mengubah skala fitur menjadi z = (x − mean) / std. (5)

**scikit-learn**: library machine learning Python. (5–7)

**Series**: satu kolom data di pandas. (2)

**Session state**: penyimpanan nilai antar-rerun di Streamlit. (8)

**Sigmoid**: σ(z) = 1 / (1 + e^−z), mengubah skor menjadi probabilitas. (6)

**Specificity**: dari yang benar-benar negatif, berapa yang diprediksi negatif (0,905). (7)

**SQL (Structured Query Language)**: bahasa untuk mengambil dan mengolah data di database. (0, 9)

**Standard deviation (std)**: ukuran sebaran data dari rata-rata. (4)

**Standard error (SE)**: ukuran ketidakpastian sebuah estimasi, misalnya proporsi. (4)

**Stratify / stratified**: menjaga proporsi kelas tetap sama saat membagi data. (5, 6)

**Streamlit**: library Python untuk membuat aplikasi web data dengan cepat. (8)

**Supervised learning**: belajar dari data yang sudah berlabel. (6)

**Target**: variabel yang ingin diprediksi (`is_dropout`). (0)

**Template tag**: variabel `{{...}}` dalam query SQL Metabase. (9)

**Test set (data uji)**: data yang disimpan untuk penilaian akhir (885 mahasiswa). (5)

**Threshold**: batas probabilitas untuk memutuskan kelas (default 0,5). (7)

**Train set (data latih)**: data untuk melatih model (3.539 mahasiswa). (5)

**True Negative (TN) / True Positive (TP)**: prediksi negatif/positif yang benar. (7)

**Underfitting**: model terlalu sederhana sehingga buruk bahkan di data latih. (6)

**`UNION ALL`**: menumpuk hasil dua query SQL. (9)

**Univariat**: analisis satu variabel. (0)

**Unpivot / melt**: mengubah kolom-kolom menjadi baris. (9)

**Virtual environment (venv)**: lingkungan Python terisolasi dengan library sendiri. (8)

**Wheel**: paket Python siap pakai yang tidak perlu di-compile. Ketiadaannya untuk Python 3.14 membuat deploy macet. (8)

**z-score**: nilai setelah standardisasi. (5)
