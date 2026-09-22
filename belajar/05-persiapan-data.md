# Modul 5 — Persiapan Data (Data Preparation)

> **Tujuan modul:** memahami setiap langkah yang mengubah data mentah menjadi input model: **pemisahan data pemodelan vs data prediksi**, perumusan target, seleksi fitur, pembagian data, encoding, scaling, dan pipeline. Modul ini juga membahas **data leakage**, kesalahan paling berbahaya dalam machine learning.

---

## 5.1 Merumuskan target: keputusan desain pertama (dan pelajaran dari review)

Kolom `Status` punya 3 nilai: `Dropout`, `Graduate`, dan `Enrolled`. Setidaknya ada 3 cara merumuskan masalahnya:

| Opsi | Target | Kelebihan | Kekurangan |
|---|---|---|---|
| A. Multikelas | Dropout / Enrolled / Graduate | informasi lengkap | "Enrolled" bukan hasil akhir; metrik lebih rumit |
| **B. Dipakai sekarang ✅** | **Dropout (1) vs Graduate (0)**; Enrolled dipisahkan sebagai **data prediksi** | label **pasti dan tidak ambigu**; mencerminkan cara sistem dipakai di dunia nyata | data latih berkurang 794 baris |
| C. Versi awal ❌ | Dropout (1) vs "Tidak Dropout" (Graduate + Enrolled = 0) | memakai semua data | **target ambigu**: sebagian Enrolled mungkin kelak dropout, tetapi model diajari bahwa mereka "tidak dropout" |

### 📌 Kisah nyata proyek ini
Versi pertama proyek memakai **opsi C**, dan submission-nya **ditolak reviewer Dicoding** dengan alasan:

> Enrolled tidak seharusnya dilibatkan dalam training, karena tujuannya memprediksi apakah mahasiswa akan **Dropout atau Graduate**. Menyamakan Graduate dan Enrolled menjadi 0 membuat target **ambigu** dan dapat menurunkan validitas model.

Kenapa reviewer benar? Mahasiswa `Enrolled` **belum punya status akhir**. Jika mereka diberi label 0, sebagian label itu **salah**: mahasiswa yang kelak dropout ikut diajarkan ke model sebagai "tidak dropout". Model belajar dari **label yang kotor**.

**Buktinya ada di data:** pada model versi lama, **42 dari 57 false positive** adalah mahasiswa Enrolled. Model menilai mereka mirip mahasiswa dropout (dan mungkin memang benar), tetapi karena labelnya 0, hal itu dihitung sebagai "kesalahan". Label yang ambigu mengacaukan proses belajar **dan** proses evaluasi.

Perbaikannya (opsi B):

```python
df_model = df[df["Status"].isin(["Dropout", "Graduate"])].copy()   # status akhir sudah pasti
df_enrolled = df[df["Status"] == "Enrolled"].copy()                  # status akhir belum diketahui → diprediksi

TARGET = "is_dropout"          # 1 = Dropout, 0 = Graduate
y = df_model[TARGET]           # 1.421 bernilai 1, 2.209 bernilai 0 (39,15% dropout)
```

```mermaid
flowchart LR
    D[data.csv<br/>4.424 mahasiswa] --> M[df_model<br/>Dropout + Graduate<br/>3.630]
    D --> E[df_enrolled<br/>Enrolled<br/>794]
    M --> T[train 2.904 / test 726<br/>→ latih & evaluasi model]
    T --> BM[Model terbaik]
    BM --> P[Prediksi risiko<br/>794 mahasiswa aktif]
    E --> P
```

Dampaknya justru **positif**: F1 naik dari 0,818 menjadi **0,907**, dan ROC-AUC dari 0,930 menjadi **0,973**. Label yang bersih membuat pola dropout vs lulus lebih tegas.

> 💡 **Prinsip:** data latih hanya boleh berisi contoh yang **jawabannya sudah pasti**. Data yang jawabannya belum diketahui adalah **data yang akan diprediksi**, bukan data untuk belajar.

## 5.2 Seleksi fitur: dari 36 menjadi 19

**Kriteria pemilihan:**
1. **Relevan**: berbeda jelas antara mahasiswa dropout dan lulus (lihat hasil EDA).
2. **Tersedia dan bisa dipantau** oleh institusi.
3. **Praktis diinput** di aplikasi. Form dengan 36 isian akan melelahkan staf.

**Yang dikeluarkan dan alasannya:**
| Fitur | Alasan |
|---|---|
| 4 kolom orang tua (29–46 kategori) | korelasi sangat lemah, one-hot-nya menghasilkan sangat banyak kolom |
| `Nacionality`, `International` | 97,5% mahasiswa Portugis, sehingga hampir tidak ada variasi |
| `Unemployment_rate`, `Inflation_rate`, `GDP` | tidak membedakan status; tidak bisa dikendalikan institusi |
| `Marital_status`, `Previous_qualification`, `Application_order`, `Educational_special_needs` | sinyal lemah |
| `..._credited`, `..._without_evaluations` | sinyal lemah, sebagian besar bernilai 0 |

**19 fitur yang dipakai:**
- Kategori (2): `Application_mode`, `Course`
- Numerik/biner (17): `Daytime_evening_attendance`, `Previous_qualification_grade`, `Admission_grade`, `Displaced`, `Debtor`, `Tuition_fees_up_to_date`, `Gender`, `Scholarship_holder`, `Age_at_enrollment`, serta `enrolled`, `evaluations`, `approved`, `grade` untuk semester 1 dan 2.

**Pembuktian dengan eksperimen** (5-fold cross-validation pada data Dropout vs Graduate, Logistic Regression):

| Set fitur | Jumlah | F1 | Recall | ROC-AUC |
|---|---|---|---|---|
| Seluruh fitur | 36 | 0,878 | 0,868 | 0,954 |
| **Fitur terpilih** | **19** | **0,880** | 0,868 | 0,953 |

Hasilnya praktis sama, tapi modelnya jauh lebih sederhana. Seleksi fitur yang baik **tidak hanya berdasarkan intuisi**, tetapi dibuktikan dengan angka.

> 🧠 **Prinsip parsimoni (Occam's razor):** jika dua model sama baiknya, pilih yang lebih sederhana. Model sederhana lebih mudah dijelaskan, lebih murah dirawat, dan lebih kecil risikonya "menghafal" data.

## 5.3 Membagi data: train dan test

```python
X = df_model[SELECTED_FEATURES]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

| Parameter | Arti |
|---|---|
| `test_size=0.2` | 20% data (726 mahasiswa) disimpan untuk ujian akhir, 80% (2.904) untuk belajar |
| `stratify=y` | proporsi dropout di train dan test **sama**: 39,15% vs 39,12% |
| `random_state=42` | pengacakan bisa diulang dengan hasil sama (*reproducible*) |

**Kenapa perlu data uji?** Model harus dinilai pada data yang **belum pernah dilihatnya**, sama seperti ujian yang soalnya berbeda dari latihan. Kalau dinilai pada data latih, skornya akan terlalu optimis.

Jumlah persisnya: data latih berisi 1.137 dropout dan 1.767 graduate. Data uji berisi 284 dropout dan 442 graduate.

> Perhatikan perbedaan **data uji** dan **data prediksi**: data uji (726) **punya jawaban**, dipakai untuk mengukur performa. Data prediksi (794 Enrolled) **belum punya jawaban**, dipakai untuk menghasilkan peringatan.

## 5.4 ⚠️ Data leakage: kebocoran informasi

**Data leakage** terjadi ketika informasi dari luar data latih (misalnya dari data uji atau dari masa depan) ikut masuk ke proses pelatihan. Akibatnya skor evaluasi tampak bagus tapi gagal di dunia nyata.

Dua jenis yang relevan di proyek ini:

**1. Leakage dari preprocessing.** Contoh yang **salah**:
```python
scaler.fit(X)                         # ❌ mean & std dihitung dari SELURUH data (termasuk test)
X_train, X_test = train_test_split(...)
```
Rata-rata data uji ikut "bocor" ke data latih. Solusinya: fit scaler **hanya pada data latih**. Di proyek ini hal itu dijamin oleh **Pipeline** (bagian 5.7). Di dalam cross-validation pun, pipeline di-fit ulang di setiap fold hanya dengan data fold latih.

**2. Leakage waktu (target leakage).** Fitur yang baru tersedia **setelah** atau **bersamaan dengan** kejadian yang ingin diprediksi. Model ini memakai data **semester 2**. Artinya model baru bisa dipakai **setelah semester 2 selesai**, bukan saat mahasiswa baru masuk. Ini bukan kesalahan, tetapi batasan penting yang harus disampaikan dengan jujur ([Modul 10](10-bisnis-kesimpulan-etika.md)). Model yang hanya memakai data semester 1 masih mencapai **F1 0,857** (lihat notebook, *Evaluation → 4. Model deteksi dini*), jadi deteksi bisa dilakukan satu semester lebih awal dengan sedikit penurunan akurasi.

## 5.5 One-Hot Encoding untuk kategori

One-hot mengubah satu kolom kategori menjadi beberapa kolom 0/1, satu untuk setiap kategori:

| Course (asli) | Course_9500 | Course_9119 | Course_9130 | … |
|---|---|---|---|---|
| 9500 (Nursing) | 1 | 0 | 0 | … |
| 9119 (Informatics) | 0 | 1 | 0 | … |

```python
OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
```

- `min_frequency=20`: kategori yang muncul **kurang dari 20 kali** di data latih digabung menjadi satu kolom `infrequent`. Contohnya prodi Biofuel (12 mahasiswa) dan jalur-jalur pendaftaran langka. Tujuannya supaya model tidak "menghafal" kategori yang datanya terlalu sedikit untuk dipercaya.
- `handle_unknown="infrequent_if_exist"`: jika aplikasi menerima kode yang **tidak pernah dilihat** saat latihan, kode itu dimasukkan ke kolom `infrequent` dan tidak menyebabkan error.

## 5.6 Standardisasi (StandardScaler)

$$z = \frac{x - \text{mean}}{\text{std}}$$

Mean dan std dihitung dari **data latih** (2.904 mahasiswa Dropout + Graduate). Contoh nyata dari model yang tersimpan:

| Fitur | Mean | Std | Contoh x | z |
|---|---|---|---|---|
| `Admission_grade` | 127,34 | 14,65 | 150 | (150 − 127,34) / 14,65 = **+1,55** |
| `Age_at_enrollment` | 23,48 | 7,85 | 29 | (29 − 23,48) / 7,85 = **+0,70** |
| `Curricular_units_2nd_sem_approved` | 4,55 | 3,14 | 0 | (0 − 4,55) / 3,14 = **−1,45** |

**Kenapa perlu scaling untuk Logistic Regression?**
1. Regularisasi (parameter `C`) menghukum besarnya koefisien. Tanpa scaling, fitur berskala besar (nilai masuk 95–190) dan kecil (biner 0/1) diperlakukan tidak adil.
2. Koefisien bisa **dibandingkan** antar fitur, karena semuanya dalam satuan "per 1 standar deviasi".
3. Proses optimasi lebih cepat konvergen.

> Random Forest dan Gradient Boosting **tidak** membutuhkan scaling karena bekerja dengan memotong nilai (split), bukan mengalikannya. Tapi scaling juga tidak merugikan mereka, jadi pipeline yang sama bisa dipakai untuk ketiga model.

## 5.7 ColumnTransformer + Pipeline

```python
def build_preprocessor(categorical, numerical):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), categorical),
        ("numerical", StandardScaler(), numerical),
    ])

pipe = Pipeline([
    ("preprocessor", build_preprocessor(CATEGORICAL_FEATURES, NUMERICAL_FEATURES)),
    ("model", LogisticRegression(...)),
])
```

```mermaid
flowchart LR
    R[Data mentah<br/>19 kolom] --> CT{ColumnTransformer}
    CT -->|Application_mode, Course| OH[OneHotEncoder]
    CT -->|17 kolom numerik| SS[StandardScaler]
    OH --> J[46 kolom]
    SS --> J
    J --> M[LogisticRegression]
    M --> P[Probabilitas dropout]
```

- **ColumnTransformer** memberi perlakuan berbeda untuk kolom yang berbeda, lalu menggabungkan hasilnya. 2 kolom kategori menjadi 29 kolom one-hot, ditambah 17 kolom numerik, **total 46 fitur**.
- **Pipeline** merangkai langkah-langkah itu menjadi **satu objek**:
  - `pipe.fit(X_train, y_train)`: fit encoder dan scaler **hanya** pada data latih, lalu melatih model.
  - `pipe.predict_proba(X_baru)`: otomatis menerapkan transformasi yang sama, lalu memprediksi. Inilah yang dipakai untuk memprediksi 794 mahasiswa Enrolled.
  - Disimpan sebagai satu file (`model.joblib`), sehingga **aplikasi cukup memberi data mentah**.

Nama fitur hasil transformasi bisa dilihat dengan `pipe.named_steps["preprocessor"].get_feature_names_out()`. Formatnya `categorical__Course_9500` dan `numerical__Admission_grade`.

## 5.8 Dataset khusus dashboard

Model butuh angka, tetapi **manusia butuh label**. Karena itu notebook membuat `df_dashboard`. Dashboard memakai **seluruh 4.424 mahasiswa** (termasuk Enrolled), karena tujuannya memantau kondisi semua mahasiswa, bukan melatih model.

```python
"course": df["Course"].map(COURSE_MAP),                    # 9500 → "Nursing"
"gender": df["Gender"].map({0: "Female", 1: "Male"}),
"age_group": df["Age_group"].astype(str),                  # fitur turunan
"sem2_approval_rate": df["Approval_rate_2nd_sem"].round(4) # fitur turunan
```

Lalu dataset ini dikirim ke PostgreSQL:

```python
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:root123@localhost:5433/jaya_jaya_institut")
engine = create_engine(DATABASE_URL)
df_dashboard.to_sql("students", engine, if_exists="replace", index=False)
```

- `os.getenv(..., default)`: alamat database bisa diganti lewat environment variable, tanpa mengubah kode. Ini praktik yang baik untuk menyimpan kredensial.
- `if_exists="replace"`: menjalankan ulang notebook tidak membuat data dobel.
- Dibungkus `try/except`: notebook tetap bisa dijalankan orang lain yang tidak punya database.

---

## ✍️ Cek pemahaman

1. Kenapa mahasiswa Enrolled tidak boleh diberi label 0 saat melatih model?
2. Apa beda **data uji** dan **data prediksi** di proyek ini?
3. Kenapa `stratify=y` penting pada data yang tidak seimbang?
4. Berikan contoh data leakage yang mungkin terjadi jika Anda menghitung `StandardScaler` sebelum `train_test_split`.
5. Sebuah prodi baru dengan kode 9999 dimasukkan ke aplikasi. Apa yang terjadi di one-hot encoder?
6. Mahasiswa berusia 18 tahun. Berapa nilai z-nya?
7. Kenapa 2 kolom kategori bisa menjadi 29 kolom, bukan 17 + 18 = 35?

<details>
<summary>Lihat jawaban</summary>

1. Status akhirnya belum diketahui. Sebagian dari mereka mungkin kelak dropout, sehingga label 0 bisa salah. Model akan belajar dari label kotor (target ambigu), dan validitasnya turun.
2. Data uji (726 mahasiswa Dropout/Graduate) **punya jawaban** dan dipakai untuk mengukur performa model. Data prediksi (794 mahasiswa Enrolled) **belum punya jawaban** dan dipakai untuk menghasilkan peringatan dini.
3. Tanpa stratify, pembagian acak bisa kebetulan membuat proporsi dropout di test jauh berbeda (misalnya 35% vs 43%), sehingga evaluasi tidak mewakili kondisi sebenarnya.
4. Mean dan std dihitung dari seluruh data, termasuk data uji. Informasi tentang sebaran data uji "bocor" ke data latih, sehingga skor evaluasi sedikit terlalu optimis.
5. Kode itu dimasukkan ke kolom `infrequent` (karena `handle_unknown="infrequent_if_exist"`), jadi tidak terjadi error.
6. (18 − 23,48) / 7,85 ≈ **−0,70**.
7. Karena `min_frequency=20`: kategori langka digabung menjadi satu kolom `infrequent` per fitur, sehingga jumlah kolomnya berkurang.
</details>

➡️ Lanjut ke [Modul 6 — Machine learning](06-machine-learning.md)
