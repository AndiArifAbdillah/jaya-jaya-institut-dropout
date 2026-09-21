# Modul 6 — Machine Learning: dari Konsep sampai Rumus

> **Tujuan modul:** memahami cara kerja ketiga algoritma yang dibandingkan, terutama **Logistic Regression** sampai ke rumus dan cara membaca koefisiennya, serta memahami cross-validation dan tuning hyperparameter.

---

## 6.1 Konsep dasar

- **Machine learning** = membuat komputer menemukan pola dari data, bukan dari aturan yang ditulis manual.
- **Supervised learning** = belajar dari contoh yang **sudah punya jawaban** (label). Di sini: 3.539 mahasiswa latih yang statusnya sudah diketahui.
- **Klasifikasi** = memprediksi kategori (dropout / tidak). Kebalikannya adalah **regresi**, yang memprediksi angka.
- **Fitur (X)** = input (19 kolom). **Label/target (y)** = output (`is_dropout`).
- **Training** = mencari parameter model yang membuat prediksi paling cocok dengan label.
- **Inference/prediksi** = memakai model yang sudah dilatih pada data baru.

## 6.2 Logistic Regression — model terpilih

### Langkah 1: skor linier
Setiap fitur (yang sudah distandardisasi) dikalikan dengan **koefisien** (bobot), lalu dijumlahkan bersama **intercept**:

$$z = b_0 + w_1x_1 + w_2x_2 + \dots + w_{47}x_{47}$$

### Langkah 2: fungsi sigmoid
Skor z bisa bernilai berapa saja (−∞ sampai +∞), padahal probabilitas harus di antara 0 dan 1. Sigmoid "menekuk" z ke rentang itu:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

| z | p |
|---|---|
| −4 | 0,018 |
| −2 | 0,119 |
| **0** | **0,500** |
| +2 | 0,881 |
| +4 | 0,982 |

z = 0 menghasilkan p = 0,5, yaitu **threshold** keputusan. z positif berarti condong ke dropout.

### Langkah 3: log-odds, cara membaca koefisien
**Odds** = p / (1 − p). Contoh: p = 0,8 → odds = 4, artinya "4 banding 1". Logistic regression sebenarnya memodelkan **log-odds** secara linier:

$$\ln\frac{p}{1-p} = z$$

Maka: **menaikkan satu fitur sebanyak 1 standar deviasi mengubah log-odds sebesar koefisiennya**, atau **mengalikan odds dengan e^koefisien** (disebut *odds ratio*), dengan syarat fitur lain dibuat tetap.

### Koefisien nyata dari model proyek ini

Intercept (b₀) = −0,181. Beberapa koefisien (per 1 std):

| Fitur | Koefisien | Odds ratio (e^w) | Artinya |
|---|---|---|---|
| MK lulus semester 2 | **−1,428** | 0,24 | +1 std (≈ 3 MK lebih banyak lulus) → odds dropout tinggal **24%** |
| MK diambil semester 2 | **+0,807** | 2,24 | lihat penjelasan di bawah |
| MK lulus semester 1 | −0,679 | 0,51 | |
| Biaya kuliah lunas | −0,661 | 0,52 | per 1 std (0,32); dari 0→1 ≈ 3,1 std → odds × **0,13** (≈ 8× lebih kecil) |
| Rata-rata nilai semester 2 | −0,417 | 0,66 | |
| Usia saat mendaftar | +0,282 | 1,33 | +7,4 tahun → odds naik 33% |
| Beasiswa | −0,225 | 0,80 | |
| Debtor | +0,199 | 1,22 | |
| Gender (laki-laki) | +0,167 | 1,18 | |
| Prodi Social Service (9238) | −0,498 | 0,61 | dibanding rata-rata prodi |
| Prodi Basic Education (9853) | +0,571 | 1,77 | |

**Teka-teki: kenapa "MK diambil semester 2" koefisiennya positif?** Bukankah mengambil lebih banyak mata kuliah itu baik? Kuncinya ada di frasa *"fitur lain dibuat tetap"*. Jika jumlah MK **lulus** tetap, menambah MK **diambil** berarti menambah MK yang **gagal**. Jadi kedua koefisien bekerja **berpasangan** untuk menangkap **approval rate**, yaitu fitur paling kuat dari EDA (r = −0,66). Model menemukan konsep "rasio kelulusan" sendiri dari dua kolom mentah.

> ⚠️ Karena fitur-fitur akademik saling berkorelasi (*multikolinearitas*), koefisien satu per satu bisa terlihat aneh. Baca koefisien **berkelompok**, jangan terpisah.

### Contoh hitung prediksi

Mahasiswa pada baris ke-5 `sample_students.csv`: berusia 19 tahun, biaya kuliah lunas, tidak menunggak, tetapi **0 mata kuliah lulus** di semester 1 dan 2.

- Total skor: z = −0,181 + Σ(wᵢ·xᵢ) = **4,118**
- p = 1 / (1 + e^−4,118) = **0,984** → **98% risiko dropout**

Pelajaran: kondisi finansial yang baik **tidak bisa** mengimbangi kegagalan akademik total, karena koefisien akademik jauh lebih besar.

Coba sendiri di notebook:

```python
import numpy as np
row = sample.iloc[[4]]
pre, lr = best_model.named_steps["preprocessor"], best_model.named_steps["model"]
z = lr.intercept_[0] + np.asarray(pre.transform(row))[0] @ lr.coef_[0]
print(z, 1 / (1 + np.exp(-z)), best_model.predict_proba(row)[0, 1])   # 4.118  0.984  0.984
```

### Bagaimana model "belajar"?
Model mencari koefisien yang meminimalkan **log-loss** (cross-entropy):

$$\text{LogLoss} = -\frac{1}{n}\sum \big[y\ln p + (1-y)\ln(1-p)\big]$$

Artinya: model dihukum berat kalau **yakin tapi salah** (misalnya memberi p = 0,99 kepada mahasiswa yang ternyata lulus). Pencarian dilakukan secara numerik (solver `lbfgs`), sedikit demi sedikit menuruni "lembah" error.

### Regularisasi (`C`)
Tanpa rem, model bisa memberi koefisien sangat besar untuk pola kebetulan di data latih (*overfitting*). **Regularisasi L2** menambahkan hukuman untuk koefisien besar:

$$\text{Loss} = \text{LogLoss} + \frac{1}{C}\sum w_i^2$$

- `C` kecil = rem kuat, koefisien lebih kecil dan model lebih "hati-hati".
- `C` besar = rem lemah.
- GridSearch mencoba C ∈ {0,01; 0,1; 1; 10}, dan **C = 0,1** memberi CV F1 terbaik.

### `class_weight="balanced"`
Tanpa pembobotan, model cenderung mengutamakan kelas mayoritas (tidak dropout). Bobot `balanced` = n_total / (2 × n_kelas):

| Kelas | n (data latih) | Bobot |
|---|---|---|
| Dropout | 1.137 | 3.539 / (2 × 1.137) = **1,556** |
| Tidak dropout | 2.402 | 3.539 / (2 × 2.402) = **0,737** |

Kesalahan pada mahasiswa dropout "dihitung" **±2,1 kali lebih mahal**. Dampaknya recall naik, sesuai tujuan bisnis.

## 6.3 Decision Tree → Random Forest

**Decision tree** membuat serangkaian pertanyaan ya/tidak:

```
MK lulus smt 2 ≤ 2?
├── ya  → Biaya lunas?
│         ├── tidak → DROPOUT (95%)
│         └── ya    → ...
└── tidak → ...
```

Setiap potongan dipilih supaya kelompok hasilnya semakin "murni" (diukur dengan *Gini impurity*). Satu pohon mudah dipahami, tetapi **mudah overfit**: ia bisa terus bercabang sampai menghafal data.

**Random Forest** = **ratusan pohon** yang masing-masing:
1. dilatih dengan sampel acak dari data (*bootstrap / bagging*),
2. hanya melihat sebagian fitur acak di setiap potongan.

Prediksi akhirnya adalah **rata-rata suara** semua pohon. Kesalahan tiap pohon saling menetralkan, sehingga modelnya lebih stabil.

Hyperparameter yang dituning: `max_depth` ∈ {None, 10, 20} dan `min_samples_leaf` ∈ {1, 3, 5}. Hasil terbaik: **max_depth=None, min_samples_leaf=3** (daun minimal berisi 3 mahasiswa, sebagai rem terhadap overfitting).

## 6.4 Gradient Boosting

Kebalikan dari Random Forest: pohon-pohon kecil dibangun **berurutan**, dan setiap pohon baru fokus **memperbaiki kesalahan** pohon sebelumnya.

- `learning_rate` = seberapa besar kontribusi setiap pohon (kecil = lebih hati-hati, butuh lebih banyak pohon).
- Hasil terbaik: **learning_rate=0,05, max_depth=2, n_estimators=200**.
- `GradientBoostingClassifier` di scikit-learn **tidak punya** `class_weight`. Karena itu recall-nya paling rendah (0,73): model cenderung bermain aman ke kelas mayoritas.

## 6.5 Overfitting vs underfitting

| | Underfitting | Pas | Overfitting |
|---|---|---|---|
| Gejala | buruk di latih **dan** uji | bagus di keduanya | sangat bagus di latih, buruk di uji |
| Penyebab | model terlalu sederhana | – | model terlalu rumit / menghafal |
| Obat | fitur lebih informatif, model lebih kompleks | – | regularisasi, data lebih banyak, model lebih sederhana |

## 6.6 Cross-validation (5-fold stratified)

Satu kali split train/test bisa "beruntung" atau "sial". **K-fold cross-validation** membagi data latih menjadi 5 bagian, lalu:

```
Fold 1: [TEST ][train][train][train][train]
Fold 2: [train][TEST ][train][train][train]
Fold 3: [train][train][TEST ][train][train]
Fold 4: [train][train][train][TEST ][train]
Fold 5: [train][train][train][train][TEST ]
→ skor akhir = rata-rata 5 skor
```

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` menjaga proporsi dropout sama di setiap fold.

## 6.7 GridSearchCV: mencari hyperparameter terbaik

**Parameter** = dipelajari model dari data (koefisien). **Hyperparameter** = ditentukan manusia sebelum belajar (C, max_depth, …).

GridSearchCV mencoba **setiap kombinasi** dan menilainya dengan 5-fold CV:

| Model | Grid | Kombinasi | Jumlah pelatihan |
|---|---|---|---|
| Logistic Regression | C: 4 nilai | 4 | 4 × 5 = 20 |
| Random Forest | max_depth 3 × min_samples_leaf 3 | 9 | 45 |
| Gradient Boosting | n_estimators 2 × learning_rate 2 × max_depth 2 | 8 | 40 |

Setelah itu, kombinasi terbaik **dilatih ulang pada seluruh data latih** (`refit=True`, default).

**Hasil (CV F1 di data latih):**

| Model | CV F1 | Parameter terbaik |
|---|---|---|
| **Logistic Regression** | **0,7930** | C = 0,1 |
| Random Forest | 0,7915 | max_depth=None, min_samples_leaf=3 |
| Gradient Boosting | 0,7788 | learning_rate=0,05, max_depth=2, n_estimators=200 |

## 6.8 Kenapa model paling sederhana yang menang?

1. **Polanya memang sebagian besar linier**: semakin sedikit MK lulus, semakin berisiko, konsisten tanpa pola rumit.
2. **Datanya tidak besar** (±3.500 baris latih). Model kompleks lebih mudah overfit di data kecil.
3. **Bonus bisnis: mudah dijelaskan.** Koefisien bisa diterjemahkan menjadi "faktor yang menaikkan/menurunkan risiko", dan itulah yang ditampilkan di aplikasi ([Modul 8](08-aplikasi-streamlit.md)). Untuk keputusan yang menyangkut manusia (mahasiswa), kemampuan dijelaskan (*explainability*) sangat berharga.

---

## ✍️ Cek pemahaman

1. Jika z = 0, berapa probabilitas dropout? Jika z = −1,428?
2. Jelaskan dengan kalimat sendiri arti koefisien −1,428 pada "MK lulus semester 2".
3. Kenapa koefisien "MK diambil semester 2" positif?
4. Apa bedanya parameter dan hyperparameter? Beri contoh dari proyek ini.
5. Kenapa Gradient Boosting punya recall paling rendah?

<details>
<summary>Lihat jawaban</summary>

1. z = 0 → p = 0,5. z = −1,428 → p = 1 / (1 + e^1,428) = 1 / (1 + 4,17) ≈ **0,193**.
2. Dengan fitur lain tetap, setiap tambahan ±3 mata kuliah lulus di semester 2 (1 std) mengalikan odds dropout dengan 0,24, atau menurunkannya ±76%.
3. Dengan jumlah MK lulus tetap, menambah MK diambil berarti menambah MK yang gagal. Kedua koefisien bersama-sama menangkap rasio kelulusan.
4. Parameter dipelajari dari data (koefisien Logistic Regression, isi pohon). Hyperparameter ditentukan sebelum pelatihan (C = 0,1, max_depth, learning_rate).
5. Karena tidak memakai pembobotan kelas, model cenderung memprediksi kelas mayoritas (tidak dropout), sehingga lebih banyak mahasiswa dropout yang terlewat.
</details>

➡️ Lanjut ke [Modul 7 — Evaluasi model](07-evaluasi-model.md)
