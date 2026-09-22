# Modul 6 — Machine Learning: dari Konsep sampai Rumus

> **Tujuan modul:** memahami cara kerja ketiga algoritma yang dibandingkan, terutama **Logistic Regression** sampai ke rumus dan cara membaca koefisiennya, serta memahami cross-validation, tuning hyperparameter, dan **cara memilih model secara jujur**.

---

## 6.1 Konsep dasar

- **Machine learning** = membuat komputer menemukan pola dari data, bukan dari aturan yang ditulis manual.
- **Supervised learning** = belajar dari contoh yang **sudah punya jawaban** (label). Di sini: 2.904 mahasiswa latih yang status akhirnya sudah pasti (Dropout atau Graduate).
- **Klasifikasi** = memprediksi kategori (Dropout / Graduate). Kebalikannya adalah **regresi**, yang memprediksi angka.
- **Fitur (X)** = input (19 kolom). **Label/target (y)** = output (`is_dropout`: 1 = Dropout, 0 = Graduate).
- **Training** = mencari parameter model yang membuat prediksi paling cocok dengan label.
- **Inference/prediksi** = memakai model yang sudah dilatih pada data baru. Di proyek ini: 794 mahasiswa **Enrolled**.

## 6.2 Logistic Regression — model terpilih

### Langkah 1: skor linier
Setiap fitur (yang sudah distandardisasi) dikalikan dengan **koefisien** (bobot), lalu dijumlahkan bersama **intercept**:

$$z = b_0 + w_1x_1 + w_2x_2 + \dots + w_{46}x_{46}$$

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

z = 0 menghasilkan p = 0,5, yaitu **threshold** keputusan. z positif berarti condong ke Dropout, z negatif condong ke Graduate.

### Langkah 3: log-odds, cara membaca koefisien
**Odds** = p / (1 − p). Contoh: p = 0,8 → odds = 4, artinya "4 banding 1". Logistic regression sebenarnya memodelkan **log-odds** secara linier:

$$\ln\frac{p}{1-p} = z$$

Maka: **menaikkan satu fitur sebanyak 1 standar deviasi mengubah log-odds sebesar koefisiennya**, atau **mengalikan odds dengan e^koefisien** (disebut *odds ratio*), dengan syarat fitur lain dibuat tetap.

### Koefisien nyata dari model proyek ini

Intercept (b₀) = +0,302. Beberapa koefisien (per 1 std):

| Fitur | Koefisien | Odds ratio (e^w) | Artinya |
|---|---|---|---|
| MK lulus semester 2 | **−1,974** | 0,14 | +1 std (≈ 3 MK lebih banyak lulus) → odds dropout tinggal **14%** |
| MK lulus semester 1 | −1,222 | 0,30 | |
| MK diambil semester 2 | **+0,909** | 2,48 | lihat penjelasan di bawah |
| MK diambil semester 1 | +0,755 | 2,13 | |
| Biaya kuliah lunas | −0,694 | 0,50 | per 1 std (0,34); dari 0→1 ≈ 2,9 std → odds × **0,13** (≈ 7,7× lebih kecil) |
| Rata-rata nilai semester 2 | −0,604 | 0,55 | |
| Debtor | +0,345 | 1,41 | |
| Beasiswa | −0,306 | 0,74 | |
| Gender (laki-laki) | +0,196 | 1,22 | |
| Usia saat mendaftar | +0,160 | 1,17 | +7,9 tahun → odds naik 17% |
| Prodi Basic Education (9853) | +0,743 | 2,10 | dibanding rata-rata prodi |
| Prodi Informatics Engineering (9119) | +0,518 | 1,68 | |
| Prodi Social Service (9238) | −0,579 | 0,56 | |

**Teka-teki: kenapa "MK diambil" koefisiennya positif?** Bukankah mengambil lebih banyak mata kuliah itu baik? Kuncinya ada di frasa *"fitur lain dibuat tetap"*. Jika jumlah MK **lulus** tetap, menambah MK **diambil** berarti menambah MK yang **gagal**. Jadi pasangan koefisien "lulus" (negatif) dan "diambil" (positif) bekerja **berpasangan** untuk menangkap **approval rate**, yaitu fitur paling kuat dari EDA (r = −0,66). Model menemukan konsep "rasio kelulusan" sendiri dari kolom-kolom mentah.

> ⚠️ Karena fitur-fitur akademik saling berkorelasi (*multikolinearitas*), koefisien satu per satu bisa terlihat aneh. Baca koefisien **berkelompok**, jangan terpisah.

### Contoh hitung prediksi

Mahasiswa Enrolled pertama di `data_enrolled.csv`: berusia 18 tahun, prodi Social Service, biaya kuliah lunas, tidak menunggak. Tetapi di semester 1 ia hanya lulus **1 dari 6** mata kuliah, dan di semester 2 **2 dari 6**.

- Total skor: z = 0,302 + Σ(wᵢ·xᵢ) = **2,587**
- p = 1 / (1 + e^−2,587) = **0,930** → **93% risiko dropout**

Pelajaran: kondisi finansial yang baik dan prodi berisiko rendah (Social Service) **tidak cukup** mengimbangi kegagalan akademik, karena koefisien akademik jauh lebih besar.

Coba sendiri di notebook:

```python
import numpy as np
row = df_enrolled[SELECTED_FEATURES].iloc[[0]]
pre, lr = best_model.named_steps["preprocessor"], best_model.named_steps["model"]
z = lr.intercept_[0] + np.asarray(pre.transform(row))[0] @ lr.coef_[0]
print(z, 1 / (1 + np.exp(-z)), best_model.predict_proba(row)[0, 1])   # 2.587  0.930  0.930
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
Data latih berisi 39% dropout dan 61% graduate. Tanpa pembobotan, model cenderung mengutamakan kelas mayoritas. Bobot `balanced` = n_total / (2 × n_kelas):

| Kelas | n (data latih) | Bobot |
|---|---|---|
| Dropout | 1.137 | 2.904 / (2 × 1.137) = **1,277** |
| Graduate | 1.767 | 2.904 / (2 × 1.767) = **0,822** |

Kesalahan pada mahasiswa dropout "dihitung" **±1,55 kali lebih mahal**. Dampaknya recall naik, sesuai tujuan bisnis.

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
- Hasil terbaik: **learning_rate=0,05, max_depth=3, n_estimators=200**.
- `GradientBoostingClassifier` di scikit-learn **tidak punya** `class_weight`. Karena itu recall-nya paling rendah (CV recall 0,832): model cenderung bermain aman ke kelas mayoritas.

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
→ skor akhir = rata-rata 5 skor (± standar deviasinya)
```

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` menjaga proporsi dropout sama di setiap fold. **Standar deviasi** antar-fold menunjukkan seberapa "goyang" skornya. Angka ini penting untuk memilih model (bagian 6.8).

## 6.7 GridSearchCV: mencari hyperparameter terbaik

**Parameter** = dipelajari model dari data (koefisien). **Hyperparameter** = ditentukan manusia sebelum belajar (C, max_depth, …).

GridSearchCV mencoba **setiap kombinasi** dan menilainya dengan 5-fold CV:

| Model | Grid | Kombinasi | Jumlah pelatihan |
|---|---|---|---|
| Logistic Regression | C: 4 nilai | 4 | 4 × 5 = 20 |
| Random Forest | max_depth 3 × min_samples_leaf 3 | 9 | 45 |
| Gradient Boosting | n_estimators 2 × learning_rate 2 × max_depth 2 | 8 | 40 |

```python
GridSearchCV(pipe, grid, scoring={"f1": "f1", "recall": "recall"}, refit="f1", cv=cv)
```
Dua metrik dicatat sekaligus. `refit="f1"` berarti kombinasi terbaik dipilih berdasarkan F1, lalu **dilatih ulang pada seluruh data latih**.

**Hasil (cross-validation di data latih):**

| Model | CV F1 | ± std | CV Recall | Parameter terbaik |
|---|---|---|---|---|
| **Logistic Regression** | 0,8723 | **±0,0111** | **0,8540** | C = 0,1 |
| Random Forest | 0,8685 | ±0,0169 | 0,8364 | max_depth=None, min_samples_leaf=3 |
| Gradient Boosting | **0,8733** | ±0,0206 | 0,8320 | learning_rate=0,05, max_depth=3, n_estimators=200 |

## 6.8 ⭐ Memilih model secara jujur: aturan "1 standar deviasi"

Sekilas Gradient Boosting "menang" (CV F1 0,8733). Tapi selisihnya dengan Logistic Regression hanya **0,001**, jauh lebih kecil dari standar deviasi antar-fold (±0,011 sampai ±0,021). Artinya perbedaan itu **tidak bermakna**. Jika pembagian fold diacak ulang, urutannya bisa berubah.

Karena itu notebook memakai aturan yang transparan:
1. Cari CV F1 tertinggi beserta standar deviasinya.
2. Model yang CV F1-nya dalam rentang **1 std** dari yang tertinggi dianggap **setara secara statistik**.
3. Di antara yang setara, pilih yang **paling sederhana dan mudah dijelaskan** (LR → RF → GB).

```python
SIMPLICITY_ORDER = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
top_name = results["CV F1"].idxmax()                                   # Gradient Boosting
tolerance = results.loc[top_name, "CV F1 std"]                        # 0,0206
equivalent = [m for m in SIMPLICITY_ORDER
              if results.loc[m, "CV F1"] >= results.loc[top_name, "CV F1"] - tolerance]
best_name = equivalent[0]                                              # Logistic Regression
```

Aturan ini **tidak melihat data uji sama sekali**, jadi data uji tetap menjadi penilaian akhir yang jujur. Logistic Regression juga punya **CV recall tertinggi** dan **variasi terkecil**, dua hal yang mendukung pilihannya secara bisnis.

> 💡 Ini versi sederhana dari **"one-standard-error rule"** yang dikenal di statistika: jika beberapa model sama baiknya dalam batas ketidakpastian, pilih yang paling sederhana.

## 6.9 Kenapa model paling sederhana layak dipilih?

1. **Polanya memang sebagian besar linier**: semakin sedikit MK lulus, semakin berisiko, konsisten tanpa pola rumit. Ketiga model mencapai ROC-AUC ±0,97.
2. **Datanya tidak besar** (±2.900 baris latih). Model kompleks lebih mudah overfit di data kecil.
3. **Bonus bisnis: mudah dijelaskan.** Koefisien bisa diterjemahkan menjadi "faktor yang menaikkan/menurunkan risiko", dan itulah yang ditampilkan di aplikasi ([Modul 8](08-aplikasi-streamlit.md)). Untuk keputusan yang menyangkut manusia (mahasiswa), kemampuan dijelaskan (*explainability*) sangat berharga.

---

## ✍️ Cek pemahaman

1. Jika z = 0, berapa probabilitas dropout? Jika z = −1,974?
2. Jelaskan dengan kalimat sendiri arti koefisien −1,974 pada "MK lulus semester 2".
3. Kenapa koefisien "MK diambil semester 2" positif?
4. Apa bedanya parameter dan hyperparameter? Beri contoh dari proyek ini.
5. Gradient Boosting punya CV F1 tertinggi. Kenapa yang dipilih Logistic Regression?
6. Kenapa Gradient Boosting punya recall paling rendah?

<details>
<summary>Lihat jawaban</summary>

1. z = 0 → p = 0,5. z = −1,974 → p = 1 / (1 + e^1,974) = 1 / (1 + 7,20) ≈ **0,122**.
2. Dengan fitur lain tetap, setiap tambahan ±3 mata kuliah lulus di semester 2 (1 std) mengalikan odds dropout dengan 0,14, atau menurunkannya ±86%.
3. Dengan jumlah MK lulus tetap, menambah MK diambil berarti menambah MK yang gagal. Kedua koefisien bersama-sama menangkap rasio kelulusan.
4. Parameter dipelajari dari data (koefisien Logistic Regression, isi pohon). Hyperparameter ditentukan sebelum pelatihan (C = 0,1, max_depth, learning_rate).
5. Selisihnya hanya 0,001, jauh lebih kecil dari standar deviasi antar-fold (±0,02), jadi ketiga model setara secara statistik. Di antara yang setara dipilih yang paling sederhana dan mudah dijelaskan, yang juga punya CV recall tertinggi.
6. Karena tidak memakai pembobotan kelas, model cenderung memprediksi kelas mayoritas (Graduate), sehingga lebih banyak mahasiswa dropout yang terlewat.
</details>

➡️ Lanjut ke [Modul 7 — Evaluasi model](07-evaluasi-model.md)
