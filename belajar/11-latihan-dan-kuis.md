# Modul 11 — Latihan, Kuis & Simulasi Review

> **Tujuan modul:** mengubah pemahaman menjadi keterampilan. Kerjakan latihan di notebook baru (salin `notebook.ipynb`, jangan ubah aslinya), jawab kuis tanpa melihat jawaban, lalu latih diri menjawab pertanyaan "sidang".

Setiap latihan punya **petunjuk** dan, bila memungkinkan, **angka acuan** agar Anda bisa memeriksa hasil sendiri.

> Catatan: latihan 1–4 memakai **seluruh data** (EDA). Latihan 5 ke atas memakai **data pemodelan** `df_model` (Dropout vs Graduate), seperti di notebook.

---

## 🧪 Bagian A — 13 Latihan praktik

### Level 1: pandas & EDA

**Latihan 1 — Status pernikahan.** Hitung tingkat dropout dan jumlah mahasiswa per `Marital_status`. Kelompok mana yang terlalu kecil untuk dipercaya?

<details><summary>Petunjuk & acuan</summary>

`df.groupby("Marital_status")["is_dropout"].agg(["mean", "size"])`. Acuan: single 30,2% (n=3.919), married 47,2% (n=379), divorced 46,2% (n=91). Kode 3 (n=4) dan 6 (n=6) terlalu kecil. Hitung rentang kepercayaannya ([Modul 4](04-statistik-dan-eda.md#43-hati-hati-dengan-kelompok-kecil-ketidakpastian)).
</details>

**Latihan 2 — Approval rate semester 1.** Buat kolom approval rate semester 1 dan hitung korelasinya dengan `is_dropout`. Bandingkan dengan semester 2 (−0,66).

<details><summary>Acuan</summary>

Korelasi ≈ **−0,59**. Semester 2 lebih kuat karena lebih dekat dengan waktu terjadinya dropout.
</details>

**Latihan 3 — Bersihkan anomali prodi 171.** Keluarkan mahasiswa dengan `Curricular_units_2nd_sem_enrolled == 0`, lalu hitung tingkat dropout mahasiswa yang **mengambil** MK tetapi tidak lulus satu pun di semester 2.

<details><summary>Acuan</summary>

690 mahasiswa, tingkat dropout **94,2%** (vs 84% jika anomali ikut dihitung).
</details>

**Latihan 4 — Confounding.** Hitung tingkat dropout mahasiswa **perantau vs bukan perantau** di dalam dua kelompok usia (≤ 20 dan > 20). Apakah "perantau lebih jarang dropout" masih berlaku?

<details><summary>Acuan</summary>

≤ 20: bukan perantau 20,0%, perantau 21,8%. > 20: bukan perantau 49,4%, perantau 42,6%. Efek keseluruhan (38% vs 28%) sebagian besar dijelaskan oleh **usia**, karena perantau rata-rata lebih muda (20,8 vs 26,3 tahun).
</details>

### Level 2: modeling & evaluasi

**Latihan 5 — Model tanpa Gender.** Latih ulang Logistic Regression (C = 0,1, class_weight balanced) tanpa fitur `Gender`. Bandingkan F1, recall, dan AUC.

<details><summary>Acuan</summary>

F1 0,902, recall 0,919, AUC 0,971 (model utama: 0,907 / 0,923 / 0,973).
</details>

**Latihan 6 — Model deteksi dini (semester 1 saja).** Buang semua fitur `..._2nd_sem_...` lalu latih ulang. (Versi ini sudah ada di notebook, *Evaluation → 4*. Coba tulis ulang kodenya sendiri tanpa melihat.)

<details><summary>Acuan</summary>

15 fitur → F1 **0,857**, recall 0,884, precision 0,831, AUC **0,946**.
</details>

**Latihan 7 — Tabel threshold.** Untuk threshold 0,3 sampai 0,7, hitung precision, recall, F1, dan jumlah mahasiswa yang diberi peringatan.

<details><summary>Petunjuk & acuan</summary>

```python
proba = best_model.predict_proba(X_test)[:, 1]
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba >= t).astype(int)
    print(t, precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred), pred.sum())
```
Bandingkan dengan tabel di [Modul 7](07-evaluasi-model.md#75-threshold-tarik-ulur-precision-vs-recall). Pertanyaan lanjutan: kenapa F1 tertinggi ada di 0,7 tetapi proyek tetap memakai 0,5?
</details>

**Latihan 8 — Fairness per kelompok.** Hitung recall model utama untuk perempuan vs laki-laki, dan untuk usia ≤ 24 vs > 24. Apa artinya bagi institusi?

<details><summary>Acuan & diskusi</summary>

| Kelompok | Recall |
|---|---|
| Perempuan (130 dropout di data uji) | 0,915 |
| Laki-laki (154 dropout) | 0,929 |
| Usia ≤ 24 | **0,893** |
| Usia > 24 | 0,960 |

Perbedaan gender kecil. Yang lebih perlu diperhatikan: model **lebih sering melewatkan mahasiswa muda** yang akan dropout, karena pola dropout mereka "kurang khas" (tidak didukung sinyal usia). Tindak lanjut: pertimbangkan threshold lebih rendah atau pemantauan tambahan untuk mahasiswa muda, lalu pantau metrik per kelompok secara rutin.
</details>

**Latihan 9 — Eksperimen hyperparameter.** Tambahkan regularisasi L1 ke grid Logistic Regression:
```python
{"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l1", "l2"], "model__solver": ["liblinear"]}
```
Fitur mana yang koefisiennya menjadi **nol**? Apa artinya?

<details><summary>Petunjuk</summary>

L1 (*Lasso*) bisa membuat koefisien tepat 0, yang berarti fitur itu "dibuang" otomatis. Ini bentuk seleksi fitur bawaan model. Periksa dengan `best.named_steps["model"].coef_`.
</details>

**Latihan 10 — Bandingkan perumusan target lama vs baru.** Latih model versi lama (Enrolled diberi label 0, semua 4.424 data) dan versi baru (Dropout vs Graduate). Bandingkan F1 dan ROC-AUC masing-masing pada data ujinya sendiri. Lalu hitung: dari mahasiswa Enrolled di data uji versi lama, berapa persen yang diberi probabilitas ≥ 0,5?

<details><summary>Acuan & diskusi</summary>

Versi lama: F1 0,818, ROC-AUC 0,930. Versi baru: F1 0,907, ROC-AUC 0,973. Di data uji versi lama ada 162 mahasiswa Enrolled, dan **25,9%** di antaranya diberi probabilitas ≥ 0,5 meski berlabel 0. Lebih mencolok lagi: **42 dari 57 false positive** versi lama adalah mahasiswa Enrolled. Sebagian besar "kesalahan" model lama sebenarnya berasal dari label yang ambigu, karena model diajari bahwa mahasiswa yang mirip dropout adalah "tidak dropout".
</details>

### Level 3: aplikasi & dashboard

**Latihan 11 — Threshold yang bisa diatur di aplikasi.** Tambahkan slider di sidebar `app.py` supaya staf bisa mengatur batas level "Tinggi".

<details><summary>Petunjuk</summary>

```python
with st.sidebar:
    high = st.slider("Batas risiko tinggi", 0.3, 0.9, 0.5, 0.05)
```
Lalu gunakan `high` di fungsi `risk_level` dan di perhitungan `verdict`. Jalankan dengan `python -m streamlit run app.py`. Coba dengan data Enrolled: berapa mahasiswa berisiko tinggi jika batasnya 0,9? (Acuan: 119.)
</details>

**Latihan 12 — Question baru di Metabase.** Buat question SQL: tingkat dropout per prodi **per gender** (hanya kombinasi dengan n ≥ 30), lalu tampilkan sebagai grafik batang berkelompok.

<details><summary>Petunjuk</summary>

```sql
SELECT course, gender, AVG(is_dropout)::float AS dropout_rate, COUNT(*) AS n
FROM students
WHERE status IN ('Dropout', 'Graduate')     -- samakan populasinya dengan data model
GROUP BY course, gender
HAVING COUNT(*) >= 30
ORDER BY course, gender;
```
</details>

**Latihan 13 — Proyek mini.** Buat **model bertahap**: model semester 1 (latihan 6) + model semester 2 (model utama). Di aplikasi, jika data semester 2 belum diisi, gunakan model semester 1. Tulis 1 paragraf di README tentang manfaat bisnisnya.

---

## 📝 Bagian B — Kuis 32 soal

Jawab dulu, baru buka kunci di akhir bagian ini.

**Bisnis & data**
1. Berapa tingkat dropout Jaya Jaya Institut, dan berapa jumlah mahasiswanya?
2. Apa arti status `Enrolled`?
3. Kenapa `Course` tidak boleh dianalisis dengan korelasi Pearson?
4. Kenapa kolom `GDP` hanya punya 10 nilai unik?
5. Apa anomali data prodi 171?

**Statistik & EDA**
6. Mean vs median nilai semester 2 mahasiswa dropout: berapa nilainya dan apa artinya?
7. Kenapa grafik memakai tingkat (rate), bukan jumlah?
8. Kenapa angka 67% untuk prodi Biofuel tidak bisa dipercaya?
9. Apa itu confounder? Berikan contoh dari proyek ini.
10. Fitur apa yang paling kuat berkorelasi dengan dropout?

**Persiapan data**
11. Sebutkan 3 opsi perumusan target. Opsi mana yang dipakai sekarang, dan kenapa versi pertama ditolak reviewer?
12. Apa beda data uji dan data prediksi di proyek ini? Berapa jumlah masing-masing?
13. Berapa fitur yang dipakai, dan bagaimana membuktikan pengurangannya tidak merugikan?
14. Apa fungsi `stratify=y`?
15. Apa itu data leakage? Beri 2 contoh.
16. Apa fungsi `min_frequency=20` pada OneHotEncoder?
17. Kenapa hasil preprocessing berjumlah 46 kolom?

**Machine learning**
18. Tuliskan fungsi sigmoid.
19. Arti koefisien −1,974 pada MK lulus semester 2?
20. Kenapa koefisien MK diambil semester 2 positif?
21. Apa fungsi `C` dan berapa nilai terbaiknya?
22. Berapa bobot kelas dropout pada `class_weight="balanced"`?
23. Apa beda Random Forest dan Gradient Boosting?
24. Berapa kali model dilatih dalam GridSearch Logistic Regression?
25. Gradient Boosting punya CV F1 tertinggi. Kenapa yang dipilih Logistic Regression?

**Evaluasi**
26. Hitung precision dan recall dari confusion matrix proyek.
27. Kenapa akurasi bukan metrik utama?
28. Apa arti AUC 0,973?
29. Apa yang terjadi jika threshold diturunkan ke 0,3?
30. Berapa mahasiswa Enrolled yang berisiko tinggi, dan bagaimana membaca angka itu dengan benar?

**Deployment & dashboard**
31. Kenapa `scikit-learn` harus di-pin ke versi 1.7.2 di `requirements.txt`?
32. Kenapa reviewer yang hanya memuat `metabase.db.mv.db` melihat grafik error?

<details>
<summary><b>🔑 Kunci jawaban kuis</b></summary>

1. 32,1% (1.421 dari 4.424).
2. Masih terdaftar di akhir masa studi normal (terlambat lulus); status akhirnya belum diketahui.
3. Karena kodenya label nominal, bukan angka yang punya urutan dan jarak.
4. Nilainya sama untuk semua mahasiswa yang masuk di tahun yang sama; mencerminkan tahun masuk.
5. Semua 180 mahasiswa dengan 0 MK diambil berasal dari prodi 171; datanya kemungkinan tidak tercatat.
6. Mean 5,9, median 0 → mayoritas mahasiswa dropout tidak lulus satu MK pun di semester 2.
7. Ukuran kelompok berbeda; pertanyaan bisnisnya soal risiko, bukan volume.
8. Hanya 12 mahasiswa; rentang kepercayaan 95% ±27%.
9. Variabel ketiga yang memengaruhi dua variabel lain. Contoh: usia memengaruhi pilihan kelas malam **dan** risiko dropout.
10. Approval rate semester 2 (r = −0,66).
11. Multikelas; Dropout vs Graduate dengan Enrolled diprediksi (**dipakai sekarang**); Dropout vs "Tidak Dropout" (Graduate + Enrolled). Versi pertama memakai opsi ketiga dan ditolak karena label Enrolled belum final, sehingga target ambigu.
12. Data uji (726 mahasiswa Dropout/Graduate) punya jawaban dan dipakai untuk mengukur performa. Data prediksi (794 Enrolled) belum punya jawaban dan dipakai untuk menghasilkan peringatan.
13. 19 fitur; eksperimen CV: F1 0,880 vs 0,878 untuk 36 fitur.
14. Menjaga proporsi dropout di train dan test tetap sama (39,15% vs 39,12%).
15. Informasi dari luar data latih ikut masuk ke pelatihan. Contoh: scaler di-fit pada seluruh data; memakai fitur yang baru tersedia setelah kejadian.
16. Kategori dengan < 20 sampel digabung menjadi `infrequent` untuk mencegah overfitting dan menangani kategori baru.
17. 29 kolom one-hot (dari 2 fitur kategori) + 17 kolom numerik.
18. σ(z) = 1 / (1 + e^−z).
19. Dengan fitur lain tetap, +1 std (±3 MK lulus) mengalikan odds dropout dengan 0,14.
20. Dengan MK lulus tetap, menambah MK diambil berarti menambah MK gagal; keduanya menangkap approval rate.
21. Kebalikan kekuatan regularisasi; terbaik C = 0,1.
22. 2.904 / (2 × 1.137) = 1,277.
23. RF: banyak pohon paralel dari sampel acak, lalu dirata-rata (bagging). GB: pohon kecil berurutan, masing-masing memperbaiki kesalahan sebelumnya (boosting).
24. 4 nilai C × 5 fold = 20 kali, ditambah 1 kali refit.
25. Selisihnya 0,001, jauh lebih kecil dari standar deviasi antar-fold (±0,02), jadi setara secara statistik. Aturan "1 standar deviasi" memilih yang paling sederhana. LR juga punya CV recall tertinggi dan bisa dijelaskan.
26. Precision = 262/294 = 0,891; recall = 262/284 = 0,923.
27. Kelas tidak seimbang; model yang selalu menebak "Graduate" sudah mendapat akurasi 60,9% dengan recall 0%.
28. 97% peluang model memberi skor lebih tinggi kepada mahasiswa dropout dibanding mahasiswa lulus yang dipilih acak.
29. Recall naik ke 0,958, precision turun ke 0,747, dan mahasiswa yang diberi peringatan bertambah dari 294 menjadi 364.
30. 438 dari 794 (55%). Angka ini adalah peringkat prioritas, bukan kepastian: Enrolled secara akademik berada di antara Dropout dan Graduate (pergeseran populasi), dan sebagian data (prodi 171) tidak lengkap.
31. `model.joblib` adalah objek pickle yang terikat pada versi scikit-learn pembuatnya.
32. File itu hanya berisi definisi dashboard dan koneksi, bukan data. Tabel `students` di PostgreSQL tidak ada di laptop reviewer.
</details>

---

## 🎤 Bagian C — Simulasi "sidang" / review

Latih menjawab dengan suara keras, maksimal 1 menit per pertanyaan.

<details><summary><b>1. "Jelaskan proyek Anda dalam 1 menit."</b></summary>

"Jaya Jaya Institut kehilangan 32% mahasiswanya karena dropout. Saya menganalisis data 4.424 mahasiswa dan menemukan tiga kelompok faktor: performa akademik semester awal, kondisi finansial, dan profil pendaftaran seperti usia. Saya melatih model Logistic Regression hanya dengan mahasiswa yang status akhirnya sudah diketahui, yaitu dropout dan lulus. Model ini mendeteksi 92% mahasiswa yang akan dropout. Model lalu saya pakai untuk memprediksi 794 mahasiswa yang masih aktif, dan 438 di antaranya berisiko tinggi. Semua ini bisa diakses lewat aplikasi Streamlit dan dipantau di dashboard Metabase, disertai rekomendasi seperti pendampingan akademik dan intervensi finansial."
</details>

<details><summary><b>2. "Kenapa mahasiswa Enrolled tidak dipakai untuk melatih model?"</b></summary>

"Karena status akhirnya belum diketahui. Kalau mereka diberi label 'tidak dropout', sebagian labelnya bisa salah, karena ada yang kelak dropout. Targetnya jadi ambigu. Jadi saya hanya melatih model dengan Dropout dan Graduate, lalu memakai model itu untuk memprediksi mahasiswa Enrolled. Itu juga persis cara sistem ini dipakai di dunia nyata. Setelah perbaikan ini, F1 naik dari 0,82 ke 0,91."
</details>

<details><summary><b>3. "Kenapa memilih Logistic Regression, padahal Gradient Boosting skornya lebih tinggi?"</b></summary>

"CV F1 Gradient Boosting hanya lebih tinggi 0,001, sedangkan standar deviasi antar-fold ±0,02, jadi secara statistik ketiga model setara. Saya memakai aturan '1 standar deviasi': di antara model yang setara, pilih yang paling sederhana. Logistic Regression juga punya CV recall tertinggi, dan koefisiennya bisa menjelaskan faktor risiko setiap mahasiswa. Semua keputusan ini diambil dari cross-validation, tanpa melihat data uji."
</details>

<details><summary><b>4. "Akurasi Anda 92,6%, apakah itu bagus?"</b></summary>

"Akurasi bukan metrik utama karena kelasnya tidak seimbang. Model yang selalu menebak 'lulus' sudah mendapat 61%. Saya fokus pada recall (92%) karena melewatkan mahasiswa berisiko lebih mahal daripada memberi peringatan yang keliru, dan pada F1 (0,91) untuk menjaga keseimbangannya dengan precision (89%)."
</details>

<details><summary><b>5. "Bagaimana Anda mencegah overfitting dan data leakage?"</b></summary>

"Preprocessing dibungkus dalam Pipeline sehingga encoder dan scaler hanya di-fit di data latih, termasuk di setiap fold cross-validation. Hyperparameter dan model dipilih dengan 5-fold stratified CV, dan data uji hanya dipakai sekali di akhir. Mahasiswa Enrolled dipisahkan sejak awal dan tidak pernah menyentuh proses training. Selain itu ada regularisasi (C = 0,1) dan `min_frequency` pada one-hot encoder untuk kategori langka."
</details>

<details><summary><b>6. "Apa keterbatasan terbesar proyek ini?"</b></summary>

"Pertama, model memakai data semester 2, jadi baru bisa dipakai setelah semester 2. Model semester 1 saja masih mencapai F1 0,86 untuk deteksi lebih dini. Kedua, mahasiswa Enrolled berbeda karakternya dari data latih, jadi probabilitasnya lebih tepat dipakai untuk mengurutkan prioritas. Ketiga, temuannya bersifat korelasi, bukan sebab-akibat. Efek kelas malam misalnya ternyata dijelaskan oleh usia. Keempat, ada anomali data di prodi 171. Terakhir, recall untuk mahasiswa muda lebih rendah (0,89), jadi perlu dipantau."
</details>

<details><summary><b>7. "438 dari 794 mahasiswa aktif berisiko tinggi. Apakah itu masuk akal?"</b></summary>

"Masuk akal, karena mahasiswa Enrolled adalah mahasiswa yang belum lulus di akhir masa studi normal. Mereka memang tertinggal: rata-rata lulus 4,1 mata kuliah di semester 2, dibanding 6,2 pada lulusan. Tapi angka ini harus dibaca sebagai peringkat prioritas, bukan kepastian. Saya sarankan mulai dari 119 mahasiswa dengan probabilitas di atas 90%, dan memvalidasi prediksi ini setelah status akhir mereka diketahui."
</details>

<details><summary><b>8. "Bagaimana institusi memakai hasil ini sehari-hari?"</b></summary>

"Setiap akhir semester, staf membuka tab Prediksi Batch dan memprediksi seluruh mahasiswa Enrolled. Aplikasi mengurutkan mahasiswa dari risiko tertinggi. Mahasiswa berisiko tinggi dihubungi dosen wali dalam 1–2 minggu, lengkap dengan alasan dan rekomendasi dari aplikasi. Pimpinan memantau tren di dashboard Metabase setiap bulan."
</details>

<details><summary><b>9. "Apakah model ini adil?"</b></summary>

"Saya mengevaluasinya. Menghapus fitur gender hanya menurunkan F1 dari 0,907 ke 0,902, jadi fitur itu bisa dihapus jika institusi mau. Recall untuk perempuan (0,92) dan laki-laki (0,93) hampir sama, tapi recall untuk mahasiswa muda lebih rendah (0,89), jadi kelompok itu perlu dipantau. Yang paling penting, model hanya dipakai untuk menawarkan bantuan, bukan menghukum, dan keputusan akhir tetap di tangan dosen wali."
</details>

<details><summary><b>10. "Kalau datanya bertambah tahun depan, apa yang Anda lakukan?"</b></summary>

"Pertama, validasi prediksi Enrolled tahun ini dengan status akhir mereka. Lalu evaluasi ulang model dengan data terbaru (cek apakah ada data drift). Jika performanya turun, latih ulang dengan notebook yang sama karena seluruh prosesnya sudah reproducible. Mahasiswa yang sudah punya status akhir masuk ke data latih, dan yang masih Enrolled menjadi data prediksi. Setelah itu perbarui `model.joblib`, `model_metadata.json`, dan `data_enrolled.csv`, lalu push ke GitHub supaya Streamlit Cloud otomatis memakai versi baru."
</details>

<details><summary><b>11. "Kenapa dashboard memakai SQL native, bukan query builder?"</b></summary>

"SQL memberi kendali penuh: menghitung tingkat dropout dengan `AVG(is_dropout)`, menyaring kelompok kecil dengan `HAVING`, mengurutkan kategori secara kustom, dan meng-unpivot data semester dengan `UNION ALL`. Dengan field filter `{{course}}` dan lainnya, keempat filter tetap berlaku di semua grafik."
</details>

---

## 🏁 Penutup

Jika Anda sudah:
- menyelesaikan 12 modul (0–11),
- mengerjakan minimal latihan 1–8 dan latihan 10,
- mendapat ≥ 27/32 di kuis, dan
- bisa menjawab 11 pertanyaan sidang dengan lancar,

maka Anda **menguasai proyek ini secara menyeluruh**, dari data mentah sampai keputusan bisnis. Kembali ke [checklist penguasaan](README.md#-checklist-penguasaan) untuk konfirmasi terakhir. Selamat! 🎓
