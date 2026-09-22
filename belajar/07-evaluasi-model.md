# Modul 7 — Evaluasi Model

> **Tujuan modul:** bisa menghitung dan menjelaskan setiap metrik **dengan tangan**, memahami kenapa recall dan F1 dipakai sebagai metrik utama, memahami cara kerja threshold serta feature importance, dan memahami cara membaca **hasil prediksi mahasiswa Enrolled**.

---

## 7.1 Confusion matrix model terpilih (data uji, 726 mahasiswa)

|  | **Prediksi: Graduate** | **Prediksi: Dropout** | Total |
|---|---|---|---|
| **Aktual: Graduate** | **TN = 410** ✅ | **FP = 32** ⚠️ | 442 |
| **Aktual: Dropout** | **FN = 22** ❌ | **TP = 262** ✅ | 284 |
| Total | 432 | 294 | 726 |

| Sel | Nama | Arti bisnis |
|---|---|---|
| TP | True Positive | mahasiswa berisiko **terdeteksi**, lalu dibimbing ✅ |
| TN | True Negative | mahasiswa yang akan lulus diprediksi lulus ✅ |
| FP | False Positive | mahasiswa yang akan lulus **diberi peringatan** → biaya kecil (sesi bimbingan tambahan) |
| FN | False Negative | mahasiswa berisiko **terlewat** → biaya besar (mahasiswa dropout tanpa pernah dibantu) |

**Asimetri biaya** inilah alasan utama pemilihan metrik: FN jauh lebih mahal daripada FP.

## 7.2 Menghitung metrik dengan tangan

| Metrik | Rumus | Hitungan | Hasil | Pertanyaan yang dijawab |
|---|---|---|---|---|
| **Accuracy** | (TP+TN) / total | (262+410) / 726 | **0,926** | Berapa persen tebakan benar? |
| **Precision** | TP / (TP+FP) | 262 / 294 | **0,891** | Dari yang diberi peringatan, berapa yang benar-benar dropout? |
| **Recall** (sensitivity) | TP / (TP+FN) | 262 / 284 | **0,923** | Dari semua yang akan dropout, berapa yang tertangkap? |
| **F1** | 2·P·R / (P+R) | 2·0,891·0,923 / 1,814 | **0,907** | Keseimbangan precision & recall |
| Specificity | TN / (TN+FP) | 410 / 442 | 0,928 | Dari yang akan lulus, berapa yang diprediksi lulus? |

> 🧮 **Kenapa F1 memakai rata-rata harmonik, bukan rata-rata biasa?** Rata-rata harmonik "menghukum" ketimpangan. Model dengan P = 1,0 dan R = 0,1 punya rata-rata biasa 0,55, tetapi F1-nya hanya 0,18. F1 hanya bisa tinggi jika **keduanya** tinggi.

## 7.3 Kenapa akurasi 92,6% bukan angka terpenting

Data pemodelan berisi 61% Graduate. Model "bodoh" yang **selalu** menebak "Graduate" sudah mendapat **akurasi 60,9%**, padahal recall-nya **0%** karena tidak ada satu pun mahasiswa berisiko yang terdeteksi. Pada data yang tidak seimbang, akurasi bisa tinggi padahal model tidak berguna.

## 7.4 ROC curve dan AUC

Model sebenarnya menghasilkan **probabilitas**, bukan 0/1. Keputusan 0/1 bergantung pada **threshold** (default 0,5).

**ROC curve** menggambar, untuk *semua* kemungkinan threshold:
- sumbu Y: **True Positive Rate** (= recall)
- sumbu X: **False Positive Rate** (= FP / (FP+TN) = 1 − specificity)

**AUC** (Area Under Curve) = luas di bawah kurva:
- 0,5 = sama dengan tebakan acak (garis diagonal)
- 1,0 = sempurna
- **0,973** = model proyek ini

**Arti AUC yang intuitif:** jika Anda mengambil **1 mahasiswa dropout dan 1 mahasiswa lulus secara acak**, ada **97% peluang** model memberi probabilitas lebih tinggi kepada yang dropout. AUC mengukur kemampuan model **mengurutkan** risiko, dan itu sangat relevan untuk fitur "prediksi batch" yang mengurutkan mahasiswa Enrolled dari risiko tertinggi.

Ketiga model punya AUC hampir sama (0,969–0,975). Artinya kemampuan mengurutkannya setara, dan perbedaannya lebih banyak di **posisi threshold** efektif (dipengaruhi `class_weight`).

## 7.5 Threshold: tarik-ulur precision vs recall

Hasil model terpilih pada data uji untuk berbagai threshold:

| Threshold | Precision | Recall | F1 | Mahasiswa diberi peringatan |
|---|---|---|---|---|
| 0,3 | 0,747 | **0,958** | 0,840 | 364 |
| 0,4 | 0,825 | 0,947 | 0,882 | 326 |
| **0,5** | **0,891** | **0,923** | **0,907** | **294** |
| 0,6 | 0,924 | 0,901 | 0,913 | 277 |
| 0,7 | **0,972** | 0,870 | 0,918 | 254 |

- **Threshold diturunkan** → lebih banyak mahasiswa diberi peringatan → recall naik, precision turun.
- **Threshold dinaikkan** → peringatan lebih selektif → precision naik, recall turun.

**Perhatikan:** F1 tertinggi justru ada di threshold 0,7. Ini efek `class_weight="balanced"`, yang "mendorong" probabilitas ke atas. Tapi F1 tertinggi belum tentu pilihan terbaik: proyek ini **memprioritaskan recall**. Di threshold 0,5, hanya 22 mahasiswa dropout yang terlewat. Di threshold 0,7, jumlahnya naik menjadi 37.

**Threshold adalah keputusan bisnis, bukan keputusan teknis.** Jika institusi punya banyak konselor, threshold 0,3 bisa dipakai untuk menangkap 95,8% mahasiswa berisiko. Jika konselornya sedikit, threshold 0,7 menghasilkan daftar yang lebih pendek dengan precision 97%.

Aplikasi Streamlit mengatasinya dengan **3 level risiko**: Tinggi (≥ 50%), Sedang (30–49%), Rendah (< 30%). Level **Sedang** menangkap wilayah "abu-abu" tanpa mengubah prediksi utama.

## 7.6 Perbandingan model dan cara memilihnya

| Model | CV F1 (± std) | CV Recall | Test Accuracy | Precision | Recall | Test F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0,872 (±0,011) | **0,854** | 0,926 | 0,891 | **0,923** | **0,907** | 0,973 |
| Random Forest | 0,869 (±0,017) | 0,836 | **0,927** | 0,911 | 0,901 | 0,906 | 0,969 |
| Gradient Boosting | **0,873** (±0,021) | 0,832 | 0,924 | **0,916** | 0,887 | 0,902 | **0,975** |

Model dipilih **hanya berdasarkan kolom CV** (lihat aturan "1 standar deviasi" di [Modul 6](06-machine-learning.md#68--memilih-model-secara-jujur-aturan-1-standar-deviasi)), bukan skor data uji. Kenapa?

> Jika model dipilih berdasarkan skor data uji, data uji ikut "dipakai untuk mengambil keputusan", sehingga skor data uji tidak lagi menjadi perkiraan jujur untuk data baru. Ini bentuk halus dari data leakage (*selection bias*).

Data uji harus diperlakukan seperti **ujian akhir**: hanya dibuka sekali, setelah semua keputusan dibuat.

Di data uji, Logistic Regression juga memberi **recall dan F1 tertinggi**. Pilihan berdasarkan CV terbukti konsisten dengan hasil akhirnya.

## 7.7 Permutation importance: fitur mana yang paling berpengaruh?

**Ide:** acak (*shuffle*) nilai satu fitur di data uji, lalu lihat seberapa turun skor F1. Jika skornya anjlok, fitur itu penting. Jika tidak berubah, model tidak bergantung padanya.

```python
permutation_importance(best_model, X_test, y_test, scoring="f1", n_repeats=15, random_state=42)
```

`n_repeats=15`: pengacakan diulang 15 kali lalu dirata-rata supaya hasilnya stabil. Kelebihannya: berlaku untuk **model apa pun** dan mengukur pada **fitur asli** (19 kolom), bukan 46 kolom hasil one-hot.

| Peringkat | Fitur | Penurunan F1 |
|---|---|---|
| 1 | MK lulus semester 2 | **0,227** |
| 2 | MK lulus semester 1 | 0,109 |
| 3 | Biaya kuliah lunas | 0,057 |
| 4 | MK diambil semester 2 | 0,052 |
| 5 | MK diambil semester 1 | 0,038 |
| 6 | Rata-rata nilai semester 2 | 0,030 |
| 7 | Debtor | 0,012 |
| 8 | Gender | 0,009 |

**Hati-hati menafsirkan:** "Rata-rata nilai semester 1" tidak muncul di daftar teratas. Itu bukan karena nilai tidak berkaitan dengan dropout (korelasinya −0,48). Informasinya **sudah terwakili** oleh "MK lulus semester 1" (ingat [Modul 3](03-memahami-data.md): grade dihitung dari MK yang lulus). Jika satu fitur diacak, fitur kembarannya masih membawa informasi yang sama. Permutation importance **meremehkan fitur-fitur yang saling berkorelasi**.

## 7.8 Membandingkan dengan model yang lebih "adil" dan lebih dini

Dua eksperimen tambahan (Logistic Regression dengan pengaturan yang sama). Varian **semester 1** juga ada di notebook, bagian *Evaluation → 4. Model deteksi dini*:

| Varian | F1 | Recall | ROC-AUC | Kapan dipakai |
|---|---|---|---|---|
| **Model utama (19 fitur)** | 0,907 | 0,923 | 0,973 | setelah semester 2 |
| Tanpa `Gender` (18 fitur) | 0,902 | 0,919 | 0,971 | jika institusi tidak ingin keputusan dipengaruhi gender |
| Hanya data semester 1 (15 fitur) | 0,857 | 0,884 | 0,946 | **satu semester lebih awal** |

Dampak kedua pilihan itu terhadap performa **kecil**. Pembahasan etika dan waktu deteksi ada di [Modul 10](10-bisnis-kesimpulan-etika.md).

## 7.9 Membaca hasil prediksi mahasiswa Enrolled

Setelah dievaluasi, model dipakai pada **794 mahasiswa Enrolled** (*Evaluation → 5* di notebook):

| Level risiko | Jumlah | % |
|---|---|---|
| 🔴 Tinggi (≥ 50%) | **438** | 55% |
| 🟠 Sedang (30–49%) | 150 | 19% |
| 🟢 Rendah (< 30%) | 206 | 26% |

Sebanyak **119 mahasiswa** memiliki probabilitas ≥ 90%. Mereka prioritas utama.

**Kenapa proporsi berisiko tingginya besar?** Mahasiswa Enrolled adalah mahasiswa yang **belum lulus di akhir masa studi normal**, jadi mereka memang tertinggal. Secara akademik mereka berada **di antara** lulusan dan mahasiswa dropout:

| Kelompok | Rata-rata MK lulus semester 2 | Approval rate semester 2 |
|---|---|---|
| Graduate | 6,2 | 93% |
| **Enrolled** | **4,1** | **67%** |
| Dropout | 1,9 | 31% |

**Dua catatan penting saat membaca hasil ini:**
1. **Pergeseran populasi.** Model dilatih untuk membedakan Dropout vs Graduate, sedangkan Enrolled adalah kelompok ketiga yang berbeda karakternya. Probabilitasnya paling tepat dipakai untuk **mengurutkan prioritas**, bukan sebagai kepastian "55% mahasiswa aktif pasti dropout". Evaluasi terbaik adalah mengecek ulang setelah status akhir mereka diketahui.
2. **Kualitas data.** 28 dari 37 mahasiswa Enrolled prodi *Animation and Multimedia Design* tidak punya data mata kuliah (tercatat 0). Rata-rata probabilitas mereka 0,60. Risikonya kemungkinan dinilai terlalu tinggi, jadi perlu verifikasi manual.

---

## ✍️ Cek pemahaman

1. Hitung precision dan recall jika TP = 200, FP = 100, FN = 50.
2. Institusi hanya punya 2 konselor dan ingin daftar mahasiswa sependek mungkin tapi tepat. Threshold mana yang Anda sarankan dari tabel 7.5, dan apa konsekuensinya?
3. Jelaskan AUC = 0,973 kepada kepala bagian akademik tanpa istilah teknis.
4. Kenapa kita tidak memilih model berdasarkan F1 data uji?
5. Kenapa permutation importance "Rata-rata nilai semester 1" kecil padahal korelasinya dengan dropout cukup kuat?
6. Seorang pimpinan membaca "438 dari 794 mahasiswa Enrolled berisiko tinggi" lalu menyimpulkan "55% mahasiswa aktif kita pasti dropout". Apa yang perlu Anda luruskan?

<details>
<summary>Lihat jawaban</summary>

1. Precision = 200 / 300 = 0,667; recall = 200 / 250 = 0,80.
2. Misalnya threshold 0,7: precision 97% (hampir semua yang didaftar memang berisiko) dengan 254 mahasiswa dari 726. Konsekuensinya recall turun ke 87%, sehingga 37 dari 284 mahasiswa dropout tidak masuk daftar. Untuk mahasiswa Enrolled, alternatifnya adalah mulai dari 119 mahasiswa dengan probabilitas ≥ 90%.
3. "Jika kita ambil satu mahasiswa yang akhirnya dropout dan satu yang lulus, sistem ini 97 dari 100 kali akan menilai mahasiswa yang dropout sebagai lebih berisiko."
4. Karena data uji harus menjadi penilaian yang tidak dipakai untuk mengambil keputusan. Jika dipakai untuk memilih model, skornya menjadi terlalu optimis.
5. Karena informasinya sudah dibawa oleh "MK lulus semester 1" (grade dihitung dari MK yang lulus). Saat satu fitur diacak, fitur kembarannya masih memberi informasi yang sama kepada model.
6. Angka itu adalah **peringkat risiko**, bukan kepastian. Mahasiswa Enrolled sudah tertinggal secara akademik, sehingga banyak yang terlihat mirip mahasiswa dropout. Model dilatih pada Dropout vs Graduate, jadi probabilitasnya paling tepat dipakai untuk menentukan siapa yang didampingi lebih dulu. Selain itu, sebagian data (prodi Animation and Multimedia Design) tidak lengkap dan perlu dicek manual.
</details>

➡️ Lanjut ke [Modul 8 — Aplikasi Streamlit & deployment](08-aplikasi-streamlit.md)
