# Modul 7 — Evaluasi Model

> **Tujuan modul:** bisa menghitung dan menjelaskan setiap metrik **dengan tangan**, memahami kenapa recall dan F1 dipakai sebagai metrik utama, dan memahami cara kerja threshold serta feature importance.

---

## 7.1 Confusion matrix model terpilih (data uji, 885 mahasiswa)

|  | **Prediksi: Tidak Dropout** | **Prediksi: Dropout** | Total |
|---|---|---|---|
| **Aktual: Tidak Dropout** | **TN = 544** ✅ | **FP = 57** ⚠️ | 601 |
| **Aktual: Dropout** | **FN = 48** ❌ | **TP = 236** ✅ | 284 |
| Total | 592 | 293 | 885 |

| Sel | Nama | Arti bisnis |
|---|---|---|
| TP | True Positive | mahasiswa berisiko **terdeteksi**, lalu dibimbing ✅ |
| TN | True Negative | mahasiswa aman diprediksi aman ✅ |
| FP | False Positive | mahasiswa aman **diberi peringatan** → biaya kecil (sesi bimbingan tambahan) |
| FN | False Negative | mahasiswa berisiko **terlewat** → biaya besar (mahasiswa dropout tanpa pernah dibantu) |

**Asimetri biaya** inilah alasan utama pemilihan metrik: FN jauh lebih mahal daripada FP.

## 7.2 Menghitung metrik dengan tangan

| Metrik | Rumus | Hitungan | Hasil | Pertanyaan yang dijawab |
|---|---|---|---|---|
| **Accuracy** | (TP+TN) / total | (236+544) / 885 | **0,881** | Berapa persen tebakan benar? |
| **Precision** | TP / (TP+FP) | 236 / 293 | **0,805** | Dari yang diberi peringatan, berapa yang benar-benar berisiko? |
| **Recall** (sensitivity) | TP / (TP+FN) | 236 / 284 | **0,831** | Dari semua yang akan dropout, berapa yang tertangkap? |
| **F1** | 2·P·R / (P+R) | 2·0,805·0,831 / 1,636 | **0,818** | Keseimbangan precision & recall |
| Specificity | TN / (TN+FP) | 544 / 601 | 0,905 | Dari yang aman, berapa yang diprediksi aman? |

> 🧮 **Kenapa F1 memakai rata-rata harmonik, bukan rata-rata biasa?** Rata-rata harmonik "menghukum" ketimpangan. Model dengan P = 1,0 dan R = 0,1 punya rata-rata biasa 0,55, tetapi F1-nya hanya 0,18. F1 hanya bisa tinggi jika **keduanya** tinggi.

## 7.3 Kenapa akurasi 88% bukan angka terpenting

Ingat baseline dari [Modul 3](03-memahami-data.md): model yang **selalu** menebak "tidak dropout" sudah mendapat **akurasi 67,9%**, padahal recall-nya **0%** karena tidak ada satu pun mahasiswa berisiko yang terdeteksi. Pada data yang tidak seimbang, akurasi bisa tinggi padahal model tidak berguna.

## 7.4 ROC curve dan AUC

Model sebenarnya menghasilkan **probabilitas**, bukan 0/1. Keputusan 0/1 bergantung pada **threshold** (default 0,5).

**ROC curve** menggambar, untuk *semua* kemungkinan threshold:
- sumbu Y: **True Positive Rate** (= recall)
- sumbu X: **False Positive Rate** (= FP / (FP+TN) = 1 − specificity)

**AUC** (Area Under Curve) = luas di bawah kurva:
- 0,5 = sama dengan tebakan acak (garis diagonal)
- 1,0 = sempurna
- **0,930** = model proyek ini

**Arti AUC yang intuitif:** jika Anda mengambil **1 mahasiswa dropout dan 1 mahasiswa tidak dropout secara acak**, ada **93% peluang** model memberi probabilitas lebih tinggi kepada yang dropout. AUC mengukur kemampuan model **mengurutkan** risiko, dan itu sangat relevan untuk fitur "prediksi batch" yang mengurutkan mahasiswa dari risiko tertinggi.

Ketiga model punya AUC hampir sama (0,929–0,930). Artinya kemampuan mengurutkannya setara, dan perbedaannya lebih banyak di **posisi threshold** efektif (dipengaruhi `class_weight`).

## 7.5 Threshold: tarik-ulur precision vs recall

Hasil model terpilih pada data uji untuk berbagai threshold:

| Threshold | Precision | Recall | F1 | Mahasiswa diberi peringatan |
|---|---|---|---|---|
| 0,3 | 0,637 | **0,915** | 0,751 | 408 |
| 0,4 | 0,735 | 0,887 | 0,804 | 343 |
| **0,5** | **0,805** | **0,831** | **0,818** | **293** |
| 0,6 | 0,854 | 0,782 | 0,816 | 260 |
| 0,7 | **0,894** | 0,715 | 0,795 | 227 |

- **Threshold diturunkan** → lebih banyak mahasiswa diberi peringatan → recall naik, precision turun.
- **Threshold dinaikkan** → peringatan lebih selektif → precision naik, recall turun.

**Threshold adalah keputusan bisnis, bukan keputusan teknis.** Jika institusi punya banyak konselor, threshold 0,3 bisa dipakai untuk menangkap 91,5% mahasiswa berisiko. Jika konselornya sedikit, threshold 0,6 menghasilkan daftar yang lebih pendek dan lebih tepat sasaran.

Aplikasi Streamlit mengatasinya dengan **3 level risiko**: Tinggi (≥ 50%), Sedang (30–49%), Rendah (< 30%). Level **Sedang** menangkap wilayah "abu-abu" antara threshold 0,3 dan 0,5 tanpa mengubah prediksi utama.

## 7.6 Memilih model: CV, bukan data uji

| Model | CV F1 | Test Accuracy | Precision | Recall | Test F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0,793** | 0,881 | 0,805 | **0,831** | **0,818** | **0,930** |
| Random Forest | 0,792 | 0,880 | 0,825 | 0,796 | 0,810 | 0,929 |
| Gradient Boosting | 0,779 | 0,875 | **0,856** | 0,732 | 0,789 | 0,929 |

Model dipilih berdasarkan kolom **CV F1**, bukan skor data uji. Kenapa?

> Jika model dipilih berdasarkan skor data uji, data uji ikut "dipakai untuk mengambil keputusan", sehingga skor data uji tidak lagi menjadi perkiraan jujur untuk data baru. Ini bentuk halus dari data leakage (*selection bias*).

Data uji harus diperlakukan seperti **ujian akhir**: hanya dibuka sekali, setelah semua keputusan dibuat.

Kebetulan, di proyek ini pilihan berdasarkan CV dan berdasarkan data uji sama-sama jatuh ke Logistic Regression. Itu kabar baik karena menunjukkan hasilnya konsisten.

## 7.7 Permutation importance: fitur mana yang paling berpengaruh?

**Ide:** acak (*shuffle*) nilai satu fitur di data uji, lalu lihat seberapa turun skor F1. Jika skornya anjlok, fitur itu penting. Jika tidak berubah, model tidak bergantung padanya.

```python
permutation_importance(best_model, X_test, y_test, scoring="f1", n_repeats=15, random_state=42)
```

`n_repeats=15`: pengacakan diulang 15 kali lalu dirata-rata supaya hasilnya stabil. Kelebihannya: berlaku untuk **model apa pun** dan mengukur pada **fitur asli** (19 kolom), bukan 47 kolom hasil one-hot.

| Peringkat | Fitur | Penurunan F1 |
|---|---|---|
| 1 | MK lulus semester 2 | **0,229** |
| 2 | Biaya kuliah lunas | 0,069 |
| 3 | MK diambil semester 2 | 0,059 |
| 4 | MK lulus semester 1 | 0,058 |
| 5 | MK diambil semester 1 | 0,030 |
| 6 | Rata-rata nilai semester 2 | 0,029 |
| 7 | Usia saat mendaftar | 0,014 |
| 8 | Program studi | 0,010 |

**Hati-hati menafsirkan:** "Rata-rata nilai semester 1" tampak tidak penting. Itu bukan karena nilai tidak berkaitan dengan dropout (korelasinya −0,48). Informasinya **sudah terwakili** oleh "MK lulus semester 1" (ingat [Modul 3](03-memahami-data.md): grade dihitung dari MK yang lulus). Jika satu fitur diacak, fitur kembarannya masih membawa informasi yang sama. Permutation importance **meremehkan fitur-fitur yang saling berkorelasi**.

## 7.8 Membandingkan dengan model yang lebih "adil" dan lebih dini

Dua eksperimen tambahan (Logistic Regression dengan pengaturan yang sama):

| Varian | F1 | Recall | ROC-AUC | Kapan dipakai |
|---|---|---|---|---|
| **Model utama (19 fitur)** | 0,818 | 0,831 | 0,930 | setelah semester 2 |
| Tanpa `Gender` (18 fitur) | 0,804 | 0,817 | 0,929 | jika institusi tidak ingin keputusan dipengaruhi gender |
| Hanya data semester 1 (15 fitur) | 0,778 | 0,789 | 0,905 | **satu semester lebih awal** |

Dampak kedua pilihan itu terhadap performa **kecil**. Pembahasan etika dan waktu deteksi ada di [Modul 10](10-bisnis-kesimpulan-etika.md).

---

## ✍️ Cek pemahaman

1. Hitung precision dan recall jika TP = 200, FP = 100, FN = 50.
2. Institusi hanya punya 2 konselor dan ingin daftar mahasiswa sependek mungkin tapi tepat. Threshold mana yang Anda sarankan dari tabel 7.5, dan apa konsekuensinya?
3. Jelaskan AUC = 0,93 kepada kepala bagian akademik tanpa istilah teknis.
4. Kenapa kita tidak memilih model berdasarkan F1 data uji?
5. Kenapa permutation importance "Rata-rata nilai semester 1" kecil padahal korelasinya dengan dropout cukup kuat?

<details>
<summary>Lihat jawaban</summary>

1. Precision = 200 / 300 = 0,667; recall = 200 / 250 = 0,80.
2. Misalnya threshold 0,7: precision 89% (hampir semua yang didaftar memang berisiko) dan hanya 227 mahasiswa (dari 885). Konsekuensinya recall turun ke 71,5%, jadi ±28% mahasiswa berisiko tidak masuk daftar. Kompromi yang wajar: threshold 0,6 (precision 85%, recall 78%).
3. "Jika kita ambil satu mahasiswa yang akhirnya dropout dan satu yang tidak, sistem ini 93 dari 100 kali akan menilai mahasiswa yang dropout sebagai lebih berisiko."
4. Karena data uji harus menjadi penilaian yang tidak dipakai untuk mengambil keputusan. Jika dipakai untuk memilih model, skornya menjadi terlalu optimis.
5. Karena informasinya sudah dibawa oleh "MK lulus semester 1" (grade dihitung dari MK yang lulus). Saat satu fitur diacak, fitur kembarannya masih memberi informasi yang sama kepada model.
</details>

➡️ Lanjut ke [Modul 8 — Aplikasi Streamlit & deployment](08-aplikasi-streamlit.md)
