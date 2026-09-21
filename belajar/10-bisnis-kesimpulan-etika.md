# Modul 10 — Bisnis, Kesimpulan, Keterbatasan & Etika

> **Tujuan modul:** menghubungkan temuan data dengan keputusan bisnis, mengenali keterbatasan proyek secara jujur, dan memahami tanggung jawab etis dalam membuat model yang menyangkut manusia.

Data scientist yang baik tidak hanya bisa membuat model, tetapi juga bisa menjawab: *"lalu apa yang harus dilakukan?"* dan *"kapan model ini tidak boleh dipercaya?"*

---

## 10.1 Dari insight ke action item

Setiap rekomendasi di README harus bisa ditelusuri ke **bukti data**, dan harus punya **ukuran keberhasilan**.

| # | Action item | Insight yang mendasari | Cara mengukur keberhasilan |
|---|---|---|---|
| 1 | Sistem peringatan dini bertahap (setelah semester 1 → konfirmasi setelah semester 2) | Model utama mendeteksi 83% mahasiswa dropout; model semester 1 masih F1 0,778 | % mahasiswa risiko tinggi yang dihubungi ≤ 2 minggu |
| 2 | Pendampingan akademik | 0 MK lulus di semester 2 → 84% dropout; lulus semua → 6% | approval rate semester berikutnya pada mahasiswa yang didampingi |
| 3 | Intervensi finansial proaktif | Tidak lunas → 87%; debtor → 62% | % mahasiswa menunggak yang ikut skema cicilan & tetap aktif |
| 4 | Perluasan beasiswa | Penerima beasiswa hanya 12% dropout (vs 39%) | tingkat dropout penerima beasiswa baru |
| 5 | Jadwal fleksibel untuk mahasiswa dewasa | Usia > 24 tahun → > 50% dropout | tingkat dropout kelompok usia > 24 tahun |
| 6 | Evaluasi prodi berisiko tinggi | Equinculture 55%, Informatics 54%, Management (evening) 51% | tingkat dropout per prodi per tahun |
| 7 | Monitoring rutin lewat dashboard | Dashboard + filter per segmen | rapat bulanan terlaksana; target tercapai |
| 8 | Perkaya data (kehadiran, LMS, nilai UTS) | Model saat ini baru bisa dipakai setelah semester berakhir | performa model semester-1 / mid-semester |

**Contoh target:** menurunkan dropout dari **32% menjadi < 25%**, dievaluasi setiap tahun ajaran.

### Cara membuktikan intervensi benar-benar berhasil
Membandingkan dropout "sebelum vs sesudah program" saja **tidak cukup**, karena angkatan yang berbeda bisa punya kondisi berbeda. Cara yang lebih kuat:
- **Uji coba (pilot) terkontrol:** sebagian mahasiswa berisiko mendapat program lebih dulu, sebagian menunggu, lalu hasil keduanya dibandingkan. Pertimbangkan etikanya: kelompok yang menunggu tetap harus mendapat layanan standar.
- Bandingkan pada kelompok yang **setara** (misalnya probabilitas risiko yang sama).

## 10.2 Keterbatasan proyek ini (jujur adalah kekuatan)

Menyebutkan keterbatasan **bukan** tanda proyek yang lemah. Justru itu menunjukkan Anda memahaminya secara mendalam.

### 1. Waktu deteksi: model butuh data semester 2
Model memakai 8 fitur akademik semester 1 **dan 2**, sehingga baru bisa dipakai **setelah semester 2 berakhir**.

Versi awal kesimpulan notebook menulis "deteksi dapat dilakukan segera setelah semester pertama berakhir". Kalimat itu **tidak tepat untuk model yang disimpan**, karena model tersebut membutuhkan data semester 2. Klaim ini **sudah diperbaiki** di notebook dan README. Notebook sekarang juga punya bagian baru *Evaluation → 4. Model deteksi dini* sebagai buktinya:

> Model semester 1 saja (15 fitur, hyperparameter sama dengan model utama) masih mencapai **F1 0,778, recall 0,789, AUC 0,905**. Jadi deteksi satu semester lebih awal **bisa dilakukan** dengan model terpisah, dengan sedikit penurunan performa.

**Pelajaran:** setiap klaim di kesimpulan harus bisa ditunjuk buktinya di analisis. Jika belum ada buktinya, tambahkan analisisnya atau perlunak klaimnya.

Idealnya institusi memakai **dua model bertahap**: model semester 1 untuk peringatan awal, dan model semester 2 untuk konfirmasi.

### 2. Korelasi ≠ sebab-akibat
Model menemukan **asosiasi**, bukan **penyebab**. Contoh di [Modul 4](04-statistik-dan-eda.md): "kelas malam" tampak berisiko, padahal yang berperan adalah usia. Rekomendasi seperti "perluas beasiswa" masuk akal, tetapi efeknya harus dibuktikan lewat pilot. Ada kemungkinan mahasiswa penerima beasiswa memang sudah lebih berprestasi sejak awal.

### 3. Label "Enrolled" yang ambigu
26,4% dari kelas "tidak dropout" adalah mahasiswa **Enrolled** (terlambat lulus). Sebagian dari mereka mungkin akhirnya dropout. Artinya ada sedikit **noise** pada label.

### 4. Kualitas data: prodi 171 tanpa data akademik
Seluruh 180 mahasiswa dengan 0 MK diambil berasal dari prodi 171. Untuk prodi ini model hampir menebak (rata-rata probabilitas 0,49 vs dropout aktual 38%). Perbaikannya: telusuri ke sumber data. Jika memang tidak tercatat, tandai sebagai *missing* (bukan 0) atau buat penanganan khusus.

### 5. Generalisasi & data drift
Data berasal dari satu institusi di Portugal pada periode tertentu. Pola bisa berubah (kurikulum baru, kondisi ekonomi, pandemi). Artinya:
- Model harus **dievaluasi ulang secara berkala** dengan data terbaru.
- Model harus **dilatih ulang** jika performanya menurun (*model monitoring*).

### 6. Threshold bergantung kapasitas
Threshold 0,5 adalah titik awal. Nilai idealnya bergantung pada jumlah konselor dan biaya intervensi ([Modul 7](07-evaluasi-model.md#75-threshold-tarik-ulur-precision-vs-recall)).

## 10.3 Etika & keadilan (fairness)

Model ini memengaruhi **manusia**. Beberapa pertanyaan yang wajib dipikirkan:

### Apakah pantas memakai `Gender` sebagai fitur?
Model memberi mahasiswa laki-laki koefisien +0,167 (odds × 1,18). Secara statistik itu mencerminkan data. Tetapi jika sistem ini dipakai untuk keputusan yang **merugikan**, memakai gender bisa menjadi **diskriminasi**.

Eksperimen menunjukkan model **tanpa Gender** masih mencapai F1 0,804 (vs 0,818). Biaya performanya kecil. Pilihan yang dapat dipertanggungjawabkan:
- Hapus fitur sensitif jika dampak performanya kecil, **atau**
- Pertahankan, tetapi **evaluasi performa per kelompok** (misalnya: apakah recall untuk perempuan sama baiknya dengan laki-laki?).

Catatan: menghapus fitur sensitif tidak otomatis membuat model adil, karena fitur lain bisa menjadi "proksi". Evaluasi per kelompok tetap perlu dilakukan.

### Prediksi untuk membantu, bukan menghukum
Prediksi risiko dropout **harus** dipakai untuk **menawarkan bantuan** (bimbingan, beasiswa, cicilan), **bukan** untuk menolak pendaftaran, mencabut beasiswa, atau memberi label negatif. Jika mahasiswa diperlakukan sebagai "calon gagal", prediksi itu bisa menjadi **ramalan yang terwujud sendiri** (*self-fulfilling prophecy*).

### Manusia tetap memutuskan (human-in-the-loop)
Aplikasi menampilkan **probabilitas + alasan + rekomendasi**, bukan keputusan final. Dosen wali tetap berbicara dengan mahasiswa dan memahami konteksnya.

### Privasi
Data akademik dan finansial mahasiswa bersifat sensitif:
- Batasi siapa yang bisa melihat prediksi individu.
- Dashboard untuk pimpinan sebaiknya menampilkan **agregat**, bukan nama mahasiswa.
- Patuhi regulasi perlindungan data (di Indonesia: **UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi**).

## 10.4 Langkah pengembangan berikutnya

1. Model bertahap: semester 1 → semester 2 (dan idealnya mid-semester dengan data kehadiran/LMS).
2. Tangani data prodi 171 dengan benar.
3. Evaluasi fairness per gender, usia, dan prodi.
4. Kalibrasi probabilitas (supaya "70%" benar-benar berarti 70 dari 100 mahasiswa).
5. Monitoring model & pelatihan ulang berkala.
6. Uji dampak intervensi dengan pilot terkontrol.

---

## ✍️ Cek pemahaman

1. Pilih satu action item dan jelaskan insight data yang mendasarinya serta cara mengukur keberhasilannya.
2. Kenapa kalimat "deteksi setelah semester pertama" perlu diluruskan untuk model yang disimpan?
3. Sebutkan dua risiko etis dari sistem prediksi dropout dan cara mencegahnya.
4. Kenapa menghapus fitur `Gender` belum menjamin model menjadi adil?

<details>
<summary>Lihat jawaban</summary>

1. Contoh: pendampingan akademik. Dasarnya, mahasiswa yang tidak lulus satu pun MK di semester 2 memiliki tingkat dropout 84% (bahkan 94% jika data prodi 171 dikeluarkan). Keberhasilannya diukur dari approval rate semester berikutnya dan tingkat dropout mahasiswa yang didampingi, dibandingkan kelompok setara.
2. Karena model yang disimpan memakai fitur semester 2, sehingga baru bisa dipakai setelah semester 2. Deteksi setelah semester 1 membutuhkan model terpisah (F1 0,778).
3. (a) Diskriminasi karena fitur sensitif → evaluasi per kelompok atau hapus fitur. (b) Stigma / self-fulfilling prophecy → gunakan hanya untuk menawarkan bantuan, jaga kerahasiaan. (c) Pelanggaran privasi → batasi akses, tampilkan agregat.
4. Fitur lain bisa menjadi proksi dari gender (misalnya prodi tertentu yang didominasi satu gender). Keadilan harus diukur langsung dengan membandingkan performa model per kelompok.
</details>

➡️ Lanjut ke [Modul 11 — Latihan, kuis & simulasi review](11-latihan-dan-kuis.md)
