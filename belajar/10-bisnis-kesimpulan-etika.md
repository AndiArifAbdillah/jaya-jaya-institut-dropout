# Modul 10 — Bisnis, Kesimpulan, Keterbatasan & Etika

> **Tujuan modul:** menghubungkan temuan data dengan keputusan bisnis, mengenali keterbatasan proyek secara jujur, dan memahami tanggung jawab etis dalam membuat model yang menyangkut manusia.

Data scientist yang baik tidak hanya bisa membuat model, tetapi juga bisa menjawab: *"lalu apa yang harus dilakukan?"* dan *"kapan model ini tidak boleh dipercaya?"*

---

## 10.1 Dari insight ke action item

Setiap rekomendasi di README harus bisa ditelusuri ke **bukti data**, dan harus punya **ukuran keberhasilan**.

| # | Action item | Insight yang mendasari | Cara mengukur keberhasilan |
|---|---|---|---|
| 1 | **Tindak lanjuti 438 mahasiswa Enrolled berisiko tinggi** (mulai dari 119 dengan probabilitas ≥ 90%) | Prediksi model pada 794 mahasiswa aktif | % mahasiswa prioritas yang sudah ditemui dosen wali; berapa yang akhirnya lulus |
| 2 | Sistem peringatan dini bertahap (setelah semester 1 → konfirmasi setelah semester 2) | Model utama mendeteksi 92% mahasiswa dropout; model semester 1 masih F1 0,857 | % mahasiswa risiko tinggi yang dihubungi ≤ 2 minggu |
| 3 | Pendampingan akademik | 0 MK lulus di semester 2 → 84% dropout; lulus semua → 6% | approval rate semester berikutnya pada mahasiswa yang didampingi |
| 4 | Intervensi finansial proaktif | Tidak lunas → 87%; debtor → 62% | % mahasiswa menunggak yang ikut skema cicilan & tetap aktif |
| 5 | Perluasan beasiswa | Penerima beasiswa hanya 12% dropout (vs 39%) | tingkat dropout penerima beasiswa baru |
| 6 | Jadwal fleksibel untuk mahasiswa dewasa | Usia > 24 tahun → > 50% dropout | tingkat dropout kelompok usia > 24 tahun |
| 7 | Evaluasi prodi berisiko tinggi | Equinculture 55%, Informatics 54%, Management (evening) 51%; 83% mahasiswa Enrolled Informatics berisiko tinggi | tingkat dropout per prodi per tahun |
| 8 | Monitoring rutin lewat dashboard | Dashboard + filter per segmen | rapat bulanan terlaksana; target tercapai |
| 9 | Perkaya data (kehadiran, LMS, nilai UTS) | Model saat ini baru bisa dipakai setelah semester berakhir | performa model semester-1 / mid-semester |

**Contoh target:** menurunkan dropout dari **32% menjadi < 25%**, dievaluasi setiap tahun ajaran.

### Cara membuktikan intervensi benar-benar berhasil
Membandingkan dropout "sebelum vs sesudah program" saja **tidak cukup**, karena angkatan yang berbeda bisa punya kondisi berbeda. Cara yang lebih kuat:
- **Uji coba (pilot) terkontrol:** sebagian mahasiswa berisiko mendapat program lebih dulu, sebagian menunggu, lalu hasil keduanya dibandingkan. Pertimbangkan etikanya: kelompok yang menunggu tetap harus mendapat layanan standar.
- Bandingkan pada kelompok yang **setara** (misalnya probabilitas risiko yang sama).

## 10.2 Pelajaran dari review: validitas label

Versi pertama proyek memberi label **0 kepada mahasiswa Graduate dan Enrolled**, lalu **ditolak reviewer**: target menjadi ambigu karena status akhir Enrolled belum diketahui. Setelah diperbaiki (Dropout vs Graduate, Enrolled diprediksi), performa justru **naik** (F1 0,818 → 0,907).

**Pelajaran umum:**
- Label adalah "kunci jawaban" bagi model. Label yang tidak pasti = kunci jawaban yang sebagian salah.
- Pisahkan dengan tegas: **data yang jawabannya sudah pasti** (untuk belajar dan menguji) vs **data yang jawabannya belum ada** (untuk diprediksi).
- Pertanyaan yang selalu perlu diajukan: *"Apakah label ini sudah final pada saat data dikumpulkan?"*

## 10.3 Keterbatasan proyek ini (jujur adalah kekuatan)

Menyebutkan keterbatasan **bukan** tanda proyek yang lemah. Justru itu menunjukkan Anda memahaminya secara mendalam.

### 1. Waktu deteksi: model butuh data semester 2
Model memakai 8 fitur akademik semester 1 **dan 2**, sehingga baru bisa dipakai **setelah semester 2 berakhir**.

Versi awal kesimpulan notebook menulis "deteksi dapat dilakukan segera setelah semester pertama berakhir". Kalimat itu **tidak tepat untuk model yang disimpan**, dan **sudah diperbaiki**. Notebook sekarang punya bagian *Evaluation → 4. Model deteksi dini* sebagai buktinya:

> Model semester 1 saja (15 fitur, algoritma & hyperparameter sama dengan model terbaik) masih mencapai **F1 0,857, recall 0,884, AUC 0,946**. Jadi deteksi satu semester lebih awal **bisa dilakukan** dengan model terpisah, dengan sedikit penurunan performa.

**Pelajaran:** setiap klaim di kesimpulan harus bisa ditunjuk buktinya di analisis. Jika belum ada buktinya, tambahkan analisisnya atau perlunak klaimnya. Idealnya institusi memakai **dua model bertahap**: model semester 1 untuk peringatan awal, dan model utama untuk konfirmasi.

### 2. Pergeseran populasi saat memprediksi Enrolled
Model dilatih pada dua kelompok yang "ekstrem" (akhirnya dropout vs akhirnya lulus), lalu dipakai pada kelompok ketiga (Enrolled) yang berada **di antaranya**: rata-rata approval rate semester 2 mereka 67%, dibanding 93% (Graduate) dan 31% (Dropout). Akibatnya 55% mahasiswa Enrolled dinilai berisiko tinggi.

Angka ini sebaiknya dipakai untuk **mengurutkan prioritas**, bukan dibaca sebagai "55% pasti dropout". Validasi yang ideal: catat status akhir mahasiswa Enrolled 1–2 tahun kemudian, lalu bandingkan dengan prediksinya.

### 3. Korelasi ≠ sebab-akibat
Model menemukan **asosiasi**, bukan **penyebab**. Contoh di [Modul 4](04-statistik-dan-eda.md): "kelas malam" tampak berisiko, padahal yang berperan adalah usia. Rekomendasi seperti "perluas beasiswa" masuk akal, tetapi efeknya harus dibuktikan lewat pilot. Ada kemungkinan mahasiswa penerima beasiswa memang sudah lebih berprestasi sejak awal.

### 4. Kualitas data: prodi 171 tanpa data akademik
Seluruh 180 mahasiswa dengan 0 MK diambil berasal dari prodi 171 (*Animation and Multimedia Design*). Pada data latih, tingkat dropout aktual prodi ini 46%, tetapi rata-rata probabilitas model 0,58. Pada mahasiswa Enrolled prodi ini, 28 dari 37 tidak punya data akademik, dan rata-rata probabilitasnya 0,60. Perbaikannya: telusuri ke sumber data. Jika memang tidak tercatat, tandai sebagai *missing* (bukan 0) atau buat penanganan khusus.

### 5. Generalisasi & data drift
Data berasal dari satu institusi di Portugal pada periode tertentu. Pola bisa berubah (kurikulum baru, kondisi ekonomi, pandemi). Artinya:
- Model harus **dievaluasi ulang secara berkala** dengan data terbaru.
- Model harus **dilatih ulang** jika performanya menurun (*model monitoring*).

### 6. Threshold bergantung kapasitas
Threshold 0,5 adalah titik awal. Nilai idealnya bergantung pada jumlah konselor dan biaya intervensi ([Modul 7](07-evaluasi-model.md#75-threshold-tarik-ulur-precision-vs-recall)).

## 10.4 Etika & keadilan (fairness)

Model ini memengaruhi **manusia**. Beberapa pertanyaan yang wajib dipikirkan:

### Apakah pantas memakai `Gender` sebagai fitur?
Model memberi mahasiswa laki-laki koefisien +0,196 (odds × 1,22). Secara statistik itu mencerminkan data. Tetapi jika sistem ini dipakai untuk keputusan yang **merugikan**, memakai gender bisa menjadi **diskriminasi**.

Eksperimen menunjukkan model **tanpa Gender** masih mencapai F1 0,902 (vs 0,907). Biaya performanya sangat kecil. Pilihan yang dapat dipertanggungjawabkan:
- Hapus fitur sensitif jika dampak performanya kecil, **atau**
- Pertahankan, tetapi **evaluasi performa per kelompok**.

Evaluasi per kelompok pada data uji:

| Kelompok | Recall |
|---|---|
| Perempuan (130 dropout) | 0,915 |
| Laki-laki (154 dropout) | 0,929 |
| Usia ≤ 24 tahun | **0,893** |
| Usia > 24 tahun | 0,960 |

Perbedaan gender kecil, tetapi model **lebih sering melewatkan mahasiswa muda** yang akan dropout. Kemungkinan penyebabnya: pola dropout mereka "kurang khas" karena tidak didukung sinyal usia. Institusi sebaiknya memantau kelompok ini lebih cermat.

Catatan: menghapus fitur sensitif tidak otomatis membuat model adil, karena fitur lain bisa menjadi "proksi". Evaluasi per kelompok tetap perlu dilakukan.

### Prediksi untuk membantu, bukan menghukum
Prediksi risiko dropout **harus** dipakai untuk **menawarkan bantuan** (bimbingan, beasiswa, cicilan), **bukan** untuk menolak pendaftaran, mencabut beasiswa, atau memberi label negatif. Jika mahasiswa diperlakukan sebagai "calon gagal", prediksi itu bisa menjadi **ramalan yang terwujud sendiri** (*self-fulfilling prophecy*). Ini sangat relevan untuk 438 mahasiswa Enrolled yang dinilai berisiko tinggi.

### Manusia tetap memutuskan (human-in-the-loop)
Aplikasi menampilkan **probabilitas + alasan + rekomendasi**, bukan keputusan final. Dosen wali tetap berbicara dengan mahasiswa dan memahami konteksnya.

### Privasi
Data akademik dan finansial mahasiswa bersifat sensitif:
- Batasi siapa yang bisa melihat prediksi individu (termasuk file `hasil_prediksi_enrolled.csv`).
- Dashboard untuk pimpinan sebaiknya menampilkan **agregat**, bukan nama mahasiswa.
- Patuhi regulasi perlindungan data (di Indonesia: **UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi**).

## 10.5 Langkah pengembangan berikutnya

1. Model bertahap: semester 1 → semester 2 (dan idealnya mid-semester dengan data kehadiran/LMS).
2. Validasi prediksi Enrolled dengan status akhir mereka di kemudian hari.
3. Tangani data prodi 171 dengan benar.
4. Evaluasi fairness per gender, usia, dan prodi secara rutin.
5. Kalibrasi probabilitas (supaya "70%" benar-benar berarti 70 dari 100 mahasiswa).
6. Monitoring model & pelatihan ulang berkala.
7. Uji dampak intervensi dengan pilot terkontrol.

---

## ✍️ Cek pemahaman

1. Pilih satu action item dan jelaskan insight data yang mendasarinya serta cara mengukur keberhasilannya.
2. Kenapa versi pertama proyek ditolak reviewer, dan apa pelajaran umumnya?
3. Kenapa kalimat "deteksi setelah semester pertama" perlu diluruskan untuk model yang disimpan?
4. Kenapa 55% mahasiswa Enrolled berisiko tinggi tidak boleh dibaca sebagai "55% pasti dropout"?
5. Sebutkan dua risiko etis dari sistem prediksi dropout dan cara mencegahnya.
6. Kenapa menghapus fitur `Gender` belum menjamin model menjadi adil?

<details>
<summary>Lihat jawaban</summary>

1. Contoh: pendampingan akademik. Dasarnya, mahasiswa yang tidak lulus satu pun MK di semester 2 memiliki tingkat dropout 84% (bahkan 94% jika data prodi 171 dikeluarkan). Keberhasilannya diukur dari approval rate semester berikutnya dan tingkat dropout mahasiswa yang didampingi, dibandingkan kelompok setara.
2. Enrolled diberi label 0 (sama dengan Graduate), padahal status akhirnya belum diketahui, sehingga target ambigu. Pelajarannya: data latih hanya boleh berisi label yang sudah final. Data yang labelnya belum ada adalah data untuk diprediksi.
3. Karena model yang disimpan memakai fitur semester 2, sehingga baru bisa dipakai setelah semester 2. Deteksi setelah semester 1 membutuhkan model terpisah (F1 0,857).
4. Model dilatih pada Dropout vs Graduate, sedangkan Enrolled adalah kelompok berbeda yang secara akademik berada di antara keduanya (pergeseran populasi). Probabilitasnya paling tepat dipakai untuk mengurutkan prioritas. Sebagian datanya (prodi 171) juga tidak lengkap.
5. (a) Diskriminasi karena fitur sensitif → evaluasi per kelompok atau hapus fitur. (b) Stigma / self-fulfilling prophecy → gunakan hanya untuk menawarkan bantuan, jaga kerahasiaan. (c) Pelanggaran privasi → batasi akses, tampilkan agregat.
6. Fitur lain bisa menjadi proksi dari gender (misalnya prodi tertentu yang didominasi satu gender). Keadilan harus diukur langsung dengan membandingkan performa model per kelompok.
</details>

➡️ Lanjut ke [Modul 11 — Latihan, kuis & simulasi review](11-latihan-dan-kuis.md)
