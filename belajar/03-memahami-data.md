# Modul 3 — Memahami Data

> **Tujuan modul:** mengenal setiap kolom dataset, tipe datanya, cara membacanya, dan jebakan-jebakan yang bisa membuat analisis salah.

Prinsip yang harus selalu diingat: **model tidak lebih pintar dari datanya.** Waktu yang dihabiskan untuk memahami data adalah investasi terbaik dalam proyek data science.

---

## 3.1 Asal-usul dataset

- Nama: **Students' Performance** (Dicoding), aslinya dari UCI Machine Learning Repository: *Predict Students' Dropout and Academic Success* (Realinho dkk., 2021).
- Sumber: sebuah institusi pendidikan tinggi di **Portugal**. Hampir semua mahasiswanya (97,5%) berkewarganegaraan Portugis.
- Ukuran: **4.424 baris** (1 baris = 1 mahasiswa) × **37 kolom**.
- Kualitas: **0 missing value, 0 duplikat.**
- Semua kolom sudah berupa angka kecuali `Status`: 29 kolom integer, 7 kolom desimal, 1 kolom teks.

## 3.2 Kolom target: `Status`

| Nilai | Arti | Jumlah | % |
|---|---|---|---|
| `Graduate` | Lulus | 2.209 | 49,9% |
| `Dropout` | Keluar sebelum lulus | 1.421 | 32,1% |
| `Enrolled` | Masih terdaftar **di akhir masa studi normal** (belum lulus, belum keluar) | 794 | 17,9% |

> ⚠️ `Enrolled` bukan berarti "mahasiswa aktif yang baik-baik saja". Artinya mahasiswa tersebut **terlambat lulus**. Sebagian mungkin akhirnya lulus, sebagian mungkin dropout. Ini penting untuk [Modul 5](05-persiapan-data.md) (perumusan target) dan [Modul 10](10-bisnis-kesimpulan-etika.md) (keterbatasan).

## 3.3 Enam kelompok fitur

### A. Demografi
| Kolom | Isi | Tipe |
|---|---|---|
| `Marital_status` | 1 single, 2 married, 3 widower, 4 divorced, 5 facto union, 6 legally separated | kategori berkode |
| `Nacionality` *(salah eja dari sumber aslinya)* | 21 kode negara | kategori berkode |
| `Gender` | 1 laki-laki, 0 perempuan | biner |
| `Age_at_enrollment` | usia saat mendaftar (17–70 tahun, median 20) | numerik |
| `International` | 1 mahasiswa internasional | biner |
| `Displaced` | 1 tinggal jauh dari rumah (perantau) | biner |
| `Educational_special_needs` | 1 berkebutuhan khusus | biner |

### B. Pendaftaran & latar belakang
| Kolom | Isi | Tipe |
|---|---|---|
| `Application_mode` | 18 jalur masuk (mis. 1 = fase 1 umum, 39 = usia di atas 23 tahun) | kategori berkode |
| `Application_order` | urutan pilihan (0 = pilihan pertama … 9) | ordinal |
| `Course` | 17 program studi (mis. 9500 = Nursing, 9119 = Informatics Engineering) | kategori berkode |
| `Daytime_evening_attendance` | 1 kelas siang, 0 kelas malam | biner |
| `Previous_qualification` | 17 jenis pendidikan sebelumnya | kategori berkode |
| `Previous_qualification_grade` | nilai pendidikan sebelumnya (skala 0–200, data: 95–190) | numerik |
| `Admission_grade` | nilai masuk (skala 0–200, data: 95–190) | numerik |

### C. Orang tua
`Mothers_qualification`, `Fathers_qualification`, `Mothers_occupation`, `Fathers_occupation` — kategori berkode dengan 29–46 kategori.

### D. Finansial
| Kolom | Isi |
|---|---|
| `Debtor` | 1 memiliki tunggakan |
| `Tuition_fees_up_to_date` | 1 biaya kuliah lunas/tepat waktu |
| `Scholarship_holder` | 1 penerima beasiswa |

### E. Akademik semester 1 dan 2 (6 kolom × 2 semester)
| Kolom (`Curricular_units_1st_sem_...`) | Arti |
|---|---|
| `credited` | mata kuliah yang diakui/dikreditkan (mis. dari institusi lain) |
| `enrolled` | mata kuliah yang **diambil** |
| `evaluations` | jumlah evaluasi/ujian yang diikuti |
| `approved` | mata kuliah yang **lulus** |
| `grade` | rata-rata nilai (skala 0–20; di Portugal lulus = ≥10) |
| `without_evaluations` | mata kuliah yang diambil tapi tidak dievaluasi |

> 🔍 **Detail yang sering terlewat:** nilai `grade` bukan nol yang paling kecil adalah 9,8. Artinya `grade` adalah rata-rata dari mata kuliah yang **lulus** — jika tidak ada yang lulus, grade = 0. Karena itu `grade` dan `approved` **sangat berkaitan**. Hal ini menjelaskan temuan di [Modul 7](07-evaluasi-model.md): *permutation importance* nilai semester 1 kecil, karena informasinya sudah terwakili oleh jumlah mata kuliah yang lulus.

### F. Makroekonomi
`Unemployment_rate`, `Inflation_rate`, `GDP`. Kolom ini hanya punya **9–10 nilai unik**, karena nilainya sama untuk semua mahasiswa yang masuk pada tahun yang sama. Artinya kolom ini lebih mirip "penanda tahun masuk" daripada kondisi ekonomi masing-masing mahasiswa.

## 3.4 Jebakan besar: angka yang sebenarnya **label**

`Course = 9500` **bukan** berarti "9500 kali lebih besar" dari `Course = 33`. Angka itu hanyalah **kode nama**, seperti nomor punggung pemain bola.

Konsekuensinya:
- ❌ Korelasi `Course` atau `Application_mode` dengan dropout **tidak bermakna**. Di notebook, `Application_mode` tampak berkorelasi +0,198, padahal itu kebetulan urutan kode.
- ❌ Memasukkan `Course` sebagai angka ke Logistic Regression memaksa model menganggap prodi 9991 "lebih" dari prodi 33.
- ✅ Solusinya adalah **One-Hot Encoding**: satu kolom 0/1 untuk setiap kategori ([Modul 5](05-persiapan-data.md)).

Jenis-jenis data yang perlu dibedakan:

| Jenis | Contoh | Operasi yang masuk akal |
|---|---|---|
| **Nominal** (kategori tanpa urutan) | `Course`, `Application_mode`, `Marital_status` | hitung frekuensi, one-hot |
| **Ordinal** (kategori berurutan) | `Application_order` | urutkan, median |
| **Biner** | `Debtor`, `Gender` | proporsi, rata-rata = persentase |
| **Numerik/kontinu** | `Admission_grade`, `Age_at_enrollment` | rata-rata, std, korelasi |
| **Hitungan (count)** | `..._enrolled`, `..._approved` | rata-rata, rasio |

## 3.5 Temuan kualitas data yang penting 🔎

Data yang "bersih" (tanpa missing value) belum tentu **benar**. Coba jalankan:

```python
df[df["Curricular_units_1st_sem_enrolled"] == 0]["Course"].value_counts()
```

Hasilnya: **semua 180 mahasiswa** yang tercatat mengambil 0 mata kuliah berasal dari **satu prodi: 171 (Animation and Multimedia Design)**. Dari 215 mahasiswa prodi itu, 180 tidak punya data akademik sama sekali. Padahal 75 di antaranya **lulus**.

Kesimpulannya: prodi ini kemungkinan besar **tidak mencatat data mata kuliahnya** di sistem sumber. Dampaknya:

1. Pada grafik approval rate, kelompok "0%" (870 mahasiswa) ikut memuat 180 mahasiswa ini. Jika mereka dikeluarkan, mahasiswa yang **mengambil** mata kuliah tetapi tidak lulus satu pun di semester 2 memiliki tingkat dropout **94,2%** (690 mahasiswa), lebih ekstrem dari angka 84% di notebook.
2. Untuk mahasiswa prodi 171, model **hampir menebak**: rata-rata probabilitasnya 0,49, sementara tingkat dropout aslinya 38%.

Temuan seperti ini tidak terlihat di `df.info()` atau `df.isna()`. Temuan ini hanya muncul kalau Anda **bertanya dan memeriksa**. Perbaikannya dibahas di [Modul 10](10-bisnis-kesimpulan-etika.md) dan [latihan modul 11](11-latihan-dan-kuis.md).

## 3.6 Ketidakseimbangan kelas

Jika target dibuat biner (Dropout vs lainnya): **32% vs 68%**. Artinya model "bodoh" yang **selalu** menebak "tidak dropout" sudah mendapat **akurasi 67,9%** tanpa belajar apa pun. Inilah alasan akurasi bukan metrik utama ([Modul 7](07-evaluasi-model.md)).

---

## ✍️ Cek pemahaman

1. Kenapa `Enrolled` tidak bisa langsung dianggap "mahasiswa yang aman"?
2. Seorang teman menghitung `df[["Course", "is_dropout"]].corr()` dan mendapat -0,03, lalu menyimpulkan "program studi tidak berpengaruh terhadap dropout". Apa yang salah?
3. Kenapa kolom `GDP` kurang tepat disebut "kondisi ekonomi mahasiswa"?
4. Mahasiswa dengan `Curricular_units_1st_sem_grade = 0` — apa kemungkinan artinya?

<details>
<summary>Lihat jawaban</summary>

1. `Enrolled` berarti belum lulus di akhir masa studi normal. Mereka terlambat, dan sebagian mungkin akhirnya dropout.
2. `Course` adalah kategori nominal. Korelasi Pearson mengasumsikan angka yang punya urutan dan jarak bermakna, jadi hasilnya tidak berarti apa-apa. Cara yang benar adalah membandingkan **tingkat dropout per prodi**: hasilnya bervariasi dari 15% (Nursing) sampai 55% (Equinculture), jadi prodi **sangat** berpengaruh.
3. Nilainya sama untuk semua mahasiswa yang masuk di tahun yang sama (hanya 10 nilai unik), jadi mencerminkan tahun masuk, bukan kondisi ekonomi pribadi.
4. Tidak ada mata kuliah yang lulus di semester itu, **atau** datanya tidak tercatat (seperti prodi 171).
</details>

➡️ Lanjut ke [Modul 4 — Statistik & EDA](04-statistik-dan-eda.md)
