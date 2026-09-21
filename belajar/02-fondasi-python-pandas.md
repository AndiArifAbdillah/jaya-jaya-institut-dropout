# Modul 2 — Fondasi Python & pandas

> **Tujuan modul:** menguasai *semua* sintaks Python dan pandas yang muncul di proyek ini. Setiap konsep diberi contoh **langsung dari kode proyek**.

Buka `notebook.ipynb` di samping modul ini. Sebaiknya jalankan setiap contoh di sel baru.

---

## 2.1 Python yang perlu Anda kuasai

### Variabel, list, dictionary

```python
RANDOM_STATE = 42                                  # konstanta (huruf besar = konvensi "jangan diubah")
STATUS_ORDER = ["Graduate", "Dropout", "Enrolled"] # list: urutan penting
STATUS_COLORS = {"Graduate": "#2a78d6",            # dict: pasangan kunci → nilai
                 "Dropout": "#eb6834",
                 "Enrolled": "#1baf7a"}

STATUS_COLORS["Dropout"]   # → "#eb6834"
```

Di proyek ini dict dipakai untuk **kamus kode → nama**:

```python
COURSE_MAP = {33: "Biofuel Production Technologies", 9500: "Nursing", ...}
COURSE_MAP[9500]           # → "Nursing"
```

### f-string (format teks)

```python
status_pct = 32.1203
f"Hampir 1 dari 3 mahasiswa dropout ({status_pct:.1f}%)"   # → "... (32.1%)"
f"{4424:,}"                                                 # → "4,424"  (pemisah ribuan)
f"{0.831:.0%}"                                              # → "83%"    (format persen)
```

`:.1f` = 1 angka desimal, `:,` = pemisah ribuan, `:.0%` = kali 100 lalu tambah `%`.

### List comprehension & dict comprehension

Cara singkat membuat list/dict dari list lain. Dari notebook (bagian seleksi fitur):

```python
ALL_FEATURES = [c for c in df.columns if c not in ["Status", "is_dropout", ...]]
#               ^ ambil c   ^ untuk setiap kolom  ^ asalkan bukan kolom-kolom ini
```

Dari `app.py`:

```python
**{f"Curricular_units_{sem}_sem_{k}": v for sem, vals in semesters.items() for k, v in vals.items()}
```

Ini membuat dict berisi 8 kolom akademik (`Curricular_units_1st_sem_enrolled`, dst.) sekaligus.

### Fungsi

```python
def build_preprocessor(categorical, numerical):
    return ColumnTransformer([...])
```

Fungsi dipakai supaya kode yang sama tidak ditulis berulang. `build_preprocessor` dipanggil 3 kali di notebook (perbandingan fitur, pipeline utama, GridSearch).

### `zip`, `enumerate`, `for`

```python
for ax, (col, (title, labels)) in zip(axes.flat, binary_features.items()):
    ...
```

`zip` memasangkan dua urutan: grafik ke-1 dengan fitur ke-1, dan seterusnya. Pola ini dipakai untuk membuat 6 grafik sekaligus.

---

## 2.2 NumPy: satu fungsi yang sering muncul

```python
np.where(kondisi, nilai_jika_benar, nilai_jika_salah)
```

Dari notebook (menghitung approval rate):

```python
enrolled = df["Curricular_units_2nd_sem_enrolled"]
df["Approval_rate_2nd_sem"] = np.where(
    enrolled > 0,                                                    # jika ambil mata kuliah
    df["Curricular_units_2nd_sem_approved"] / enrolled.replace(0, 1),  # → lulus / diambil
    0)                                                                # jika tidak ambil → 0
```

> ❓ **Kenapa `enrolled.replace(0, 1)`?** `np.where` menghitung **kedua** cabang untuk semua baris. Kalau `enrolled` = 0, pembagian menghasilkan `inf`/`NaN` dan memunculkan warning. Mengganti 0 dengan 1 hanya untuk menghindari pembagian nol — hasilnya tetap dibuang karena kondisinya salah.

---

## 2.3 pandas: DataFrame dan Series

- **DataFrame** = tabel (baris × kolom). `df` di proyek ini berukuran 4.424 × 37.
- **Series** = satu kolom. `df["Status"]` adalah Series.

### Membaca data

```python
df = pd.read_csv("data.csv", sep=";")
```

⚠️ Dataset ini dipisahkan **titik koma**, bukan koma. Tanpa `sep=";"`, pandas akan membaca seluruh baris sebagai 1 kolom.

### Mengenal data

| Perintah | Hasil |
|---|---|
| `df.shape` | `(4424, 37)` |
| `df.head()` | 5 baris pertama |
| `df.info()` | nama kolom, tipe data, jumlah non-null |
| `df.describe().T` | statistik (mean, std, min, kuartil, max) setiap kolom numerik; `.T` = transpose agar mudah dibaca |
| `df.isna().sum().sum()` | total missing value → **0** |
| `df.duplicated().sum()` | jumlah baris duplikat → **0** |

### Memilih data

```python
df["Status"]                     # satu kolom (Series)
df[["Course", "Status"]]         # beberapa kolom (DataFrame)
df[df["Status"] == "Dropout"]    # filter baris
df.loc[baris, kolom]             # berdasarkan label
df.iloc[0]                       # berdasarkan posisi
```

---

## 2.4 Trik terpenting di proyek ini: **rata-rata dari 0/1 = persentase**

```python
df["is_dropout"] = (df["Status"] == "Dropout").astype(int)   # True/False → 1/0
df["is_dropout"].mean()                                       # → 0.3212 = 32,1%
```

Kenapa berhasil? Jika ada 1.421 angka 1 dan 3.003 angka 0, rata-ratanya = 1.421 / 4.424 = 0,321. **Rata-rata kolom biner = proporsi nilai 1.** Trik ini dipakai di hampir semua grafik tingkat dropout, dan juga di SQL dashboard (`AVG(is_dropout)`).

---

## 2.5 `value_counts`

```python
df["Status"].value_counts()
# Graduate    2209
# Dropout     1421
# Enrolled     794

df["Status"].value_counts(normalize=True)   # dalam proporsi: 0.499, 0.321, 0.179
df["Status"].value_counts().reindex(STATUS_ORDER)   # atur ulang urutan sesuai list
```

---

## 2.6 `groupby`: "untuk setiap kelompok, hitung sesuatu"

Ini operasi **paling penting** dalam analisis data.

```python
df.groupby("Tuition_fees_up_to_date")["is_dropout"].agg(["mean", "size"])
```

| Tuition_fees_up_to_date | mean | size |
|---|---|---|
| 0 (tidak lunas) | 0.866 | 528 |
| 1 (lunas) | 0.247 | 3896 |

Cara membaca: *"dari 528 mahasiswa yang biaya kuliahnya tidak lunas, 86,6% dropout."*

Tiga langkah di balik layar (**split–apply–combine**):
1. **Split**: pisahkan baris menjadi kelompok 0 dan 1.
2. **Apply**: hitung `mean` dan `size` di setiap kelompok.
3. **Combine**: gabungkan hasilnya menjadi tabel.

Rata-rata per status (dipakai di bagian performa akademik):

```python
df.groupby("Status")[["Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade"]].mean()
```

---

## 2.7 `pd.crosstab`: tabel silang

```python
pd.crosstab(df["Approval_rate_2nd_group"], df["Status"], normalize="index")
```

`normalize="index"` = setiap **baris** dijumlahkan menjadi 100%. Hasilnya adalah komposisi status di tiap kelompok approval rate — dasar grafik batang bertumpuk "84% dropout".

---

## 2.8 `pd.cut`: mengubah angka menjadi kelompok

```python
df["Age_group"] = pd.cut(df["Age_at_enrollment"],
                         bins=[0, 20, 24, 30, 40, 100],
                         labels=["≤20", "21–24", "25–30", "31–40", ">40"])
```

`bins=[0, 20, 24, ...]` artinya interval `(0, 20]`, `(20, 24]`, dst. — **kanan inklusif**. Jadi usia 20 masuk "≤20", usia 21 masuk "21–24".

---

## 2.9 `map`: menerjemahkan kode

```python
df["Course"].map(COURSE_MAP)          # 9500 → "Nursing"
df["Gender"].map({0: "Female", 1: "Male"})
```

Dipakai untuk membuat `df_dashboard` — tabel berlabel yang dikirim ke PostgreSQL agar dashboard mudah dibaca.

---

## 2.10 `sort_values`, `query`, `round`

```python
course_rate.sort_values("mean")          # urutkan
app_rate.query("size >= 70")             # filter dengan ekspresi teks
results.round(4)                         # bulatkan
```

---

## ✍️ Cek pemahaman

1. Apa hasil `pd.Series([1, 0, 0, 1, 1]).mean()` dan apa artinya jika kolom itu adalah `is_dropout`?
2. Tuliskan kode untuk menghitung tingkat dropout per program studi, diurutkan dari tertinggi.
3. Kenapa `pd.read_csv("data.csv")` tanpa `sep=";"` menghasilkan DataFrame berukuran 4.424 × 1?
4. Usia 24 masuk kelompok mana pada `pd.cut` di atas?

<details>
<summary>Lihat jawaban</summary>

1. `0.6` → 60% baris berisi 1, artinya 60% mahasiswa dropout.
2. ```python
   df.groupby("Course")["is_dropout"].mean().sort_values(ascending=False)
   ```
   (Bonus: `.agg(["mean", "size"])` agar terlihat jumlah mahasiswanya — penting karena prodi kecil bisa menyesatkan.)
3. Pemisah default `read_csv` adalah koma. File ini tidak punya koma di header, jadi seluruh baris dianggap satu kolom.
4. `"21–24"`, karena interval `(20, 24]` inklusif di kanan.
</details>

## 🧪 Latihan mini

Jalankan di notebook:

```python
# 1. Berapa persen mahasiswa penerima beasiswa?
df["Scholarship_holder"].mean()

# 2. Rata-rata usia mahasiswa dropout vs graduate
df.groupby("Status")["Age_at_enrollment"].mean()

# 3. Tingkat dropout per kombinasi gender & waktu kuliah
df.groupby(["Gender", "Daytime_evening_attendance"])["is_dropout"].agg(["mean", "size"])
```

Pertanyaan untuk latihan 3: kombinasi mana yang paling berisiko? Apakah jumlah mahasiswanya cukup besar untuk dipercaya?

➡️ Lanjut ke [Modul 3 — Memahami data](03-memahami-data.md)
