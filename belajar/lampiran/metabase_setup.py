"""Membangun dashboard Metabase "Jaya Jaya Institut - Student Performance Dashboard" lewat REST API.

Skrip ini dipakai untuk membuat dashboard proyek (dibahas di belajar/09-dashboard-metabase.md).
Yang dilakukan:
  1. Setup awal Metabase (akun root@mail.com / root123) dan koneksi ke PostgreSQL `jji-postgres`.
  2. Membuat 18 question berbasis SQL native dengan 4 field filter ({{course}}, {{gender}},
     {{attendance}}, {{scholarship}}).
  3. Menyusun question dalam dashboard (grid 24 kolom) dan menghubungkan filter ke setiap question.

Prasyarat: container `jji-postgres` berisi tabel `students` (dibuat oleh notebook) dan container
`metabase` berjalan. BASE memakai port 3001 karena port 3000 di laptop pembuat sudah terpakai;
sesuaikan jika Metabase Anda berjalan di port lain.
"""
import json
import time
import uuid

import requests

BASE = "http://localhost:3001"
EMAIL, PASSWORD = "root@mail.com", "root123"
DROPOUT, GRADUATE, ENROLLED = "#eb6834", "#2a78d6", "#1baf7a"
STATUS_COLORS = {"Dropout": DROPOUT, "Graduate": GRADUATE, "Enrolled": ENROLLED}
s = requests.Session()


def api(method, path, **kw):
    r = s.request(method, BASE + path, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json() if r.text else None


# ------------------------------------------------------------------ setup
props = api("GET", "/api/session/properties")
if not props.get("has-user-setup"):
    api("POST", "/api/setup", json={
        "token": props["setup-token"],
        "user": {"email": EMAIL, "password": PASSWORD, "first_name": "Root", "last_name": "Admin",
                 "site_name": "Jaya Jaya Institut"},
        "prefs": {"site_name": "Jaya Jaya Institut", "site_locale": "en", "allow_tracking": False},
        "database": {"engine": "postgres", "name": "Jaya Jaya Institut DB",
                     "details": {"host": "jji-postgres", "port": 5432, "dbname": "jaya_jaya_institut",
                                 "user": "postgres", "password": "root123", "ssl": False}},
    })
session = api("POST", "/api/session", json={"username": EMAIL, "password": PASSWORD})
s.headers["X-Metabase-Session"] = session["id"]

db = next((d for d in api("GET", "/api/database")["data"] if d["engine"] == "postgres"), None)
if db is None:
    db = api("POST", "/api/database", json={
        "engine": "postgres", "name": "Jaya Jaya Institut DB", "is_full_sync": True,
        "details": {"host": "jji-postgres", "port": 5432, "dbname": "jaya_jaya_institut",
                    "user": "postgres", "password": "root123", "ssl": False}})
DB_ID = db["id"]
api("POST", f"/api/database/{DB_ID}/sync_schema")
for _ in range(60):
    meta = api("GET", f"/api/database/{DB_ID}/metadata")
    table = next((t for t in meta["tables"] if t["name"] == "students"), None)
    if table and table["fields"]:
        break
    time.sleep(2)
FIELDS = {f["name"]: f["id"] for f in table["fields"]}
print("database", DB_ID, "fields", len(FIELDS))

# ------------------------------------------------------------------ filters
for c in api("GET", "/api/card"):
    if c.get("database_id") == DB_ID:
        api("DELETE", f"/api/card/{c['id']}")

STATUS_SORT = "CASE status WHEN 'Dropout' THEN 1 WHEN 'Enrolled' THEN 2 ELSE 3 END"
FILTERS = [  # (slug, display name, column)
    ("course", "Program Studi", "course"),
    ("gender", "Gender", "gender"),
    ("attendance", "Waktu Kuliah", "attendance"),
    ("scholarship", "Penerima Beasiswa", "scholarship_holder"),
]
WHERE = " AND ".join("{{%s}}" % slug for slug, _, _ in FILTERS)


def template_tags():
    return {slug: {"id": str(uuid.uuid4()), "name": slug, "display-name": name, "type": "dimension",
                   "dimension": ["field", FIELDS[col], None], "widget-type": "string/="}
            for slug, name, col in FILTERS}


def pct(*cols, decimals=1):
    return {json.dumps(["name", c]): {"number_style": "percent", "decimals": decimals} for c in cols}


def card(name, sql, display, viz, description=None):
    return api("POST", "/api/card", json={
        "name": name, "description": description, "display": display, "visualization_settings": viz,
        "dataset_query": {"type": "native", "database": DB_ID,
                          "native": {"query": sql.replace("__WHERE__", WHERE).replace("__STATUS_SORT__", STATUS_SORT), "template-tags": template_tags()}},
    })["id"]


def rate_bar(name, column, label, x_title, order_sql=None, display="bar", description=None):
    order = order_sql or "1"
    sql = f"""SELECT {column} AS "{label}", AVG(is_dropout)::float AS dropout_rate, COUNT(*) AS jumlah_mahasiswa
FROM students WHERE __WHERE__
GROUP BY {column} ORDER BY {order}"""
    return card(name, sql, display, {
        "graph.dimensions": [label], "graph.metrics": ["dropout_rate"],
        "graph.show_values": True, "graph.x_axis.title_text": x_title,
        "graph.y_axis.title_text": "Tingkat dropout", "graph.y_axis.auto_range": True,
        "series_settings": {"dropout_rate": {"color": DROPOUT, "title": "Tingkat dropout"}},
        "column_settings": pct("dropout_rate", decimals=0),
    }, description)


cards = {}
# KPIs
cards["total"] = card("Total Mahasiswa", "SELECT COUNT(*) AS total_mahasiswa FROM students WHERE __WHERE__",
                      "scalar", {"scalar.field": "total_mahasiswa"})
for key, status, title in [("dropout", "Dropout", "Tingkat Dropout"), ("graduate", "Graduate", "Tingkat Kelulusan"),
                           ("enrolled", "Enrolled", "Masih Terdaftar (Enrolled)")]:
    cards[key] = card(title, f"SELECT AVG(CASE WHEN status = '{status}' THEN 1 ELSE 0 END)::float AS rate "
                             f"FROM students WHERE __WHERE__", "scalar",
                      {"scalar.field": "rate", "column_settings": pct("rate")})

cards["status_pie"] = card("Distribusi Status Mahasiswa", """SELECT status, COUNT(*) AS jumlah
FROM students WHERE __WHERE__ GROUP BY status ORDER BY jumlah DESC""", "pie", {
    "pie.dimension": "status", "pie.metric": "jumlah", "pie.show_legend": True, "pie.show_total": True,
    "pie.percent_visibility": "inside", "pie.colors": STATUS_COLORS,
})

cards["age_status"] = card("Komposisi Status per Kelompok Usia Saat Mendaftar", """SELECT age_group AS "Kelompok usia", status, COUNT(*) AS jumlah
FROM students WHERE __WHERE__
GROUP BY age_group, status
ORDER BY CASE age_group WHEN '≤20' THEN 1 WHEN '21–24' THEN 2 WHEN '25–30' THEN 3 WHEN '31–40' THEN 4 ELSE 5 END, __STATUS_SORT__""", "bar", {
    "graph.dimensions": ["Kelompok usia", "status"], "graph.metrics": ["jumlah"],
    "stackable.stack_type": "normalized", "graph.x_axis.title_text": "Kelompok usia",
    "graph.y_axis.title_text": "Proporsi mahasiswa",
    "series_settings": {k: {"color": v} for k, v in STATUS_COLORS.items()},
}, "Mahasiswa yang mendaftar di atas usia 24 tahun memiliki proporsi dropout jauh lebih tinggi.")

cards["tuition"] = rate_bar("Dropout: Biaya Kuliah Lunas?", "tuition_fees_up_to_date", "Biaya kuliah lunas",
                            "Biaya kuliah lunas")
cards["debtor"] = rate_bar("Dropout: Memiliki Tunggakan (Debtor)?", "debtor", "Debtor", "Memiliki tunggakan")
cards["scholarship"] = rate_bar("Dropout: Penerima Beasiswa?", "scholarship_holder", "Beasiswa", "Penerima beasiswa")

cards["approval_group"] = card("Komposisi Status per Approval Rate Semester 2", """SELECT sem2_approval_group AS "Approval rate smt 2", status, COUNT(*) AS jumlah
FROM students WHERE __WHERE__
GROUP BY sem2_approval_group, status
ORDER BY CASE sem2_approval_group WHEN '0%' THEN 1 WHEN '1–50%' THEN 2 WHEN '51–99%' THEN 3 ELSE 4 END, __STATUS_SORT__""", "bar", {
    "graph.dimensions": ["Approval rate smt 2", "status"], "graph.metrics": ["jumlah"],
    "stackable.stack_type": "normalized", "graph.x_axis.title_text": "Persentase mata kuliah lulus di semester 2",
    "graph.y_axis.title_text": "Proporsi mahasiswa",
    "series_settings": {k: {"color": v} for k, v in STATUS_COLORS.items()},
}, "Approval rate = mata kuliah lulus / mata kuliah diambil.")

cards["approved"] = card("Rata-rata Mata Kuliah Lulus per Semester", """SELECT semester AS "Semester", status, AVG(approved)::float AS rata_rata
FROM (
  SELECT status, 'Semester 1' AS semester, sem1_approved AS approved FROM students WHERE __WHERE__
  UNION ALL
  SELECT status, 'Semester 2', sem2_approved FROM students WHERE __WHERE__
) t GROUP BY semester, status ORDER BY semester, __STATUS_SORT__""", "bar", {
    "graph.dimensions": ["Semester", "status"], "graph.metrics": ["rata_rata"], "graph.show_values": True,
    "graph.y_axis.title_text": "Rata-rata mata kuliah lulus",
    "series_settings": {k: {"color": v} for k, v in STATUS_COLORS.items()},
    "column_settings": {json.dumps(["name", "rata_rata"]): {"decimals": 1}},
})
cards["grade"] = card("Rata-rata Nilai per Semester (skala 0–20)", """SELECT semester AS "Semester", status, AVG(grade)::float AS rata_rata
FROM (
  SELECT status, 'Semester 1' AS semester, sem1_grade AS grade FROM students WHERE __WHERE__
  UNION ALL
  SELECT status, 'Semester 2', sem2_grade FROM students WHERE __WHERE__
) t GROUP BY semester, status ORDER BY semester, __STATUS_SORT__""", "bar", {
    "graph.dimensions": ["Semester", "status"], "graph.metrics": ["rata_rata"], "graph.show_values": True,
    "graph.y_axis.title_text": "Rata-rata nilai",
    "series_settings": {k: {"color": v} for k, v in STATUS_COLORS.items()},
    "column_settings": {json.dumps(["name", "rata_rata"]): {"decimals": 1}},
})

cards["course"] = rate_bar("Tingkat Dropout per Program Studi", "course", "Program studi", "Program studi",
                           order_sql="dropout_rate DESC", display="row")
cards["app_mode"] = card("Tingkat Dropout per Jalur Pendaftaran (≥ 30 mahasiswa)", """SELECT application_mode AS "Jalur pendaftaran", AVG(is_dropout)::float AS dropout_rate, COUNT(*) AS jumlah_mahasiswa
FROM students WHERE __WHERE__
GROUP BY application_mode HAVING COUNT(*) >= 30 ORDER BY dropout_rate DESC""", "row", {
    "graph.dimensions": ["Jalur pendaftaran"], "graph.metrics": ["dropout_rate"], "graph.show_values": True,
    "graph.x_axis.title_text": "Jalur pendaftaran", "graph.y_axis.title_text": "Tingkat dropout",
    "series_settings": {"dropout_rate": {"color": DROPOUT, "title": "Tingkat dropout"}},
    "column_settings": pct("dropout_rate", decimals=0),
})
cards["gender"] = rate_bar("Dropout: Gender", "gender", "Gender", "Gender")
cards["attendance"] = rate_bar("Dropout: Waktu Kuliah", "attendance", "Waktu kuliah", "Waktu kuliah")
cards["marital"] = card("Dropout: Status Pernikahan (≥ 20 mahasiswa)", """SELECT marital_status AS "Status pernikahan", AVG(is_dropout)::float AS dropout_rate, COUNT(*) AS jumlah_mahasiswa
FROM students WHERE __WHERE__
GROUP BY marital_status HAVING COUNT(*) >= 20 ORDER BY jumlah_mahasiswa DESC""", "bar", {
    "graph.dimensions": ["Status pernikahan"], "graph.metrics": ["dropout_rate"], "graph.show_values": True,
    "graph.x_axis.title_text": "Status pernikahan", "graph.y_axis.title_text": "Tingkat dropout",
    "series_settings": {"dropout_rate": {"color": DROPOUT, "title": "Tingkat dropout"}},
    "column_settings": pct("dropout_rate", decimals=0),
})

# ------------------------------------------------------------------ dashboard
for d in api("GET", "/api/dashboard"):
    if d["name"] == "Jaya Jaya Institut - Student Performance Dashboard":
        api("DELETE", f"/api/dashboard/{d['id']}")
dash_id = api("POST", "/api/dashboard", json={
    "name": "Jaya Jaya Institut - Student Performance Dashboard",
    "description": "Monitoring performa dan faktor risiko dropout mahasiswa Jaya Jaya Institut.",
})["id"]


def text(md):
    return {"card_id": None, "visualization_settings": {
        "virtual_card": {"name": None, "display": "text", "visualization_settings": {}, "dataset_query": {},
                         "archived": False}, "text": md}}


layout = [  # (key or text, col, row, size_x, size_y)
    (text("# Jaya Jaya Institut — Student Performance Dashboard\n"
          "Monitoring tingkat dropout dan faktor-faktor utamanya. Gunakan filter di atas untuk melihat segmen tertentu. "
          "**Oranye = Dropout, Hijau = Enrolled, Biru = Graduate.**"), 0, 0, 24, 2),
    ("total", 0, 2, 6, 3), ("dropout", 6, 2, 6, 3), ("graduate", 12, 2, 6, 3), ("enrolled", 18, 2, 6, 3),
    ("status_pie", 0, 5, 8, 8), ("age_status", 8, 5, 16, 8),
    (text("## Faktor Finansial\nStatus pembayaran biaya kuliah, tunggakan, dan beasiswa sangat berkaitan dengan dropout."), 0, 13, 24, 2),
    ("tuition", 0, 15, 8, 6), ("debtor", 8, 15, 8, 6), ("scholarship", 16, 15, 8, 6),
    (text("## Performa Akademik\nMahasiswa dropout lulus lebih sedikit mata kuliah dan memiliki nilai jauh lebih rendah sejak semester 1."), 0, 21, 24, 2),
    ("approved", 0, 23, 8, 7), ("grade", 8, 23, 8, 7), ("approval_group", 16, 23, 8, 7),
    (text("## Profil Pendaftaran & Demografi"), 0, 30, 24, 1),
    ("course", 0, 31, 12, 10), ("app_mode", 12, 31, 12, 10),
    ("gender", 0, 41, 8, 6), ("attendance", 8, 41, 8, 6), ("marital", 16, 41, 8, 6),
]

parameters = [{"id": slug, "name": name, "slug": slug, "type": "string/=", "sectionId": "string"}
              for slug, name, _ in FILTERS]
dashcards = []
for i, (item, col, row, sx, sy) in enumerate(layout):
    base = {"id": -(i + 1), "col": col, "row": row, "size_x": sx, "size_y": sy, "series": []}
    if isinstance(item, dict):
        base.update(item)
        base["parameter_mappings"] = []
    else:
        cid = cards[item]
        base.update({"card_id": cid, "visualization_settings": {}, "parameter_mappings": [
            {"parameter_id": slug, "card_id": cid, "target": ["dimension", ["template-tag", slug]]}
            for slug, _, _ in FILTERS]})
    dashcards.append(base)

api("PUT", f"/api/dashboard/{dash_id}", json={"dashcards": dashcards, "parameters": parameters, "width": "full"})
api("PUT", "/api/setting/enable-public-sharing", json={"value": True})
public = api("POST", f"/api/dashboard/{dash_id}/public_link")
print("dashboard", dash_id)
print("private:", f"{BASE}/dashboard/{dash_id}")
print("public :", f"{BASE}/public/dashboard/{public['uuid']}")
