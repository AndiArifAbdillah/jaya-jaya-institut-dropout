"""Jaya Jaya Institut - Dropout Early Warning System (Streamlit prototype)."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"
REFERENCE_DATA_PATH = BASE_DIR / "data.csv"
ENROLLED_PATH = BASE_DIR / "data_enrolled.csv"

RISK_LEVELS = [  # (batas bawah probabilitas, label, warna, ikon)
    (0.50, "Tinggi", "#d03b3b", "🔴"),
    (0.30, "Sedang", "#c98500", "🟠"),
    (0.00, "Rendah", "#1f8a4c", "🟢"),
]

COURSE_MAP = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening attendance)",
}
APPLICATION_MODE_MAP = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)", 7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)", 17: "2nd phase - general contingent",
    18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)", 39: "Over 23 years old",
    42: "Transfer", 43: "Change of course", 44: "Technological specialization diploma holders",
    51: "Change of institution/course", 53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}
FEATURE_LABELS = {
    "Application_mode": "Jalur pendaftaran",
    "Course": "Program studi",
    "Daytime_evening_attendance": "Waktu kuliah (siang)",
    "Previous_qualification_grade": "Nilai pendidikan sebelumnya",
    "Admission_grade": "Nilai masuk",
    "Displaced": "Mahasiswa perantau",
    "Debtor": "Memiliki tunggakan",
    "Tuition_fees_up_to_date": "Biaya kuliah lunas",
    "Gender": "Gender (laki-laki)",
    "Scholarship_holder": "Penerima beasiswa",
    "Age_at_enrollment": "Usia saat mendaftar",
    "Curricular_units_1st_sem_enrolled": "MK diambil smt 1",
    "Curricular_units_1st_sem_evaluations": "Evaluasi smt 1",
    "Curricular_units_1st_sem_approved": "MK lulus smt 1",
    "Curricular_units_1st_sem_grade": "Rata-rata nilai smt 1",
    "Curricular_units_2nd_sem_enrolled": "MK diambil smt 2",
    "Curricular_units_2nd_sem_evaluations": "Evaluasi smt 2",
    "Curricular_units_2nd_sem_approved": "MK lulus smt 2",
    "Curricular_units_2nd_sem_grade": "Rata-rata nilai smt 2",
}
YES_NO = {"Tidak": 0, "Ya": 1}

PRESETS = {
    "high": {
        "Course": 9119, "Application_mode": 39, "attendance": "Malam", "Previous_qualification_grade": 120.0,
        "Admission_grade": 115.0, "Age_at_enrollment": 29, "gender": "Laki-laki", "displaced": "Tidak",
        "scholarship": "Tidak", "debtor": "Ya", "tuition": "Tidak",
        "s1_enrolled": 6, "s1_evaluations": 8, "s1_approved": 2, "s1_grade": 10.5,
        "s2_enrolled": 6, "s2_evaluations": 6, "s2_approved": 0, "s2_grade": 0.0,
    },
    "low": {
        "Course": 9500, "Application_mode": 1, "attendance": "Siang", "Previous_qualification_grade": 140.0,
        "Admission_grade": 138.0, "Age_at_enrollment": 19, "gender": "Perempuan", "displaced": "Ya",
        "scholarship": "Ya", "debtor": "Tidak", "tuition": "Ya",
        "s1_enrolled": 7, "s1_evaluations": 8, "s1_approved": 7, "s1_grade": 13.5,
        "s2_enrolled": 7, "s2_evaluations": 8, "s2_approved": 7, "s2_grade": 13.8,
    },
}

st.set_page_config(page_title="Dropout Early Warning · Jaya Jaya Institut", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.risk-card {border-radius: 14px; padding: 1.2rem 1.4rem; color: white; margin-bottom: .8rem;}
.risk-card h2 {color: white; margin: 0; font-size: 2.4rem;}
.risk-card p {margin: 0; opacity: .95;}
.hint {color: #6b6b6b; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model utilities
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text())


@st.cache_data
def load_enrolled():
    return pd.read_csv(ENROLLED_PATH)


@st.cache_resource
def reference_mean(_model, features):
    """Rata-rata vektor fitur (setelah preprocessing) data latih (Dropout & Graduate) sebagai acuan 'mahasiswa rata-rata'."""
    if not REFERENCE_DATA_PATH.exists():
        return None
    reference = pd.read_csv(REFERENCE_DATA_PATH, sep=";")
    reference = reference[reference["Status"].isin(["Dropout", "Graduate"])][features]
    return np.asarray(_model.named_steps["preprocessor"].transform(reference)).mean(axis=0)


def risk_level(probability):
    for threshold, label, color, icon in RISK_LEVELS:
        if probability >= threshold:
            return label, color, icon


def predict(model, features, data):
    return model.predict_proba(data[features])[:, 1]


def feature_contributions(model, features, row):
    """Kontribusi tiap fitur terhadap log-odds dropout (khusus model linier) relatif terhadap rata-rata."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return None
    preprocessor = model.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(row[features]))[0]
    baseline = reference_mean(model, features)
    if baseline is not None:
        transformed = transformed - baseline
    contrib = transformed * estimator.coef_[0]
    grouped = {}
    for name, value in zip(preprocessor.get_feature_names_out(), contrib):
        original = next(f for f in features if name.split("__", 1)[1].startswith(f))
        grouped[original] = grouped.get(original, 0.0) + value
    result = pd.DataFrame({"feature": list(grouped), "contribution": list(grouped.values())})
    result["label"] = result["feature"].map(FEATURE_LABELS)
    return result


def risk_signals(row):
    r = row.iloc[0]
    signals = []
    if r["Tuition_fees_up_to_date"] == 0:
        signals.append(("💳", "Biaya kuliah belum lunas", "Tawarkan skema cicilan atau konsultasi keuangan."))
    if r["Debtor"] == 1:
        signals.append(("📄", "Memiliki tunggakan", "Hubungkan dengan bagian keuangan untuk restrukturisasi pembayaran."))
    for sem, label in [("2nd", "semester 2"), ("1st", "semester 1")]:
        enrolled, approved = int(r[f"Curricular_units_{sem}_sem_enrolled"]), int(r[f"Curricular_units_{sem}_sem_approved"])
        if enrolled > 0 and approved / enrolled < 0.5:
            signals.append(("📉", f"Lulus < 50% mata kuliah di {label} ({approved}/{enrolled})",
                            "Jadwalkan mentoring akademik dan evaluasi beban studi."))
            break
    if r["Curricular_units_2nd_sem_grade"] < 10:
        signals.append(("📝", f"Rata-rata nilai semester 2 rendah ({r['Curricular_units_2nd_sem_grade']:.1f}/20)",
                        "Libatkan dosen wali dan tutor sebaya untuk mata kuliah yang tertinggal."))
    if r["Age_at_enrollment"] > 24:
        signals.append(("🧑‍💼", f"Mendaftar di usia {int(r['Age_at_enrollment'])} tahun",
                        "Tawarkan jadwal fleksibel / kelas hybrid bagi mahasiswa yang bekerja."))
    if r["Daytime_evening_attendance"] == 0:
        signals.append(("🌙", "Kuliah kelas malam", "Pastikan layanan akademik dan konseling tersedia di jam malam."))
    if r["Scholarship_holder"] == 0 and r["Tuition_fees_up_to_date"] == 0:
        signals.append(("🎓", "Tidak menerima beasiswa", "Informasikan peluang beasiswa / bantuan biaya pendidikan."))
    return signals


# ------------------------------------------------------------------ app
try:
    model = load_model()
    metadata = load_metadata()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Jalankan `notebook.ipynb` terlebih dahulu untuk menghasilkan `model/model.joblib`.")
    st.stop()
FEATURES = metadata["features"]
metrics = metadata["test_metrics"]

with st.sidebar:
    st.markdown("## 🎓 Jaya Jaya Institut")
    st.caption("Dropout Early Warning System")
    st.divider()
    st.markdown("**Model**")
    st.write(metadata["model_name"])
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{metrics['Recall']:.0%}")
    c2.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")
    c1.metric("F1-score", f"{metrics['F1']:.2f}")
    c2.metric("Akurasi", f"{metrics['Accuracy']:.0%}")
    st.divider()
    st.markdown("**Level risiko**")
    for threshold, label, _, icon in RISK_LEVELS:
        upper = {"Tinggi": "≥ 50%", "Sedang": "30% – 49%", "Rendah": "< 30%"}[label]
        st.markdown(f"{icon} **{label}** — probabilitas {upper}")
    st.caption("Mahasiswa dengan level **Tinggi** diprediksi *Dropout* dan diprioritaskan untuk bimbingan khusus.")

st.title("🎓 Dropout Early Warning System")
st.markdown("Prototype untuk membantu staf akademik **Jaya Jaya Institut** mendeteksi mahasiswa yang berisiko "
            "*dropout* sedini mungkin berdasarkan data pendaftaran, finansial, dan performa semester 1–2. "
            "Model dilatih dari mahasiswa yang status akhirnya sudah diketahui (**Dropout** vs **Graduate**) "
            "dan dipakai untuk memprediksi mahasiswa yang masih aktif (**Enrolled**).")

tab_single, tab_batch, tab_about = st.tabs(["🧑‍🎓 Prediksi Individu", "📂 Prediksi Batch (CSV)", "ℹ️ Tentang Model"])

# ------------------------------------------------------------------ single prediction
with tab_single:
    def apply_preset(name):
        for key, value in PRESETS[name].items():
            st.session_state[key] = value

    for key, value in PRESETS["low"].items():
        st.session_state.setdefault(key, value)

    p1, p2, _ = st.columns([1.3, 1.3, 3])
    p1.button("Isi contoh risiko tinggi", on_click=apply_preset, args=("high",), width="stretch")
    p2.button("Isi contoh risiko rendah", on_click=apply_preset, args=("low",), width="stretch")

    with st.form("student_form"):
        st.subheader("1. Data pendaftaran")
        c1, c2, c3 = st.columns(3)
        course = c1.selectbox("Program studi", sorted(COURSE_MAP, key=COURSE_MAP.get),
                              format_func=COURSE_MAP.get, key="Course")
        application_mode = c2.selectbox("Jalur pendaftaran", list(APPLICATION_MODE_MAP),
                                        format_func=APPLICATION_MODE_MAP.get, key="Application_mode")
        attendance = c3.radio("Waktu kuliah", ["Siang", "Malam"], horizontal=True, key="attendance")
        c1, c2, c3 = st.columns(3)
        prev_grade = c1.number_input("Nilai pendidikan sebelumnya (0–200)", 0.0, 200.0, step=1.0,
                                     key="Previous_qualification_grade")
        admission_grade = c2.number_input("Nilai masuk (0–200)", 0.0, 200.0, step=1.0, key="Admission_grade")
        age = c3.number_input("Usia saat mendaftar", 15, 80, step=1, key="Age_at_enrollment")

        st.subheader("2. Data pribadi & finansial")
        c1, c2, c3, c4, c5 = st.columns(5)
        gender = c1.radio("Gender", ["Perempuan", "Laki-laki"], key="gender")
        displaced = c2.radio("Mahasiswa perantau?", ["Tidak", "Ya"], key="displaced")
        scholarship = c3.radio("Penerima beasiswa?", ["Tidak", "Ya"], key="scholarship")
        debtor = c4.radio("Memiliki tunggakan?", ["Tidak", "Ya"], key="debtor")
        tuition = c5.radio("Biaya kuliah lunas?", ["Tidak", "Ya"], key="tuition")

        st.subheader("3. Performa akademik")
        semesters = {}
        for sem, prefix, title in [("1st", "s1", "Semester 1"), ("2nd", "s2", "Semester 2")]:
            st.markdown(f"**{title}**")
            c1, c2, c3, c4 = st.columns(4)
            semesters[sem] = {
                "enrolled": c1.number_input("Mata kuliah diambil", 0, 30, step=1, key=f"{prefix}_enrolled"),
                "evaluations": c2.number_input("Jumlah evaluasi/ujian", 0, 50, step=1, key=f"{prefix}_evaluations"),
                "approved": c3.number_input("Mata kuliah lulus", 0, 30, step=1, key=f"{prefix}_approved"),
                "grade": c4.number_input("Rata-rata nilai (0–20)", 0.0, 20.0, step=0.1, key=f"{prefix}_grade"),
            }
        submitted = st.form_submit_button("🔍 Prediksi risiko dropout", type="primary", width="stretch")

    if submitted:
        invalid = [title for sem, title in [("1st", "semester 1"), ("2nd", "semester 2")]
                   if semesters[sem]["approved"] > semesters[sem]["enrolled"]]
        if invalid:
            st.error(f"Jumlah mata kuliah lulus tidak boleh melebihi mata kuliah yang diambil ({', '.join(invalid)}).")
            st.stop()

        row = pd.DataFrame([{
            "Application_mode": application_mode, "Course": course,
            "Daytime_evening_attendance": 1 if attendance == "Siang" else 0,
            "Previous_qualification_grade": prev_grade, "Admission_grade": admission_grade,
            "Displaced": YES_NO[displaced], "Debtor": YES_NO[debtor], "Tuition_fees_up_to_date": YES_NO[tuition],
            "Gender": 1 if gender == "Laki-laki" else 0, "Scholarship_holder": YES_NO[scholarship],
            "Age_at_enrollment": age,
            **{f"Curricular_units_{sem}_sem_{k}": v for sem, vals in semesters.items() for k, v in vals.items()},
        }])[FEATURES]

        probability = float(predict(model, FEATURES, row)[0])
        label, color, icon = risk_level(probability)
        verdict = "Diprediksi DROPOUT" if probability >= metadata["threshold"] else "Diprediksi GRADUATE (lulus)"

        st.divider()
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.markdown(f"""
<div class="risk-card" style="background:{color}">
  <p>Probabilitas dropout</p>
  <h2>{probability:.0%}</h2>
  <p>{icon} Risiko <b>{label}</b> · {verdict}</p>
</div>""", unsafe_allow_html=True)
            st.progress(probability)
            signals = risk_signals(row)
            if label == "Tinggi":
                st.markdown("**Rekomendasi:** prioritaskan mahasiswa ini untuk bimbingan khusus dalam 1–2 minggu.")
            elif label == "Sedang":
                st.markdown("**Rekomendasi:** pantau secara berkala dan jadwalkan konsultasi dengan dosen wali.")
            else:
                st.markdown("**Rekomendasi:** tidak perlu intervensi khusus; lanjutkan pemantauan rutin.")
            if signals:
                st.markdown("**Sinyal risiko yang terdeteksi:**")
                for emoji, text, action in signals:
                    st.markdown(f"- {emoji} **{text}** — {action}")

        with right:
            contributions = feature_contributions(model, FEATURES, row)
            if contributions is not None:
                top = contributions.reindex(contributions["contribution"].abs().sort_values(ascending=False).index).head(8)
                top["arah"] = np.where(top["contribution"] > 0, "Menaikkan risiko", "Menurunkan risiko")
                st.markdown("**Faktor yang paling memengaruhi prediksi**")
                chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("contribution:Q", title="Kontribusi terhadap risiko (log-odds, relatif thd. rata-rata)"),
                    y=alt.Y("label:N", sort=alt.EncodingSortField("contribution", order="descending"), title=None,
                            scale=alt.Scale(paddingInner=0.35), axis=alt.Axis(labelLimit=220)),
                    color=alt.Color("arah:N", scale=alt.Scale(domain=["Menaikkan risiko", "Menurunkan risiko"],
                                                              range=["#eb6834", "#2a78d6"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("label:N", title="Fitur"),
                             alt.Tooltip("contribution:Q", title="Kontribusi", format=".2f")],
                )
                st.altair_chart(chart, width="stretch", height=340)
                st.caption("Batang oranye mendorong risiko dropout naik dibanding mahasiswa rata-rata; "
                           "batang biru menurunkannya.")

# ------------------------------------------------------------------ batch prediction
with tab_batch:
    st.markdown("Prediksi risiko dropout untuk banyak mahasiswa sekaligus — misalnya seluruh mahasiswa **Enrolled** "
                "(masih aktif) yang status akhirnya belum diketahui. Unggah file CSV yang memuat kolom berikut "
                "(kolom lain akan tetap dipertahankan):")
    st.code(", ".join(FEATURES), language=None)
    c1, c2 = st.columns([1, 1])
    c1.download_button("⬇️ Unduh template / data mahasiswa Enrolled (CSV)", load_enrolled().to_csv(index=False),
                       file_name="data_mahasiswa_enrolled.csv", mime="text/csv", width="stretch")
    use_sample = c2.toggle(f"Gunakan data mahasiswa Enrolled ({len(load_enrolled())} mahasiswa aktif)")
    uploaded = st.file_uploader("Unggah CSV", type=["csv"])

    batch = None
    if uploaded is not None:
        batch = pd.read_csv(uploaded, sep=None, engine="python")
    elif use_sample:
        batch = load_enrolled()

    if batch is not None:
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing)}")
        else:
            result = batch.copy()
            result["Probabilitas_Dropout"] = predict(model, FEATURES, batch)
            result["Level_Risiko"] = result["Probabilitas_Dropout"].map(lambda p: risk_level(p)[0])
            result["Prediksi"] = np.where(result["Probabilitas_Dropout"] >= metadata["threshold"], "Dropout",
                                          "Graduate")
            result.insert(0, "Program_Studi", result["Course"].map(COURSE_MAP))
            result = result.sort_values("Probabilitas_Dropout", ascending=False).reset_index(drop=True)

            counts = result["Level_Risiko"].value_counts().reindex(["Tinggi", "Sedang", "Rendah"], fill_value=0)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total mahasiswa", len(result))
            for col, level, icon in [(m2, "Tinggi", "🔴"), (m3, "Sedang", "🟠"), (m4, "Rendah", "🟢")]:
                col.metric(f"{icon} Risiko {level.lower()}", f"{int(counts[level])} ({counts[level] / len(result):.0%})")

            chart_data = counts.rename_axis("Level").reset_index(name="Jumlah")
            st.altair_chart(alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Jumlah:Q", title="Jumlah mahasiswa"),
                y=alt.Y("Level:N", sort=["Tinggi", "Sedang", "Rendah"], title=None, scale=alt.Scale(paddingInner=0.3)),
                color=alt.Color("Level:N", scale=alt.Scale(domain=[r[1] for r in RISK_LEVELS],
                                                           range=[r[2] for r in RISK_LEVELS]), legend=None),
                tooltip=["Level", "Jumlah"],
            ), width="stretch", height=180)

            display_cols = ["Program_Studi", "Probabilitas_Dropout", "Level_Risiko", "Prediksi", "Age_at_enrollment",
                            "Tuition_fees_up_to_date", "Debtor", "Scholarship_holder",
                            "Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved",
                            "Curricular_units_2nd_sem_grade"]
            st.dataframe(result[display_cols + [c for c in result.columns if c not in display_cols]],
                         width="stretch", hide_index=True, column_config={
                             "Probabilitas_Dropout": st.column_config.ProgressColumn(
                                 "Probabilitas dropout", format="percent", min_value=0.0, max_value=1.0),
                             "Program_Studi": "Program studi", "Level_Risiko": "Level risiko",
                         })
            st.download_button("⬇️ Unduh hasil prediksi", result.to_csv(index=False),
                               file_name="hasil_prediksi_dropout.csv", mime="text/csv", type="primary")

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("Tentang model")
    st.markdown(f"""
- **Algoritma:** {metadata['model_name']} (parameter terbaik: `{metadata['best_params']}`), dibungkus dalam
  pipeline `OneHotEncoder` + `StandardScaler`.
- **Target:** `Dropout` (1) vs `Graduate` (0). Threshold prediksi {metadata['threshold']:.0%}.
- **Data:** hanya mahasiswa dengan status akhir yang sudah diketahui (Dropout & Graduate):
  {metadata['n_train']:,} data latih & {metadata['n_test']:,} data uji (split 80:20 stratified).
- **Mahasiswa Enrolled** ({metadata['n_enrolled']:,}) tidak dipakai untuk melatih model karena status akhirnya belum
  diketahui — mereka adalah data yang **diprediksi**: {metadata['enrolled_risk_counts']['Tinggi']} berisiko tinggi,
  {metadata['enrolled_risk_counts']['Sedang']} sedang, {metadata['enrolled_risk_counts']['Rendah']} rendah.
- **Fitur:** {len(FEATURES)} fitur pendaftaran, finansial, dan akademik semester 1–2.
""")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, key in zip([c1, c2, c3, c4, c5], ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        col.metric(key, f"{metrics[key]:.3f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Perbandingan model (data uji)**")
        st.dataframe(pd.DataFrame(metadata["model_comparison"]).set_index("Model").round(3),
                     width="stretch")
        cm = metadata["confusion_matrix"]
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                  index=["Aktual Graduate", "Aktual Dropout"],
                                  columns=["Prediksi Graduate", "Prediksi Dropout"]), width="stretch")
    with right:
        importance = pd.DataFrame({"feature": list(metadata["top_features"]),
                                   "importance": list(metadata["top_features"].values())})
        importance["label"] = importance["feature"].map(FEATURE_LABELS)
        st.markdown("**Fitur paling berpengaruh (permutation importance)**")
        st.altair_chart(alt.Chart(importance).mark_bar(cornerRadiusEnd=4, color="#2a78d6").encode(
            x=alt.X("importance:Q", title="Penurunan F1 saat fitur diacak"),
            y=alt.Y("label:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.3),
                    axis=alt.Axis(labelLimit=220)),
            tooltip=[alt.Tooltip("label:N", title="Fitur"), alt.Tooltip("importance:Q", format=".3f")],
        ), width="stretch", height=360)
