import io
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import libsql_client
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).resolve().parent
IMAGE_DIR = APP_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Teemill Quality Intelligence Trial", layout="wide")


def db():
    url = st.secrets["TURSO_DATABASE_URL"]
    token = st.secrets["TURSO_AUTH_TOKEN"]
    conn = libsql_client.create_client_sync(url=url, auth_token=token)
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            order_id TEXT,
            sku TEXT,
            garment_colour TEXT,
            printer_id TEXT,
            operator TEXT,
            shift TEXT,
            print_profile TEXT,
            defect_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            decision TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            corrective_action TEXT,
            notes TEXT,
            feature_json TEXT NOT NULL,
            verified INTEGER DEFAULT 1
        )
        """
    )
    return conn


def image_features(image: Image.Image) -> np.ndarray:
    """Lightweight visual fingerprint for trial use; not production-grade CV."""
    img = ImageOps.exif_transpose(image).convert("RGB").resize((96, 96))
    arr = np.asarray(img).astype(np.float32) / 255.0

    # RGB histograms
    feats = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
        feats.extend(hist.tolist())

    # Brightness and contrast blocks
    gray = arr.mean(axis=2)
    for rows in np.array_split(gray, 4, axis=0):
        for block in np.array_split(rows, 4, axis=1):
            feats.extend([float(block.mean()), float(block.std())])

    # Edge proxy
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    feats.extend([float(gray.mean()), float(gray.std()), float(gx), float(gy)])
    return np.array(feats, dtype=np.float32)


def save_image(uploaded_file) -> tuple[str, Image.Image]:
    image = Image.open(uploaded_file).convert("RGB")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{ts}.jpg"
    path = IMAGE_DIR / safe_name
    image.save(path, quality=92)
    return str(path), image


def load_data() -> pd.DataFrame:
    conn = db()
    result = conn.execute("SELECT * FROM inspections ORDER BY id DESC")
    rows = result.rows
    columns = result.columns
    conn.close()
    
    if not rows:
        return pd.DataFrame(columns=[
            "id", "created_at", "image_path", "order_id", "sku", "garment_colour",
            "printer_id", "operator", "shift", "print_profile", "defect_type",
            "severity", "decision", "root_cause", "corrective_action", "notes",
            "feature_json", "verified"
        ])
    return pd.DataFrame(rows, columns=columns)


def train_models(df: pd.DataFrame):
    if df.empty:
        return None
    verified = df[df["verified"] == 1].copy()
    if len(verified) < 8:
        return None
    X = np.vstack(verified["feature_json"].apply(lambda x: np.array(json.loads(x), dtype=np.float32)))

    models = {}
    for target in ["decision", "defect_type", "root_cause", "severity"]:
        if verified[target].nunique() < 2:
            continue
        enc = LabelEncoder()
        y = enc.fit_transform(verified[target].astype(str))
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X, y)
        models[target] = (model, enc)
    return {"X": X, "df": verified.reset_index(drop=True), "models": models}


def predict_bundle(image: Image.Image, trained):
    feat = image_features(image)
    if not trained:
        return feat, {}, []

    preds = {}
    for target, (model, enc) in trained["models"].items():
        probs = model.predict_proba([feat])[0]
        idx = int(np.argmax(probs))
        preds[target] = {
            "value": enc.inverse_transform([idx])[0],
            "confidence": float(probs[idx]),
        }

    similarities = cosine_similarity([feat], trained["X"])[0]
    top_idx = np.argsort(similarities)[::-1][:5]
    examples = []
    for idx in top_idx:
        row = trained["df"].iloc[idx]
        examples.append({
            "similarity": float(similarities[idx]),
            "decision": row["decision"],
            "defect_type": row["defect_type"],
            "root_cause": row["root_cause"],
            "severity": row["severity"],
            "image_path": row["image_path"],
        })
    return feat, preds, examples


DEFECTS = [
    "No defect", "Masking / white halo", "Stain", "Wonky platen / positioning",
    "Fibrous print", "Dull white", "Pretreat stain", "Stripes / banding",
    "Smudge", "Wrong print", "Insufficient pretreat", "No pretreat",
    "Bleeding", "Printed on wrong side", "Other"
]
ROOT_CAUSES = [
    "No issue", "Registration drift", "Incorrect choke/spread profile",
    "Platen movement", "Garment loading", "Garment movement during print",
    "Nozzle blockage", "White ink circulation", "Pretreat dosage",
    "Pretreat nozzle alignment", "Dirty workstation / contamination",
    "Incorrect machine setting", "Wrong artwork / scan process",
    "Maintenance overdue", "Operator training gap", "Unknown / investigation required"
]
ACTIONS = [
    "Ship", "Supervisor review", "Reprint", "Run registration calibration",
    "Check platen seating/alignment", "Review print profile", "Run nozzle check/clean",
    "Check white ink circulation", "Calibrate pretreat dosage", "Clean pretreat nozzles",
    "Clean workstation and platen", "Retrain operator", "Investigate machine history",
    "Quarantine batch"
]

st.title("Teemill Quality Intelligence — Trial")
st.caption("Human-in-the-loop quality learning, reprint approval and root-cause capture")

page = st.sidebar.radio("Go to", ["Add training example", "Check a print", "Standards dashboard", "Data export"])
df = load_data()
trained = train_models(df)

with st.sidebar:
    st.metric("Verified examples", int((df["verified"] == 1).sum()) if not df.empty else 0)
    st.metric("Minimum useful dataset", "50+ per key defect")
    if trained:
        st.success("Learning model active")
    else:
        st.info("Add at least 8 varied examples to activate trial predictions")

if page == "Add training example":
    st.header("Add a labelled quality example")
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("Upload print photo", type=["jpg", "jpeg", "png"], key="train")
        if uploaded:
            preview = Image.open(uploaded)
            st.image(preview, caption="Training image", use_container_width=True)
    with right:
        order_id = st.text_input("Order ID")
        sku = st.text_input("Garment SKU")
        garment_colour = st.text_input("Garment colour")
        printer_id = st.text_input("Printer ID")
        operator = st.text_input("Operator")
        shift = st.selectbox("Shift", ["Day", "Night", "Weekend", "Other"])
        print_profile = st.text_input("Print profile")
        defect = st.selectbox("Observed defect", DEFECTS)
        severity = st.selectbox("Severity", ["Acceptable", "Borderline", "Reject"])
        decision = st.selectbox("Final decision", ["Ship", "Review", "Reprint"])
        root_cause = st.selectbox("Confirmed or suspected root cause", ROOT_CAUSES)
        corrective_action = st.selectbox("Corrective action", ACTIONS)
        notes = st.text_area("Notes / 5 Whys evidence")

        if st.button("Save verified example", type="primary", disabled=uploaded is None):
            uploaded.seek(0)
            path, image = save_image(uploaded)
            feat = image_features(image)
            conn = db()
            conn.execute(
                """INSERT INTO inspections
                (created_at,image_path,order_id,sku,garment_colour,printer_id,operator,shift,print_profile,
                 defect_type,severity,decision,root_cause,corrective_action,notes,feature_json,verified)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (
                    datetime.utcnow().isoformat(), path, order_id, sku, garment_colour, printer_id,
                    operator, shift, print_profile, defect, severity, decision, root_cause,
                    corrective_action, notes, json.dumps(feat.tolist())
                )
            )
            conn.close()
            st.success("Example saved and added to the learning dataset.")
            st.rerun()

elif page == "Check a print":
    st.header("AI-assisted reprint check")
    uploaded = st.file_uploader("Upload suspected defect photo", type=["jpg", "jpeg", "png"], key="check")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        feat, preds, examples = predict_bundle(image, trained)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.image(image, caption="Print under review", use_container_width=True)
        with c2:
            if not preds:
                st.warning("Not enough labelled examples yet. Add more standards first.")
            else:
                st.subheader("Suggested assessment")
                for target, label in [
                    ("decision", "Decision"), ("defect_type", "Defect"),
                    ("severity", "Severity"), ("root_cause", "Root cause")
                ]:
                    if target in preds:
                        st.metric(label, preds[target]["value"], f"{preds[target]['confidence']:.0%} confidence")
                if preds.get("decision", {}).get("confidence", 0) < 0.75:
                    st.warning("Low confidence: mandatory supervisor review")

        st.subheader("Most similar verified standards")
        if examples:
            cols = st.columns(min(5, len(examples)))
            for col, ex in zip(cols, examples):
                with col:
                    try:
                        st.image(ex["image_path"], use_container_width=True)
                    except Exception:
                        pass
                    st.caption(f"{ex['similarity']:.0%} similar")
                    st.write(f"**{ex['decision']}**")
                    st.write(ex["defect_type"])
                    st.write(ex["root_cause"])

        st.divider()
        st.subheader("Human confirmation")
        cc1, cc2 = st.columns(2)
        with cc1:
            final_decision = st.selectbox("Final decision", ["Ship", "Review", "Reprint"], key="fd")
            final_defect = st.selectbox("Final defect", DEFECTS, key="fdef")
            final_severity = st.selectbox("Final severity", ["Acceptable", "Borderline", "Reject"], key="fsev")
        with cc2:
            final_root = st.selectbox("Final root cause", ROOT_CAUSES, key="frc")
            action = st.selectbox("Action", ACTIONS, key="fact")
            notes = st.text_area("Evidence / investigation notes", key="fnotes")
        meta1, meta2, meta3 = st.columns(3)
        with meta1: printer_id = st.text_input("Printer ID", key="cp")
        with meta2: sku = st.text_input("SKU", key="cs")
        with meta3: print_profile = st.text_input("Print profile", key="cpp")

        if st.button("Confirm and add to learning", type="primary"):
            buf = io.BytesIO(); image.save(buf, format="JPEG", quality=92); buf.seek(0)
            class UploadWrap:
                def __init__(self, b): self.b = b
                def read(self, *a): return self.b.read(*a)
                def seek(self, *a): return self.b.seek(*a)
            path, _ = save_image(UploadWrap(buf))
            conn = db()
            conn.execute(
                """INSERT INTO inspections
                (created_at,image_path,sku,printer_id,print_profile,defect_type,severity,decision,
                 root_cause,corrective_action,notes,feature_json,verified)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                (datetime.utcnow().isoformat(), path, sku, printer_id, print_profile,
                 final_defect, final_severity, final_decision, final_root, action, notes,
                 json.dumps(feat.tolist()))
            )
            conn.close()
            st.success("Decision confirmed and used as a new training example.")
            st.rerun()

elif page == "Standards dashboard":
    st.header("Quality standards and root-cause dashboard")
    if df.empty:
        st.info("No examples saved yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Examples", len(df))
        m2.metric("Reprint rate", f"{(df['decision'].eq('Reprint').mean() * 100):.1f}%")
        m3.metric("Ship rate", f"{(df['decision'].eq('Ship').mean() * 100):.1f}%")
        m4.metric("Root causes captured", df["root_cause"].nunique())

        st.subheader("Defect Pareto")
        defect_counts = df["defect_type"].value_counts().rename_axis("Defect").reset_index(name="Count")
        st.bar_chart(defect_counts.set_index("Defect"))

        st.subheader("Root-cause Pareto")
        root_counts = df["root_cause"].value_counts().rename_axis("Root cause").reset_index(name="Count")
        st.bar_chart(root_counts.set_index("Root cause"))

        st.subheader("Recent labelled standards")
        st.dataframe(df[["created_at","printer_id","sku","defect_type","severity","decision","root_cause","corrective_action","notes"]], use_container_width=True)

elif page == "Data export":
    st.header("Export / reset trial data")
    if df.empty:
        st.info("No data available.")
    else:
        csv = df.drop(columns=["feature_json"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download inspection data CSV", csv, "tqi_inspections.csv", "text/csv")
        st.dataframe(df.drop(columns=["feature_json"]), use_container_width=True)

    st.warning("Reset clears stored images.")
    confirm = st.checkbox("I understand")
    if st.button("Reset all trial data", disabled=not confirm):
        for p in IMAGE_DIR.glob("*"):
            p.unlink()
        st.success("Local images cleared.")
        st.rerun()
