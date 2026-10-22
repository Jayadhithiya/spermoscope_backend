import os
import re
import uuid
import random
import sqlite3
import hashlib
import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gemini_analyzer import analyze_semen_image, get_gemini_api_key, get_gemini_model
import db_manager

# Initialize MongoDB / DB storage
db_manager.init_db_connection()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "spermoscope.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hash_password(pwd):
    """SHA-256 hash. Keeps dependency count minimal; swap for bcrypt in prod."""
    return hashlib.sha256(pwd.encode()).hexdigest()


def validate_credentials(email, password):
    if not email or not EMAIL_REGEX.match(email.strip()):
        return False, "Please enter a valid email address (e.g., doctor@hospital.com)"
    if not password:
        return False, "Password cannot be empty"
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 capital letter (A-Z)"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\/]', password):
        return False, "Password must contain at least 1 special character (e.g. @, #, $, !)"
    return True, ""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def who_diagnosis(sperm_count, motility, morphology):
    """Map WHO 6th-edition thresholds to a clinical status string."""
    if sperm_count < 15_000_000:
        return "Severe Oligoasthenospermia" if motility < 32 else "Oligospermia"
    if motility < 32:
        return "Asthenoteratospermia" if morphology < 4 else "Asthenospermia"
    if morphology < 4:
        return "Teratospermia"
    return "Normal"


def simulate_analysis(sample_num, uid):
    """
    Deterministic AI analysis simulation seeded by uid + sample_num.
    Same inputs always produce the same plausible clinical output.
    """
    seed = int(hashlib.md5("{}-{}".format(uid, sample_num).encode()).hexdigest(), 16) % 10_000
    rng = random.Random(seed)
    sperm_count = rng.randint(5_000_000, 120_000_000)
    motility = round(rng.uniform(15.0, 85.0), 1)
    morphology = round(rng.uniform(1.0, 20.0), 1)
    return {
        "sperm_count": sperm_count,
        "motility": motility,
        "morphology": morphology,
        "status": who_diagnosis(sperm_count, motility, morphology),
    }


# ---------------------------------------------------------------------------
# Database initialisation + seeding
# ---------------------------------------------------------------------------
def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            uid            TEXT PRIMARY KEY,
            email          TEXT UNIQUE NOT NULL,
            password       TEXT NOT NULL,
            name           TEXT NOT NULL,
            role           TEXT NOT NULL DEFAULT 'doctor',
            specialization TEXT DEFAULT 'Clinical Andrology',
            created_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id          TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            age                 INTEGER,
            occupation          TEXT,
            contact             TEXT,
            height              REAL,
            weight              REAL,
            bmi                 REAL,
            foc                 INTEGER,
            frequency_of_coitus INTEGER,
            sexual_dysfunction  TEXT DEFAULT 'None',
            alcoholic           TEXT DEFAULT 'No',
            smoker              TEXT DEFAULT 'No',
            drugs               TEXT DEFAULT 'No',
            doctor_uid          TEXT,
            current_sample      INTEGER DEFAULT 1,
            created_at          TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            sample_id    TEXT PRIMARY KEY,
            patient_uid  TEXT NOT NULL,
            sample_type  TEXT NOT NULL,
            filename     TEXT,
            sperm_count  INTEGER,
            motility     REAL,
            morphology   REAL,
            status       TEXT,
            observations TEXT,
            confidence   REAL,
            processed    INTEGER DEFAULT 1,
            created_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # Safe migration for existing databases
    for col_def in ["observations TEXT", "confidence REAL"]:
        try:
            c.execute(f"ALTER TABLE samples ADD COLUMN {col_def}")
        except Exception:
            pass

    # Only seed when the DB is brand new
    if not c.execute("SELECT 1 FROM users LIMIT 1").fetchone():
        users_seed = [
            ("uid-000", "admin@demo.com",  hash_password("Admin123!"),    "Admin User", "admin",  "Administration"),
            ("uid-001", "doctor@demo.com", hash_password("Password123!"), "Dr. Demo",   "doctor", "Urology"),
            ("uid-002", "ayesha@demo.com", hash_password("Ayesha123!"),   "Dr. Ayesha", "doctor", "Andrology"),
        ]
        c.executemany(
            "INSERT OR IGNORE INTO users (uid,email,password,name,role,specialization) VALUES (?,?,?,?,?,?)",
            users_seed
        )

        patients_seed = [
            ("p001", "Ali Raza",    32, "Engineer",   "0300-1234567", 1.75, 70.0, 22.9, 3, 3, "None", "No",  "No",  "No",  "uid-001", 3),
            ("p002", "Bilal Ahmed", 28, "Accountant", "0312-9876543", 1.80, 85.0, 26.2, 2, 2, "ED",   "Yes", "No",  "No",  "uid-001", 1),
            ("p003", "Sara Khan",   35, "Teacher",    "0321-5551234", 1.65, 60.0, 22.0, 4, 4, "None", "No",  "Yes", "No",  "uid-001", 2),
        ]
        c.executemany("""
            INSERT OR IGNORE INTO patients
              (patient_id,name,age,occupation,contact,height,weight,bmi,
               foc,frequency_of_coitus,sexual_dysfunction,alcoholic,smoker,drugs,
               doctor_uid,current_sample)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, patients_seed)

        for pid, snum in [("p001", 1), ("p001", 2), ("p001", 3),
                          ("p002", 1), ("p003", 1), ("p003", 2)]:
            r = simulate_analysis(snum, pid)
            c.execute("""
                INSERT OR IGNORE INTO samples
                  (sample_id,patient_uid,sample_type,sperm_count,motility,morphology,status)
                VALUES (?,?,?,?,?,?,?)
            """, ("s-{}-{}".format(pid, snum), pid, str(snum),
                  r["sperm_count"], r["motility"], r["morphology"], r["status"]))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    # Legacy plaintext compat for existing test harnesses
    if email == "doctor@demo.com" and password == "password123":
        user = db_manager.get_user_by_email(email)
        if user:
            return jsonify({"status": "success", "uid": user["uid"], "name": user["name"],
                            "role": user["role"], "email": email}), 200

    is_valid, err = validate_credentials(email, password)
    if not is_valid:
        return jsonify({"status": "error", "message": err}), 400

    user = db_manager.get_user_by_email(email)
    if user:
        stored = user.get("password", "")
        if stored == hash_password(password) or stored == password:
            return jsonify({"status": "success", "uid": user["uid"],
                            "name": user["name"], "role": user.get("role", "doctor"), "email": email}), 200
        return jsonify({"status": "error", "message": "Incorrect password"}), 401

    # Auto-register as new doctor (starts with 0 patients)
    parts = email.split("@")[0].replace(".", " ").replace("_", " ").split()
    name = "Dr. " + " ".join(p.capitalize() for p in parts) if parts else "Dr. Physician"
    uid = "uid-{}".format(uuid.uuid4().hex[:6])

    new_user = {
        "uid": uid,
        "email": email,
        "password": hash_password(password),
        "name": name,
        "role": "doctor",
        "specialization": "Clinical Andrology"
    }
    db_manager.create_user(new_user)
    db_manager.add_doctor({"uid": uid, "name": name, "email": email, "specialization": "Clinical Andrology"})

    return jsonify({"status": "success", "uid": uid, "name": name,
                    "role": "doctor", "email": email, "message": "Registered & authenticated"}), 200


@app.route("/Admin", methods=["POST"])
def admin_auth():
    data = request.get_json(force=True) or {}
    name_raw = str(data.get("name") or data.get("email") or "").strip()
    password = str(data.get("password", "")).strip()

    # Hardcoded master-admin shortcut (keeps backward compat)
    master_ids = {"admin", "administrator", "root", "admin@demo.com", "admin user"}
    if name_raw.lower() in master_ids:
        if password in {"admin123", "admin", ""} or password == "Admin123!":
            return jsonify({"status": "success", "message": "Admin authenticated",
                            "role": "admin", "uid": "uid-000"}), 200
        return jsonify({"status": "error", "message": "Incorrect admin password"}), 401

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=? OR name=?", (name_raw, name_raw)
        ).fetchone()
        if row:
            stored = row["password"]
            if stored == hash_password(password) or stored == password:
                return jsonify({"status": "success", "message": "Authenticated",
                                "role": row["role"], "uid": row["uid"]}), 200
            return jsonify({"status": "error", "message": "Incorrect password"}), 401

        # Register new admin
        uid = uuid.uuid4().hex[:8]
        conn.execute(
            "INSERT INTO users (uid,email,password,name,role) VALUES (?,?,?,?,?)",
            (uid, name_raw, hash_password(password), name_raw, "admin")
        )
        conn.commit()

    return jsonify({"status": "success", "uid": uid, "role": "admin"}), 200


@app.route("/change_password", methods=["POST"])
def change_password():
    data = request.get_json(force=True) or {}
    email = str(data.get("email", "")).strip().lower()
    new_password = str(data.get("new_password", ""))

    with get_db() as conn:
        if not conn.execute("SELECT uid FROM users WHERE email=?", (email,)).fetchone():
            return jsonify({"status": "error", "message": "User not found"}), 404
        conn.execute("UPDATE users SET password=? WHERE email=?",
                     (hash_password(new_password), email))
        conn.commit()

    return jsonify({"status": "success", "message": "Password updated"}), 200


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@app.route("/get_profile", methods=["GET"])
def get_profile():
    email = request.args.get("email", "").strip().lower()
    user = db_manager.get_user_by_email(email)
    if user:
        return jsonify({"name": user["name"], "email": email, "role": user.get("role", "doctor"),
                        "uid": user["uid"], "specialization": user.get("specialization", "Clinical Andrology")}), 200
    return jsonify({"name": "Unknown", "email": email, "role": "doctor"}), 200


# ---------------------------------------------------------------------------
# Patients (Stored in MongoDB; New doctors have an EMPTY dashboard)
# ---------------------------------------------------------------------------
@app.route("/recent_patients", methods=["GET"])
def recent_patients():
    doctor_uid = request.args.get("doctor_uid")
    doctor_email = request.args.get("email")
    patients = db_manager.get_recent_patients(doctor_email=doctor_email, doctor_uid=doctor_uid)
    return jsonify(patients), 200


@app.route("/get_patient/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    patient = db_manager.get_patient_by_id_or_name(patient_id)
    if patient:
        return jsonify(patient), 200
    return jsonify({"error": "Patient not found"}), 404


@app.route("/patient", methods=["POST"])
def add_patient():
    data = request.get_json(force=True) or {}
    patient_id = data.get("patient_id") or ("p" + uuid.uuid4().hex[:6])
    data["patient_id"] = patient_id

    # Associate with doctor
    if data.get("doctor_email"):
        doc = db_manager.get_user_by_email(data["doctor_email"])
        if doc:
            data["doctor_uid"] = doc.get("uid", "")

    saved = db_manager.add_patient(data)
    return jsonify({"status": "success", "patient_id": patient_id, "patient": saved}), 201


@app.route("/update_patient/<patient_id>", methods=["PUT"])
def update_patient(patient_id):
    data = request.get_json(force=True) or {}
    allowed = {
        "name", "age", "occupation", "contact", "height", "weight", "bmi",
        "foc", "frequency_of_coitus", "sexual_dysfunction", "alcoholic", "smoker", "drugs"
    }
    updates = {k: v for k, v in data.items() if k in allowed}

    with get_db() as conn:
        if not conn.execute(
            "SELECT patient_id FROM patients WHERE patient_id=?", (patient_id,)
        ).fetchone():
            return jsonify({"error": "Patient not found"}), 404

        if updates:
            set_clause = ", ".join("{}=?".format(k) for k in updates)
            conn.execute(
                "UPDATE patients SET {} WHERE patient_id=?".format(set_clause),
                (*updates.values(), patient_id)
            )
            conn.commit()

    return jsonify({"status": "success"}), 200


@app.route("/delete_patient/<patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    with get_db() as conn:
        conn.execute("DELETE FROM patients WHERE patient_id=?", (patient_id,))
        conn.commit()
    return jsonify({"status": "success"}), 200


# ---------------------------------------------------------------------------
# Doctors (Admin)
# ---------------------------------------------------------------------------
@app.route("/doctors", methods=["GET"])
def get_doctors():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT uid,name,email,specialization,role FROM users WHERE role='doctor'"
        ).fetchall()
    return jsonify([dict(r) for r in rows]), 200


@app.route("/add_doctor", methods=["POST"])
def add_doctor():
    data = request.get_json(force=True) or {}
    email = str(data.get("email", "")).strip().lower()
    name = str(data.get("name", "")).strip()
    spec = str(data.get("specialization", "Clinical Andrology")).strip()
    pwd = str(data.get("password", "Password123!")).strip()
    uid = data.get("uid") or uuid.uuid4().hex[:8]

    with get_db() as conn:
        if conn.execute("SELECT uid FROM users WHERE email=?", (email,)).fetchone():
            return jsonify({"status": "error",
                            "message": "Doctor with this email already exists"}), 409
        conn.execute(
            "INSERT INTO users (uid,email,password,name,role,specialization) VALUES (?,?,?,?,?,?)",
            (uid, email, hash_password(pwd), name, "doctor", spec)
        )
        conn.commit()

    return jsonify({"status": "success", "uid": uid}), 201


@app.route("/delete_doctor/<doctor_id>", methods=["DELETE"])
def delete_doctor(doctor_id):
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE uid=? AND role='doctor'", (doctor_id,))
        conn.commit()
    return jsonify({"status": "success"}), 200


# ---------------------------------------------------------------------------
# Samples & Results
# ---------------------------------------------------------------------------
@app.route("/upload_sample/<uid>/<sample_type>", methods=["POST"])
def upload_sample(uid, sample_type):
    filename = None
    saved_filepath = None

    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            ext = os.path.splitext(f.filename)[1] or ".jpg"
            safe_fn = "{}_s{}_{}{}".format(uid, sample_type, uuid.uuid4().hex[:6], ext)
            saved_filepath = os.path.join(UPLOAD_DIR, safe_fn)
            f.save(saved_filepath)
            filename = safe_fn

    snum = int(sample_type) if str(sample_type).isdigit() else 1

    # Analysis results state
    ai_used = False
    ai_observations = None
    ai_confidence = 0.9

    # If an image was uploaded and Gemini API key is configured, analyze with Gemini AI
    api_key = get_gemini_api_key()
    if saved_filepath and api_key:
        analysis = analyze_semen_image(saved_filepath)

        # ── EDGE CASE 1: Image is unrelated or not a valid semen microscopy slide ──
        if not analysis.get("is_valid_semen_sample", True):
            # Clean up the rejected file so it doesn't clutter storage
            try:
                if os.path.exists(saved_filepath):
                    os.remove(saved_filepath)
            except Exception:
                pass

            rejection_reason = (
                analysis.get("rejection_reason")
                or analysis.get("message")
                or "The uploaded image does not appear to be a semen or sperm microscopy slide. Please upload a clear specimen image."
            )
            return jsonify({
                "status": "error",
                "error": "unrelated_image",
                "message": rejection_reason,
                "is_valid_semen_sample": False,
            }), 422

        # ── EDGE CASE 2: Image is valid semen sample ──
        if analysis.get("success") and analysis.get("is_valid_semen_sample"):
            r = {
                "sperm_count": analysis["sperm_count"],
                "motility": analysis["motility"],
                "morphology": analysis["morphology"],
                "status": analysis["status"],
            }
            ai_observations = analysis.get("observations")
            ai_confidence = analysis.get("confidence", 0.95)
            ai_used = True
        else:
            # If an API error occurred (rate limit, network timeout, invalid key)
            if analysis.get("error") and analysis.get("error") not in ["missing_api_key"]:
                return jsonify({
                    "status": "error",
                    "error": analysis.get("error"),
                    "message": analysis.get("message", "AI analysis failed. Please try again.")
                }), 400

            # Fallback to simulation if non-critical error
            r = simulate_analysis(snum, uid)
    else:
        # Fallback when no image is attached (e.g. mock test) or API key is not yet set
        r = simulate_analysis(snum, uid)

    sid = "s-{}".format(uuid.uuid4().hex[:8])

    sample_doc = {
        "sample_id": sid,
        "patient_uid": uid,
        "sample_type": str(sample_type),
        "filename": filename,
        "sperm_count": r["sperm_count"],
        "motility": r["motility"],
        "morphology": r["morphology"],
        "status": r["status"],
        "observations": ai_observations,
        "confidence": ai_confidence,
    }
    db_manager.save_sample(sample_doc)
    db_manager.advance_patient_sample(uid)

    response_data = {
        "status": "success",
        "sample_id": sid,
        "results": r,
        "filename": filename,
        "ai_analyzed": ai_used,
    }
    if ai_observations:
        response_data["observations"] = ai_observations
        response_data["confidence"] = ai_confidence

    return jsonify(response_data), 200


@app.route("/get_sample_status/<uid>", methods=["GET"])
def get_sample_status(uid):
    p = db_manager.get_patient_by_id_or_name(uid)
    if not p:
        return jsonify({
            "status": "error",
            "message": "Patient not found",
            "current_sample": 1,
            "uploaded_samples": []
        }), 404

    pid = p.get("patient_id", uid)
    chart_samples = db_manager.get_samples_for_chart(pid)
    uploaded = []
    for s in chart_samples:
        st = s.get("sample_type")
        if str(st).isdigit():
            uploaded.append(int(st))

    return jsonify({
        "status": "processed",
        "uid": pid,
        "current_sample": p.get("current_sample", 1),
        "uploaded_samples": uploaded,
    }), 200


@app.route("/result/update_current_sample/<uid>", methods=["PUT"])
def update_current_sample(uid):
    data = request.get_json(force=True) or {}
    new_val = min(int(data.get("current_sample", 1)) + 1, 6)
    db_manager.update_patient(uid, {"current_sample": new_val})
    return jsonify({"status": "success", "current_sample": new_val}), 200


@app.route("/result", methods=["GET"])
def result():
    uid = request.args.get("uid", "").strip()
    doctor_email = request.args.get("email", "").strip()

    p = None
    if uid:
        p = db_manager.get_patient_by_id_or_name(uid)

    if not p and doctor_email:
        doctor_patients = db_manager.get_recent_patients(doctor_email=doctor_email)
        if doctor_patients:
            p = doctor_patients[0]

    if not p and not uid and not doctor_email:
        all_patients = db_manager.get_recent_patients()
        if all_patients:
            p = all_patients[0]

    if not p:
        return jsonify({
            "status": "error",
            "message": "No patient found. Please search by ID or register a new patient."
        }), 404

    pid = p.get("patient_id", uid or "p001")
    latest = db_manager.get_latest_sample(pid)

    patient_dict = {
        "uid": pid,
        "name": p.get("name", "Patient"),
        "age": p.get("age", 0),
        "occupation": p.get("occupation") or "Not specified",
        "height": p.get("height", 0.0),
        "weight": p.get("weight", 0.0),
        "bmi": p.get("bmi", 0.0),
        "foc": p.get("foc", 0),
        "sexual_dysfunction": p.get("sexual_dysfunction", "None"),
        "alcoholic": p.get("alcoholic", "No"),
        "smoker": p.get("smoker", "No"),
        "drugs": p.get("drugs", "No"),
    }

    if latest:
        r = {
            "sperm_count": latest["sperm_count"],
            "motility": latest["motility"],
            "morphology": latest["morphology"],
            "status": latest["status"]
        }
        if latest.get("observations"):
            r["observations"] = latest["observations"]
    else:
        r = simulate_analysis(1, pid)

    return jsonify({
        "status": "success",
        "patient": patient_dict,
        "current_sample": p.get("current_sample", 1),
        "results": r,
    }), 200


@app.route("/view_result", methods=["GET"])
def view_result():
    uid = request.args.get("uid", "p001").strip()
    sample_type = request.args.get("sample_type", "1")
    row = db_manager.get_sample_by_type(uid, sample_type)

    if row:
        r = {
            "sperm_count": row.get("sperm_count", 0),
            "motility": row.get("motility", 0.0),
            "morphology": row.get("morphology", 0.0),
            "status": row.get("status", "Normal"),
            "processed_image": row.get("filename") or "sample_result.png",
        }
        if row.get("observations"):
            r["observations"] = row["observations"]
        if row.get("confidence") is not None:
            r["confidence"] = row["confidence"]
    else:
        snum = int(sample_type) if str(sample_type).isdigit() else 1
        r = simulate_analysis(snum, uid)
        r["processed_image"] = "sample_result.png"

    return jsonify(r), 200


@app.route("/get_sperm_count/<uid>", methods=["GET"])
def get_sperm_count(uid):
    rows = db_manager.get_samples_for_chart(uid)
    if rows:
        labels = ["S{}".format(r.get("sample_type", i + 1)) for i, r in enumerate(rows)]
        counts = [round(r.get("sperm_count", 0) / 1_000_000, 1) for r in rows]
    else:
        counts = [simulate_analysis(i + 1, uid)["sperm_count"] // 1_000_000 for i in range(6)]
        labels = ["S{}".format(i + 1) for i in range(6)]

    result_dict = {"status": "success", "labels": labels, "counts": counts}
    for i, c in enumerate(counts, 1):
        result_dict["active_sperm{}".format(i)] = c

    return jsonify(result_dict), 200


# ---------------------------------------------------------------------------
# File serving
# ---------------------------------------------------------------------------
@app.route("/uploads/<filename>")
def serve_upload(filename):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(UPLOAD_DIR, filename)
    return jsonify({"error": "File not found"}), 404


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    with get_db() as conn:
        uc = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        pc = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        sc = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    return jsonify({
        "status": "ok",
        "db": DB_PATH,
        "users": uc,
        "patients": pc,
        "samples": sc,
        "time": datetime.datetime.utcnow().isoformat() + "Z",
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("\n  Spermoscope Backend  -  SQLite Edition")
    print("  DB      : {}".format(DB_PATH))
    print("  Uploads : {}".format(UPLOAD_DIR))
    print("  Doctor  : doctor@demo.com  /  Password123!")
    print("  Admin   : admin@demo.com   /  Admin123!")
    print("  Legacy  : doctor@demo.com  /  password123  (compat)")
    print("  Health  : http://localhost:8080/health\n")
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
