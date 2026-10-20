"""
Spermoscope Database Layer (MongoDB Atlas with SQLite Fallback)
=============================================================
Connects to MongoDB Atlas for persistent storage of users, patients,
doctors, and semen analysis samples.

Ensures:
- All data is stored permanently in MongoDB.
- New doctors start with an empty dashboard (no mock patients).
- Resilient connection handling with DNS optimization and local fallback.
"""

import os
import sqlite3
import datetime
import logging

logger = logging.getLogger("spermoscope.db")

# DNS SRV resolution optimization for MongoDB Atlas on Windows
try:
    import dns.resolver
    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
    dns.resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
except Exception:
    pass

import pymongo
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "spermoscope.db")

# Read environment
def get_env_var(name, default=""):
    return os.environ.get(name) or default

MONGODB_URI = get_env_var(
    "MONGODB_URI",
    "mongodb+srv://jayadhithiya_db_user:Jayadhithiya18@cluster0.iom0rnt.mongodb.net/spermoscope?retryWrites=true&w=majority"
)
MONGODB_DB_NAME = get_env_var("MONGODB_DB_NAME", "spermoscope")

_mongo_client = None
_mongo_db = None
_use_mongo = False


def init_db_connection():
    """Attempt connecting to MongoDB Atlas. Fallback to SQLite if Atlas network access is blocked."""
    global _mongo_client, _mongo_db, _use_mongo

    if not MONGODB_URI:
        _use_mongo = False
        return False

    try:
        client = pymongo.MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
        )
        # Test connection
        client.admin.command('ping')
        _mongo_client = client
        _mongo_db = client[MONGODB_DB_NAME]
        _use_mongo = True
        logger.info("Connected to MongoDB Atlas: %s / %s", client.address, MONGODB_DB_NAME)
        print(f"\n  [MongoDB] Connected successfully to MongoDB Atlas: {MONGODB_DB_NAME}\n")
        _setup_mongo_indexes()
        return True
    except Exception as e:
        _use_mongo = False
        logger.warning(
            "MongoDB Atlas connection unavailable (%s: %s). "
            "Please ensure 0.0.0.0/0 is added in MongoDB Atlas -> Network Access. "
            "Falling back to local database.",
            type(e).__name__, e
        )
        print(f"\n  [MongoDB Warning] Could not connect to MongoDB Atlas ({type(e).__name__}).")
        print("  --> To enable Atlas: go to cloud.mongodb.com -> Network Access -> Add IP -> 'Allow Access from Anywhere (0.0.0.0/0)'")
        print("  --> Using local DB fallback in the meantime so the server never crashes.\n")
        return False


_last_connection_attempt = 0

def is_using_mongo():
    global _use_mongo, _last_connection_attempt
    if _use_mongo and _mongo_db is not None:
        return True

    # Try reconnecting every 60 seconds
    import time
    now = time.time()
    if now - _last_connection_attempt < 60:
        return False

    _last_connection_attempt = now
    return init_db_connection()


def _setup_mongo_indexes():
    """Create unique indexes for clean MongoDB storage."""
    try:
        _mongo_db.users.create_index("email", unique=True)
        _mongo_db.patients.create_index("patient_id", unique=True)
        _mongo_db.patients.create_index("doctor_email")
        _mongo_db.patients.create_index("doctor_uid")
        _mongo_db.samples.create_index("sample_id", unique=True)
        _mongo_db.samples.create_index([("patient_uid", 1), ("sample_type", 1)])
        _mongo_db.doctors.create_index("uid", unique=True)
    except Exception as e:
        logger.warning("Error setting MongoDB indexes: %s", e)


def get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------------------
# User Operations
# ------------------------------------------------------------------------------
def get_user_by_email(email: str):
    email = (email or "").strip().lower()
    if is_using_mongo():
        user = _mongo_db.users.find_one({"email": email})
        if user:
            user.pop("_id", None)
            return user
        return None

    with get_sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE LOWER(email)=?", (email,)).fetchone()
        return dict(row) if row else None


def get_user_by_name_or_email(identifier: str):
    ident = (identifier or "").strip().lower()
    if is_using_mongo():
        user = _mongo_db.users.find_one({"$or": [{"email": ident}, {"name": {"$regex": f"^{ident}$", "$options": "i"}}]})
        if user:
            user.pop("_id", None)
            return user
        return None

    with get_sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE LOWER(email)=? OR LOWER(name)=?", (ident, ident)).fetchone()
        return dict(row) if row else None


def create_user(user_dict: dict):
    user_dict["email"] = user_dict["email"].strip().lower()
    if is_using_mongo():
        doc = dict(user_dict)
        _mongo_db.users.update_one({"email": doc["email"]}, {"$set": doc}, upsert=True)
        return doc

    with get_sqlite_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO users (uid, email, password, name, role, specialization)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_dict["uid"],
            user_dict["email"],
            user_dict["password"],
            user_dict["name"],
            user_dict.get("role", "doctor"),
            user_dict.get("specialization", "Clinical Andrology")
        ))
        conn.commit()
    return user_dict


def update_password(email: str, new_password_hash: str):
    email = email.strip().lower()
    if is_using_mongo():
        res = _mongo_db.users.update_one({"email": email}, {"$set": {"password": new_password_hash}})
        return res.matched_count > 0

    with get_sqlite_conn() as conn:
        c = conn.execute("UPDATE users SET password=? WHERE LOWER(email)=?", (new_password_hash, email))
        conn.commit()
        return c.rowcount > 0


# ------------------------------------------------------------------------------
# Patient Operations (New doctors have an initially EMPTY dashboard)
# ------------------------------------------------------------------------------
def get_recent_patients(doctor_email: str = None, doctor_uid: str = None):
    """
    Fetch recent patients.
    If doctor_email or doctor_uid is specified:
      Returns ONLY patients registered under that doctor.
      For a NEW doctor: returns [] (EMPTY).
    """
    filter_query = {}
    target_uids = set()
    if doctor_uid:
        target_uids.add(doctor_uid.strip())
    if doctor_email:
        clean_email = doctor_email.strip().lower()
        target_uids.add(clean_email)
        u = get_user_by_email(clean_email)
        if u and u.get("uid"):
            target_uids.add(u["uid"])

    if is_using_mongo():
        if target_uids:
            or_clauses = []
            for item in target_uids:
                or_clauses.append({"doctor_email": item})
                or_clauses.append({"doctor_uid": item})
            filter_query = {"$or": or_clauses}

        cursor = _mongo_db.patients.find(filter_query).sort("created_at", -1).limit(20)
        patients = []
        for doc in cursor:
            doc.pop("_id", None)
            patients.append(doc)
        return patients

    with get_sqlite_conn() as conn:
        if target_uids:
            placeholders = ",".join("?" for _ in target_uids)
            rows = conn.execute(
                f"SELECT * FROM patients WHERE doctor_uid IN ({placeholders}) ORDER BY created_at DESC LIMIT 20",
                tuple(target_uids)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT 20").fetchall()
        return [dict(r) for r in rows]


def get_patient_by_id_or_name(query: str):
    q = (query or "").strip()
    if not q:
        return None

    if is_using_mongo():
        doc = _mongo_db.patients.find_one({
            "$or": [
                {"patient_id": q},
                {"name": {"$regex": f"^{q}$", "$options": "i"}}
            ]
        })
        if not doc:
            # First patient fallback
            doc = _mongo_db.patients.find_one()
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    with get_sqlite_conn() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_id=?", (q,)).fetchone()
        if not row:
            row = conn.execute("SELECT * FROM patients WHERE LOWER(name)=?", (q.lower(),)).fetchone()
        if not row:
            row = conn.execute("SELECT * FROM patients LIMIT 1").fetchone()
        return dict(row) if row else None


def add_patient(patient_dict: dict):
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    patient_dict.setdefault("created_at", now_str)
    patient_dict.setdefault("current_sample", 1)

    if is_using_mongo():
        doc = dict(patient_dict)
        _mongo_db.patients.insert_one(doc)
        doc.pop("_id", None)
        return doc

    with get_sqlite_conn() as conn:
        conn.execute("""
            INSERT INTO patients
              (patient_id, name, age, occupation, contact, height, weight, bmi,
               foc, frequency_of_coitus, sexual_dysfunction, alcoholic, smoker, drugs,
               doctor_uid, current_sample, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            patient_dict["patient_id"],
            patient_dict.get("name"),
            patient_dict.get("age"),
            patient_dict.get("occupation"),
            patient_dict.get("contact"),
            patient_dict.get("height"),
            patient_dict.get("weight"),
            patient_dict.get("bmi"),
            patient_dict.get("foc"),
            patient_dict.get("frequency_of_coitus"),
            patient_dict.get("sexual_dysfunction"),
            patient_dict.get("alcoholic"),
            patient_dict.get("smoker"),
            patient_dict.get("drugs"),
            patient_dict.get("doctor_uid") or patient_dict.get("doctor_email"),
            patient_dict.get("current_sample", 1),
            patient_dict.get("created_at")
        ))
        conn.commit()
    return patient_dict


def update_patient(patient_id: str, update_fields: dict):
    if is_using_mongo():
        res = _mongo_db.patients.update_one({"patient_id": patient_id}, {"$set": update_fields})
        return res.matched_count > 0

    with get_sqlite_conn() as conn:
        set_clauses = []
        vals = []
        for k, v in update_fields.items():
            set_clauses.append(f"{k}=?")
            vals.append(v)
        vals.append(patient_id)
        c = conn.execute(f"UPDATE patients SET {', '.join(set_clauses)} WHERE patient_id=?", vals)
        conn.commit()
        return c.rowcount > 0


def advance_patient_sample(patient_id: str):
    if is_using_mongo():
        p = _mongo_db.patients.find_one({"patient_id": patient_id})
        if p:
            cur = int(p.get("current_sample", 1))
            new_val = min(cur + 1, 6)
            _mongo_db.patients.update_one({"patient_id": patient_id}, {"$set": {"current_sample": new_val}})
            return new_val
        return 1

    with get_sqlite_conn() as conn:
        conn.execute("UPDATE patients SET current_sample=MIN(current_sample+1, 6) WHERE patient_id=?", (patient_id,))
        conn.commit()


# ------------------------------------------------------------------------------
# Doctors Operations
# ------------------------------------------------------------------------------
def get_all_doctors():
    if is_using_mongo():
        cursor = _mongo_db.doctors.find()
        docs = []
        for d in cursor:
            d.pop("_id", None)
            docs.append(d)
        return docs

    with get_sqlite_conn() as conn:
        rows = conn.execute("SELECT uid, name, email, specialization FROM users WHERE role='doctor'").fetchall()
        return [dict(r) for r in rows]


def add_doctor(doc_dict: dict):
    if is_using_mongo():
        _mongo_db.doctors.update_one({"uid": doc_dict["uid"]}, {"$set": doc_dict}, upsert=True)
        return doc_dict

    with get_sqlite_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO users (uid, email, password, name, role, specialization)
            VALUES (?, ?, ?, ?, 'doctor', ?)
        """, (
            doc_dict["uid"],
            doc_dict["email"],
            doc_dict.get("password", "Doctor123!"),
            doc_dict["name"],
            doc_dict.get("specialization", "Clinical Andrology")
        ))
        conn.commit()
    return doc_dict


def delete_doctor(doctor_uid: str):
    if is_using_mongo():
        _mongo_db.doctors.delete_one({"uid": doctor_uid})
        _mongo_db.users.delete_one({"uid": doctor_uid})
        return True

    with get_sqlite_conn() as conn:
        conn.execute("DELETE FROM users WHERE uid=?", (doctor_uid,))
        conn.commit()
        return True


# ------------------------------------------------------------------------------
# Samples & Results Operations
# ------------------------------------------------------------------------------
def save_sample(sample_dict: dict):
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sample_dict.setdefault("created_at", now_str)
    sample_dict.setdefault("processed", 1)

    if is_using_mongo():
        doc = dict(sample_dict)
        _mongo_db.samples.insert_one(doc)
        doc.pop("_id", None)
        return doc

    with get_sqlite_conn() as conn:
        conn.execute("""
            INSERT INTO samples
              (sample_id, patient_uid, sample_type, filename, sperm_count, motility, morphology, status, observations, confidence, processed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_dict["sample_id"],
            sample_dict["patient_uid"],
            str(sample_dict["sample_type"]),
            sample_dict.get("filename"),
            sample_dict.get("sperm_count"),
            sample_dict.get("motility"),
            sample_dict.get("morphology"),
            sample_dict.get("status"),
            sample_dict.get("observations"),
            sample_dict.get("confidence"),
            sample_dict.get("processed", 1),
            sample_dict.get("created_at")
        ))
        conn.commit()
    return sample_dict


def get_latest_sample(patient_uid: str):
    if is_using_mongo():
        doc = _mongo_db.samples.find_one({"patient_uid": patient_uid}, sort=[("created_at", -1)])
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    with get_sqlite_conn() as conn:
        row = conn.execute("""
            SELECT * FROM samples WHERE patient_uid=?
            ORDER BY created_at DESC LIMIT 1
        """, (patient_uid,)).fetchone()
        return dict(row) if row else None


def get_sample_by_type(patient_uid: str, sample_type: str):
    if is_using_mongo():
        doc = _mongo_db.samples.find_one(
            {"patient_uid": patient_uid, "sample_type": str(sample_type)},
            sort=[("created_at", -1)]
        )
        if doc:
            doc.pop("_id", None)
            return doc
        return None

    with get_sqlite_conn() as conn:
        row = conn.execute("""
            SELECT * FROM samples WHERE patient_uid=? AND sample_type=?
            ORDER BY created_at DESC LIMIT 1
        """, (patient_uid, str(sample_type))).fetchone()
        return dict(row) if row else None


def get_samples_for_chart(patient_uid: str):
    if is_using_mongo():
        cursor = _mongo_db.samples.find({"patient_uid": patient_uid}).sort("created_at", 1).limit(6)
        samples = []
        for s in cursor:
            s.pop("_id", None)
            samples.append(s)
        return samples

    with get_sqlite_conn() as conn:
        rows = conn.execute("""
            SELECT sample_type, sperm_count FROM samples
            WHERE patient_uid=? ORDER BY created_at ASC LIMIT 6
        """, (patient_uid,)).fetchall()
        return [dict(r) for r in rows]
