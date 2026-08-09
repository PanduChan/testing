"""
Sistem Informasi Prediksi Realisasi Fasilitas Likuiditas Pembiayaan Perumahan
(FLPP) Kabupaten Bungo menggunakan Metode ARIMA
Flask + ARIMA (statsmodels) — siap untuk hosting produksi maupun localhost.

Development (localhost):
    pip install -r requirements.txt
    cp .env.example .env      # lalu isi SECRET_KEY & ADMIN_PASSWORD di .env
    python app.py
Lalu buka http://127.0.0.1:5000

Produksi (hosting, mis. Render/Railway/PythonAnywhere):
    Set environment variable SECRET_KEY, ADMIN_USERNAME, ADMIN_PASSWORD,
    FLASK_ENV=production lewat dashboard hosting, lalu jalankan:
    gunicorn wsgi:app

Peran pengguna:
    - Masyarakat/MBR (pengunjung): langsung melihat dashboard di "/" tanpa perlu login.
    - Admin: login di "/login" untuk mengakses "/admin" (kelola data historis).

Untuk memakai data asli:
    Admin bisa unggah file CSV baru langsung lewat halaman /admin (disarankan,
    karena otomatis divalidasi), atau ganti manual isi file
    data/dataset_flpp_raw.csv dengan data dari tapera.go.id.

    Dua format kolom yang didukung (lihat utils/arima_utils.py -> load_data):
    1. Format olahan   : periode (YYYY-MM-DD), pengembang, jumlah_realisasi
    2. Format mentah   : 'Tanggal Pencairan', 'Nama Pengembang', dst. (tapera.go.id)
"""
import os
import shutil
import logging
from datetime import datetime, timedelta, timezone

from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import config
from utils.arima_utils import run_full_pipeline, load_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "dataset_flpp_raw.csv")
DATA_BACKUP_PATH = os.path.join(DATA_DIR, "dataset_flpp_raw.backup.csv")
UPLOAD_TMP_PATH = os.path.join(DATA_DIR, "_upload_tmp.csv")

ALLOWED_EXTENSIONS = {"csv"}

# ============================================================
# Logging — penting saat di-hosting, karena tidak ada terminal
# interaktif untuk melihat print() secara langsung.
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["SESSION_COOKIE_HTTPONLY"] = config.SESSION_COOKIE_HTTPONLY
app.config["SESSION_COOKIE_SAMESITE"] = config.SESSION_COOKIE_SAMESITE
app.config["SESSION_COOKIE_SECURE"] = config.SESSION_COOKIE_SECURE
app.config["PERMANENT_SESSION_LIFETIME"] = config.PERMANENT_SESSION_LIFETIME

for w in config.validate_production():
    logger.warning("KONFIGURASI TIDAK AMAN UNTUK PRODUKSI: %s", w)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Silakan login terlebih dahulu untuk mengakses halaman admin."


# --- Autentikasi sederhana (prototype) ---
# Kredensial diambil dari environment variable (lihat config.py), bukan hardcoded.
# Untuk versi lanjutan, ganti dengan Flask-SQLAlchemy + tabel users di database.
USERS = {
    config.ADMIN_USERNAME: {
        "id": "1",
        "password_hash": generate_password_hash(config.ADMIN_PASSWORD),
        "role": "admin",
    }
}

# --- Rate limiting login sederhana (in-memory, cukup untuk skala prototype/skripsi) ---
# Untuk trafik produksi yang lebih besar, ganti dengan Flask-Limiter + Redis.
_login_attempts = {}  # {ip: [timestamp, ...]}


def _is_rate_limited(ip: str) -> bool:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=config.LOGIN_LOCKOUT_MINUTES)
    attempts = [t for t in _login_attempts.get(ip, []) if t > window_start]
    _login_attempts[ip] = attempts
    return len(attempts) >= config.LOGIN_MAX_ATTEMPTS


def _record_failed_attempt(ip: str):
    _login_attempts.setdefault(ip, []).append(datetime.now(timezone.utc))


class User(UserMixin):
    def __init__(self, username):
        self.id = username


@login_manager.user_loader
def load_user(username):
    if username in USERS:
        return User(username)
    return None


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _data_status() -> dict:
    """Ringkasan status file data saat ini, ditampilkan di halaman admin."""
    if not os.path.exists(DATA_PATH):
        return {"exists": False, "has_backup": os.path.exists(DATA_BACKUP_PATH)}

    status = {
        "exists": True,
        "size_kb": round(os.path.getsize(DATA_PATH) / 1024, 1),
        "modified_at": datetime.fromtimestamp(os.path.getmtime(DATA_PATH)).strftime("%d %b %Y, %H:%M"),
        "has_backup": os.path.exists(DATA_BACKUP_PATH),
    }
    try:
        df = load_data(DATA_PATH)
        status["is_valid"] = True
        status["n_rows"] = len(df)
        status["n_pengembang"] = int(df["pengembang"].nunique())
        status["periode_awal"] = df["periode"].min().strftime("%b %Y")
        status["periode_akhir"] = df["periode"].max().strftime("%b %Y")
    except Exception as exc:
        status["is_valid"] = False
        status["error"] = str(exc)

    return status


# ============================================================
# Rute PUBLIK — dashboard & data pendukung, tanpa perlu login
# ============================================================

ALLOWED_HORIZONS = [3, 6, 12, 24]


@app.route("/")
@app.route("/dashboard")
def dashboard():
    horizon = request.args.get("horizon", 12, type=int)
    if horizon not in ALLOWED_HORIZONS:
        horizon = 12
    pengembang_dipilih = request.args.get("pengembang", "").strip() or None

    try:
        result = run_full_pipeline(DATA_PATH, forecast_steps=horizon, pengembang=pengembang_dipilih)
    except Exception as exc:
        logger.exception("Gagal menjalankan pipeline ARIMA")
        flash(f"Gagal memproses data: {exc}", "error")
        return render_template("dashboard.html", has_data=False)

    historis = result["historis"]
    forecast_df = result["forecast"]
    dev_summary = result["developer_summary"]
    ada_proyeksi = result["order"] is not None and len(forecast_df) > 0

    if result.get("peringatan"):
        flash(result["peringatan"], "error")

    chart_data = {
        "historis_label": [d.strftime("%b %Y") for d in historis.index],
        "historis_value": [round(float(v), 1) for v in historis.values],
        "forecast_label": [d.strftime("%b %Y") for d in forecast_df["periode"]] if ada_proyeksi else [],
        "forecast_value": forecast_df["prediksi"].tolist() if ada_proyeksi else [],
        "forecast_lower": forecast_df["batas_bawah"].tolist() if ada_proyeksi else [],
        "forecast_upper": forecast_df["batas_atas"].tolist() if ada_proyeksi else [],
    }

    kpi = {
        "total_realisasi": int(historis.sum()),
        "rata_rata_bulanan": round(float(historis.mean()), 1),
        "realisasi_terakhir": int(historis.iloc[-1]) if len(historis) else 0,
        "prediksi_bulan_depan": float(forecast_df["prediksi"].iloc[0]) if ada_proyeksi else None,
        "total_proyeksi_horizon": float(forecast_df["prediksi"].sum()) if ada_proyeksi else None,
    }

    return render_template(
        "dashboard.html",
        has_data=True,
        ada_proyeksi=ada_proyeksi,
        chart_data=chart_data,
        order=result["order"],
        adf=result["adf_test"],
        diagnostics=result["diagnostics"],
        evaluation=result["evaluation"],
        dev_summary=dev_summary.to_dict(orient="records"),
        kpi=kpi,
        using_pmdarima=result["using_pmdarima"],
        daftar_pengembang=result["daftar_pengembang"],
        pengembang_aktif=result["pengembang_aktif"],
        horizon_aktif=horizon,
        horizon_pilihan=ALLOWED_HORIZONS,
    )


@app.route("/api/forecast")
def api_forecast():
    """Endpoint JSON publik — data pendukung transparansi, bukan fungsi manajemen."""
    try:
        result = run_full_pipeline(DATA_PATH, forecast_steps=12)
    except Exception as exc:
        logger.exception("Gagal menjalankan pipeline ARIMA (api_forecast)")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "order": result["order"],
        "evaluation": result["evaluation"],
        "diagnostics": result["diagnostics"],
        "forecast": result["forecast"].to_dict(orient="records"),
    })


@app.route("/health")
def health():
    """Endpoint health-check — dipakai platform hosting untuk memantau status aplikasi."""
    return jsonify({"status": "ok", "data_tersedia": os.path.exists(DATA_PATH)})


# ============================================================
# Rute ADMIN — perlu login untuk kelola data
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin"))

    if request.method == "POST":
        client_ip = request.remote_addr or "unknown"

        if _is_rate_limited(client_ip):
            flash(
                f"Terlalu banyak percobaan login gagal. Coba lagi dalam "
                f"{config.LOGIN_LOCKOUT_MINUTES} menit.",
                "error",
            )
            return render_template("login.html")

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user_record = USERS.get(username)
        if user_record and check_password_hash(user_record["password_hash"], password):
            login_user(User(username))
            logger.info("Login berhasil untuk user '%s' dari %s", username, client_ip)
            next_url = request.args.get("next")
            return redirect(next_url or url_for("admin"))

        _record_failed_attempt(client_ip)
        logger.warning("Percobaan login gagal untuk '%s' dari %s", username, client_ip)
        flash("Username atau password salah.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("dashboard"))


@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if request.method == "POST":
        file = request.files.get("csv_file")

        if not file or file.filename == "":
            flash("Pilih file CSV terlebih dahulu.", "error")
            return redirect(url_for("admin"))

        if not _allowed_file(file.filename):
            flash("Hanya file berformat .csv yang diterima.", "error")
            return redirect(url_for("admin"))

        filename = secure_filename(file.filename)
        os.makedirs(DATA_DIR, exist_ok=True)
        file.save(UPLOAD_TMP_PATH)

        # Validasi file terlebih dahulu lewat pipeline yang sama sebelum menimpa data lama,
        # supaya file yang formatnya salah tidak sampai merusak dashboard yang sedang dipakai.
        try:
            df = load_data(UPLOAD_TMP_PATH)
            if len(df) == 0:
                raise ValueError("File tidak berisi baris data.")
        except Exception as exc:
            if os.path.exists(UPLOAD_TMP_PATH):
                os.remove(UPLOAD_TMP_PATH)
            logger.warning("Unggahan data ditolak: %s", exc)
            flash(f"File '{filename}' ditolak: {exc}", "error")
            return redirect(url_for("admin"))

        # Cadangkan data lama sebelum ditimpa, supaya bisa dipulihkan kalau perlu
        if os.path.exists(DATA_PATH):
            shutil.copy(DATA_PATH, DATA_BACKUP_PATH)

        shutil.move(UPLOAD_TMP_PATH, DATA_PATH)
        logger.info("Data historis diperbarui oleh '%s' dari file '%s'", current_user.id, filename)
        flash(
            f"Data berhasil diperbarui dari file '{filename}'. "
            "Dashboard publik otomatis memakai data terbaru saat dibuka.",
            "message",
        )
        return redirect(url_for("admin"))

    return render_template("admin.html", status=_data_status())


@app.route("/admin/restore", methods=["POST"])
@login_required
def admin_restore():
    if not os.path.exists(DATA_BACKUP_PATH):
        flash("Tidak ada data cadangan untuk dipulihkan.", "error")
        return redirect(url_for("admin"))

    shutil.copy(DATA_BACKUP_PATH, DATA_PATH)
    logger.info("Data dipulihkan dari cadangan oleh '%s'", current_user.id)
    flash("Data berhasil dipulihkan ke versi sebelumnya.", "message")
    return redirect(url_for("admin"))


# ============================================================
# Error handler — supaya traceback internal tidak terekspos ke
# publik saat aplikasi sudah di-hosting (penting untuk keamanan).
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error")
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=config.DEBUG)
