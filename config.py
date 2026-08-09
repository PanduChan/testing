"""
Konfigurasi aplikasi, dibaca dari environment variable (file .env saat development,
atau dari dashboard hosting seperti Render/Railway saat produksi).

Penting: SECRET_KEY dan ADMIN_PASSWORD harus diisi lewat environment variable saat
hosting produksi, jangan pernah memakai nilai default di bawah ini di luar localhost.
"""
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()  # baca file .env jika ada (tidak berpengaruh saat di-deploy, karena env
                # variable biasanya sudah diset langsung oleh platform hosting)


class Config:
    # --- Keamanan ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-JANGAN-dipakai-di-produksi")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Aktifkan cookie khusus HTTPS otomatis kalau berjalan di balik hosting produksi
    SESSION_COOKIE_SECURE = os.environ.get("FLASK_ENV") == "production"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

    # --- Aplikasi ---
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # batas unggah 10 MB

    # --- Kredensial admin (WAJIB diganti lewat environment variable saat hosting) ---
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "bungo2026")

    # --- Rate limiting login sederhana ---
    LOGIN_MAX_ATTEMPTS = int(os.environ.get("LOGIN_MAX_ATTEMPTS", 5))
    LOGIN_LOCKOUT_MINUTES = int(os.environ.get("LOGIN_LOCKOUT_MINUTES", 10))

    def validate_production(self):
        """Panggil saat startup untuk memperingatkan kalau nilai default masih dipakai
        pada environment produksi."""
        warnings = []
        if self.SECRET_KEY == "dev-secret-key-JANGAN-dipakai-di-produksi":
            warnings.append("SECRET_KEY masih memakai nilai default developer.")
        if self.ADMIN_PASSWORD == "bungo2026":
            warnings.append("ADMIN_PASSWORD masih memakai nilai default prototype.")
        return warnings


config = Config()
