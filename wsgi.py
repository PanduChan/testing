"""
Entry point untuk server WSGI produksi (mis. Gunicorn) saat aplikasi di-hosting.

Development (localhost):
    python app.py

Produksi (contoh hosting Render/Railway):
    gunicorn wsgi:app
"""
from app import app

if __name__ == "__main__":
    app.run()
