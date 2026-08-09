"""
Test suite formal — mengimplementasikan skenario Black Box Testing (BB-01 s.d. BB-11)
yang didokumentasikan pada BAB V skripsi, sebagai bukti otomatis dan reprodusibel.

Jalankan dengan:
    pytest tests/ -v
"""
import io
import os
import shutil

import pytest

os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "bungo2026")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from app import app, DATA_PATH, DATA_BACKUP_PATH  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
    # bersih-bersih backup sisa pengujian
    if os.path.exists(DATA_BACKUP_PATH):
        os.remove(DATA_BACKUP_PATH)


@pytest.fixture
def original_data():
    """Simpan & kembalikan isi data asli supaya pengujian tidak merusak dataset nyata."""
    with open(DATA_PATH, "rb") as f:
        content = f.read()
    yield content
    with open(DATA_PATH, "wb") as f:
        f.write(content)


def login(client, username="admin", password="bungo2026"):
    return client.post("/login", data={"username": username, "password": password},
                        follow_redirects=True)


# ---------- BB-01 s.d. BB-04: autentikasi & akses halaman ----------

def test_bb01_login_benar(client):
    r = login(client)
    assert r.status_code == 200
    assert "Kelola Data Historis".encode() in r.data


def test_bb02_login_salah(client):
    r = login(client, password="salah_banget")
    assert r.status_code == 200
    assert "Username atau password salah".encode() in r.data


def test_bb03_dashboard_tanpa_login(client):
    r = client.get("/")
    assert r.status_code == 200


def test_bb04_admin_tanpa_login_dialihkan(client):
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("Location", "")


# ---------- BB-05 s.d. BB-07: unggah data ----------

def test_bb05_upload_csv_valid(client, original_data):
    login(client)
    valid_csv = b"periode,pengembang,jumlah_realisasi\n2026-01-01,Contoh,5\n"
    r = client.post("/admin", data={"csv_file": (io.BytesIO(valid_csv), "valid.csv")},
                     content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    assert "berhasil diperbarui".encode() in r.data
    assert os.path.exists(DATA_BACKUP_PATH)


def test_bb06_upload_skema_tidak_dikenali(client, original_data):
    login(client)
    bad_csv = b"kolom_acak,lainnya\nfoo,bar\n"
    r = client.post("/admin", data={"csv_file": (io.BytesIO(bad_csv), "invalid.csv")},
                     content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    assert "ditolak".encode() in r.data


def test_bb07_upload_bukan_csv(client, original_data):
    login(client)
    r = client.post("/admin", data={"csv_file": (io.BytesIO(b"halo"), "data.txt")},
                     content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    assert "Hanya file berformat .csv".encode() in r.data


# ---------- BB-08 s.d. BB-09: restore & logout ----------

def test_bb08_restore_backup(client, original_data):
    login(client)
    valid_csv = b"periode,pengembang,jumlah_realisasi\n2026-01-01,Contoh,5\n"
    client.post("/admin", data={"csv_file": (io.BytesIO(valid_csv), "valid.csv")},
                content_type="multipart/form-data")

    r = client.post("/admin/restore", follow_redirects=True)
    assert r.status_code == 200
    assert "berhasil dipulihkan".encode() in r.data

    with open(DATA_PATH, "rb") as f:
        assert f.read() == original_data


def test_bb09_logout(client):
    login(client)
    r = client.get("/logout", follow_redirects=True)
    assert r.status_code == 200


# ---------- BB-10 s.d. BB-11: regresi (bug yang pernah ditemukan) ----------

def test_bb10_ringkasan_pengembang_tidak_selalu_satu():
    """Regresi: rata-rata bulanan per pengembang dulu selalu bernilai 1 karena
    dihitung langsung dari data level transaksi, bukan dari agregasi bulanan."""
    from utils.arima_utils import load_data, per_developer_summary
    df = load_data(DATA_PATH)
    summary = per_developer_summary(df)
    assert not (summary["rata_rata_bulanan"] == 1).all(), (
        "Regresi bug BB-10 muncul lagi: rata-rata bulanan seluruh pengembang "
        "kembali bernilai 1."
    )


def test_bb11_select_order_tidak_pakai_pmdarima():
    """Regresi: pastikan select_order() konsisten TIDAK memanggil pmdarima.auto_arima,
    sesuai metodologi grid-search AIC dengan d ditetapkan via ADF di BAB III/IV."""
    from utils.arima_utils import load_data, aggregate_monthly, run_full_pipeline
    result = run_full_pipeline(DATA_PATH, forecast_steps=3)
    assert result["using_pmdarima"] is False
    assert "metode_identifikasi" in result
