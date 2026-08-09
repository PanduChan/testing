"""
Test suite untuk fitur eksplorasi dashboard: horizon peramalan fleksibel dan
filter drill-down per pengembang. Fitur ini melengkapi (bukan mengganti) horizon
12 bulan yang menjadi komitmen utama pada Tujuan Penelitian BAB I.

Jalankan dengan:
    pytest tests/test_features.py -v
"""
import os

import pytest

os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "bungo2026")
os.environ.setdefault("SECRET_KEY", "test-secret-key")

from app import app, DATA_PATH  # noqa: E402
from utils.arima_utils import load_data, list_pengembang  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_horizon_default_12_bulan_sesuai_tujuan_penelitian(client):
    """Horizon default HARUS 12 bulan, sesuai komitmen eksplisit pada Tujuan
    Penelitian BAB I, meskipun opsi horizon lain tersedia sebagai eksplorasi."""
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Proyeksi 12 Bulan" in html


@pytest.mark.parametrize("horizon", [3, 6, 12, 24])
def test_horizon_pilihan_valid(client, horizon):
    r = client.get(f"/?horizon={horizon}")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert f"Proyeksi {horizon} Bulan" in html


def test_horizon_tidak_valid_fallback_ke_12(client):
    r = client.get("/?horizon=999")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Proyeksi 12 Bulan" in html


def test_filter_pengembang_valid_menampilkan_nama_di_header(client):
    df = load_data(DATA_PATH)
    daftar = list_pengembang(df)
    contoh = daftar[0]

    r = client.get(f"/?pengembang={contoh}")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert contoh in html


def test_filter_pengembang_data_terlalu_sedikit_tidak_error(client):
    """Pengembang dengan riwayat data pendek (< 12 bulan) tidak boleh menyebabkan
    error 500; sistem harus fallback menampilkan data historis tanpa proyeksi."""
    df = load_data(DATA_PATH)
    monthly_per_dev = df.groupby(["pengembang", "periode"]).size().reset_index()
    counts = monthly_per_dev.groupby("pengembang").size().sort_values()
    pengembang_data_sedikit = counts.index[0]

    r = client.get(f"/?pengembang={pengembang_data_sedikit}")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "data historis saja" in html.lower() or "belum ada proyeksi" in html.lower()


def test_filter_pengembang_tidak_dikenal_diabaikan(client):
    """Nama pengembang yang tidak ada di data harus diabaikan dengan aman
    (fallback ke agregat), bukan error."""
    r = client.get("/?pengembang=Nama Pengembang Yang Tidak Pernah Ada")
    assert r.status_code == 200
