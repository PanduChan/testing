"""
Modul pemodelan ARIMA mengikuti metodologi Box-Jenkins:
identifikasi -> estimasi -> pengujian diagnostik -> peramalan.

Digunakan oleh app.py untuk memodelkan jumlah realisasi KPR FLPP.
"""
import re
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False


# Beberapa nama pengembang pada data sumber (tapera.go.id) ditulis tidak
# konsisten (beda kapitalisasi, ada akhiran ", PT" yang nyasar, atau salah
# ketik ejaan), sehingga satu perusahaan yang sama bisa terhitung sebagai
# beberapa "pengembang" berbeda. _normalize_nama_pengembang() menyeragamkan
# penulisannya sebelum data diagregasi, supaya ringkasan per pengembang
# (per_developer_summary) tidak pecah menjadi baris-baris duplikat.
_TYPO_MANUAL = {
    "PT UNO RESIDANCE PROPERTY": "PT UNO RESIDENCE PROPERTY",
}


def _normalize_nama_pengembang(nama: str) -> str:
    n = str(nama).strip().upper()
    n = re.sub(r",\s*PT$", "", n)          # buang akhiran ", PT" yang nyasar
    n = re.sub(r"\s+", " ", n).strip()
    if not n.startswith("PT "):
        n = "PT " + n
    n = _TYPO_MANUAL.get(n, n)
    return n


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Muat data historis KPR FLPP.

    Mendukung dua format:
    1. Format olahan: kolom periode, pengembang, jumlah_realisasi (satu baris = agregat).
    2. Format mentah tapera.go.id (transaksi per baris): kolom 'Tanggal Pencairan' dan
       'Nama Pengembang', dsb. Setiap baris merepresentasikan satu unit realisasi KPR FLPP.
       Format ini otomatis dikonversi menjadi skema (1) di atas.

    Nama pengembang pada kedua format di atas dinormalisasi lewat
    _normalize_nama_pengembang() agar variasi penulisan (kapitalisasi, akhiran
    ", PT" yang nyasar, salah ketik ejaan) untuk perusahaan yang sama tidak
    terhitung sebagai pengembang yang berbeda-beda.
    """
    raw = pd.read_csv(csv_path)

    if {"periode", "pengembang", "jumlah_realisasi"}.issubset(raw.columns):
        df = raw.copy()
        df["periode"] = pd.to_datetime(df["periode"])
        df["pengembang"] = df["pengembang"].apply(_normalize_nama_pengembang)
        return df.sort_values("periode")

    if "Tanggal Pencairan" in raw.columns and "Nama Pengembang" in raw.columns:
        tgl = pd.to_datetime(raw["Tanggal Pencairan"], format="%d %B, %Y", errors="coerce")
        if tgl.isna().any():
            # fallback bila format tanggal sedikit berbeda antar baris
            tgl = tgl.fillna(pd.to_datetime(raw["Tanggal Pencairan"], errors="coerce"))
        df = pd.DataFrame({
            "periode": tgl.dt.to_period("M").dt.to_timestamp(),
            "pengembang": raw["Nama Pengembang"].apply(_normalize_nama_pengembang),
            "jumlah_realisasi": 1,
        })
        df = df.dropna(subset=["periode"])
        return df.sort_values("periode")

    raise ValueError(
        "Format data tidak dikenali. Kolom yang tersedia: " + ", ".join(raw.columns)
    )


def aggregate_monthly(df: pd.DataFrame) -> pd.Series:
    """Agregasi total jumlah realisasi seluruh pengembang per bulan (deret waktu utama)."""
    monthly = df.groupby("periode")["jumlah_realisasi"].sum()
    monthly = monthly.asfreq("MS")
    monthly = monthly.interpolate(limit_direction="both")  # tangani missing values
    return monthly


def list_pengembang(df: pd.DataFrame) -> list:
    """Daftar nama pengembang unik, diurutkan alfabetis, untuk pilihan filter dashboard."""
    return sorted(df["pengembang"].dropna().unique().tolist())


def per_developer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ringkasan total, rata-rata bulanan, dan realisasi bulan terakhir per pengembang.

    PENTING: df berisi data pada level baris = satu transaksi/agregat per (periode,
    pengembang), sehingga harus diagregasi ke level bulanan per pengembang terlebih
    dahulu sebelum menghitung rata-rata maupun nilai bulan terakhir. Menghitung
    langsung dari data mentah akan keliru (mis. rata-rata selalu 1 jika setiap baris
    mewakili satu unit realisasi).
    """
    monthly_per_dev = (
        df.groupby(["pengembang", "periode"])["jumlah_realisasi"].sum().reset_index()
    )

    last_period = df["periode"].max()

    total = monthly_per_dev.groupby("pengembang")["jumlah_realisasi"].sum()
    rata2 = monthly_per_dev.groupby("pengembang")["jumlah_realisasi"].mean().round(1)
    last_val = (
        monthly_per_dev[monthly_per_dev["periode"] == last_period]
        .set_index("pengembang")["jumlah_realisasi"]
    )

    summary = pd.DataFrame({
        "total_realisasi": total,
        "rata_rata_bulanan": rata2,
        "realisasi_bulan_terakhir": last_val,
    }).reset_index()
    summary["realisasi_bulan_terakhir"] = summary["realisasi_bulan_terakhir"].fillna(0).astype(int)
    summary = summary.sort_values("total_realisasi", ascending=False)
    return summary


def adf_test(series: pd.Series) -> dict:
    """Uji stasioneritas Augmented Dickey-Fuller (tahap Identifikasi)."""
    result = adfuller(series.dropna())
    return {
        "adf_stat": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": bool(result[1] < 0.05),
    }


def determine_d(series: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """
    Tahap Identifikasi (bagian 1): tentukan orde differencing d melalui pengujian
    ADF berulang. AIC hanya boleh dibandingkan antar model dengan d yang SAMA, karena
    proses differencing mengubah jumlah observasi efektif dan skala likelihood-nya
    (Hyndman & Athanasopoulos, 2021). Oleh karena itu d harus ditetapkan lebih dulu
    sebelum p dan q dibandingkan berdasarkan AIC.
    """
    current = series.dropna()
    for d in range(0, max_d + 1):
        p_value = adfuller(current)[1]
        if p_value < alpha:
            return d
        current = current.diff().dropna()
    return max_d


def select_order(series: pd.Series):
    """
    Tahap Identifikasi & Estimasi: pilih orde (p,d,q) terbaik.

    Langkah:
    1. Tetapkan d lebih dulu berdasarkan uji ADF berulang (determine_d).
    2. Bandingkan AIC hanya antar kombinasi (p,q) pada d yang sama (grid search),
       mengikuti prinsip bahwa AIC tidak valid dibandingkan lintas nilai d berbeda.

    Fungsi ini TIDAK bergantung pada penentuan d otomatis bawaan pmdarima (yang
    memakai uji KPSS secara internal dan dapat berbeda kesimpulan dari uji ADF pada
    BAB III), agar konsisten dengan metodologi yang dituliskan di BAB III/BAB IV.
    """
    d = determine_d(series)

    best_aic, best_order = np.inf, (0, d, 0)
    max_p, max_q = 3, 3
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(series, order=(p, d, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, (p, d, q)
            except Exception:
                continue
    return best_order


def fit_arima(series: pd.Series, order):
    """Tahap Estimasi: fit model ARIMA dengan Maximum Likelihood Estimation."""
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def diagnostic_check(fitted) -> dict:
    """Tahap Pengujian Diagnostik: Ljung-Box (autokorelasi residual)."""
    resid = fitted.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[min(10, max(1, len(resid) // 2))], return_df=True)
    p_value = float(lb["lb_pvalue"].iloc[0])
    return {
        "ljung_box_p_value": round(p_value, 4),
        "residual_white_noise": bool(p_value > 0.05),
        "aic": round(float(fitted.aic), 2),
        "bic": round(float(fitted.bic), 2),
    }


def evaluate_forecast(series: pd.Series, order, test_ratio: float = 0.2) -> dict:
    """
    Evaluasi model dengan split data latih:uji 80:20 (sesuai batasan penelitian).
    Menghitung MAE, RMSE, dan MAPE.
    """
    n = len(series)
    n_test = max(1, int(round(n * test_ratio)))
    train, test = series.iloc[: n - n_test], series.iloc[n - n_test:]

    fitted = fit_arima(train, order)
    forecast = fitted.forecast(steps=n_test)

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    # Hindari pembagian dengan nol pada MAPE
    safe_test = test.replace(0, np.nan)
    mape = float(np.nanmean(np.abs((test - forecast) / safe_test)) * 100)

    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "mape": round(mape, 2),
        "is_accurate": bool(mape < 10),
        "n_train": n - n_test,
        "n_test": n_test,
    }


def forecast_future(fitted, series: pd.Series, steps: int = 12) -> pd.DataFrame:
    """Tahap Peramalan: proyeksi h bulan ke depan + interval kepercayaan 95%."""
    result = fitted.get_forecast(steps=steps)
    mean = result.predicted_mean
    conf_int = result.conf_int(alpha=0.05)

    last_date = series.index[-1]
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

    out = pd.DataFrame({
        "periode": future_index,
        "prediksi": mean.values,
        "batas_bawah": conf_int.iloc[:, 0].values,
        "batas_atas": conf_int.iloc[:, 1].values,
    })
    out["prediksi"] = out["prediksi"].clip(lower=0).round(1)
    out["batas_bawah"] = out["batas_bawah"].clip(lower=0).round(1)
    out["batas_atas"] = out["batas_atas"].clip(lower=0).round(1)
    return out


def run_full_pipeline(csv_path: str, forecast_steps: int = 12, pengembang: str = None) -> dict:
    """
    Jalankan seluruh pipeline Box-Jenkins dan kembalikan hasil untuk dashboard.

    Parameters
    ----------
    forecast_steps : int
        Horizon peramalan dalam bulan. Default 12 bulan sesuai Tujuan Penelitian
        pada BAB I; nilai lain (mis. 3, 6, 24) tersedia sebagai opsi eksplorasi
        tambahan pada dashboard, tanpa mengubah komitmen horizon 12 bulan sebagai
        hasil akhir utama.
    pengembang : str, optional
        Jika diisi, pipeline hanya mengolah data historis milik satu pengembang
        tersebut (mode drill-down), bukan agregat seluruh Kabupaten Bungo.
    """
    df_full = load_data(csv_path)
    daftar_pengembang = list_pengembang(df_full)

    df = df_full
    if pengembang and pengembang in daftar_pengembang:
        df = df_full[df_full["pengembang"] == pengembang]

    monthly = aggregate_monthly(df)

    # Mode drill-down per pengembang bisa menghasilkan deret waktu yang sangat pendek
    # (mis. pengembang baru dengan riwayat < 1 tahun), yang tidak cukup andal untuk
    # diestimasi dengan ARIMA. Pada kondisi ini sistem tetap menampilkan data historis
    # apa adanya tanpa memaksakan proyeksi yang tidak bermakna secara statistik.
    MIN_OBSERVASI_ARIMA = 12
    # Pengembang dengan transaksi sangat jarang (mis. cuma 2-3 transaksi dalam
    # rentang waktu panjang) bisa menghasilkan deret waktu yang, setelah
    # interpolasi bulan-bulan kosong, nilainya konstan (tidak ada variasi sama
    # sekali). Uji ADF tidak dapat dihitung pada deret konstan (menyebabkan error
    # "x is constant" pada statsmodels), sehingga kondisi ini perlu ditangani
    # sebelum masuk ke tahap uji stasioneritas.
    data_kurang_bervariasi = monthly.nunique(dropna=True) <= 1

    if len(monthly) < MIN_OBSERVASI_ARIMA or data_kurang_bervariasi:
        if data_kurang_bervariasi:
            pesan = (
                "Data historis untuk pilihan ini tidak cukup bervariasi (transaksi "
                "sangat jarang, sehingga setelah bulan-bulan kosong diisi, nilainya "
                "menjadi datar/konstan) sehingga model ARIMA tidak dapat diestimasi "
                "secara statistik. Menampilkan data historis tanpa proyeksi."
            )
        else:
            pesan = (
                f"Data historis untuk pilihan ini hanya {len(monthly)} bulan "
                f"(minimum {MIN_OBSERVASI_ARIMA} bulan diperlukan agar estimasi ARIMA "
                "cukup andal secara statistik). Menampilkan data historis tanpa proyeksi."
            )
        return {
            "historis": monthly,
            "order": None,
            "adf_test": None,
            "diagnostics": None,
            "evaluation": None,
            "forecast": pd.DataFrame(columns=["periode", "prediksi", "batas_bawah", "batas_atas"]),
            "developer_summary": per_developer_summary(df_full),
            "using_pmdarima": False,
            "metode_identifikasi": None,
            "peringatan": pesan,
            "daftar_pengembang": daftar_pengembang,
            "pengembang_aktif": pengembang if pengembang in daftar_pengembang else None,
            "forecast_steps": forecast_steps,
        }

    adf_before = adf_test(monthly)
    order = select_order(monthly)
    fitted = fit_arima(monthly, order)
    diagnostics = diagnostic_check(fitted)
    evaluation = evaluate_forecast(monthly, order)
    forecast_df = forecast_future(fitted, monthly, steps=forecast_steps)
    dev_summary = per_developer_summary(df_full)

    return {
        "historis": monthly,
        "order": order,
        "adf_test": adf_before,
        "diagnostics": diagnostics,
        "evaluation": evaluation,
        "forecast": forecast_df,
        "developer_summary": dev_summary,
        # select_order() SELALU memakai grid-search AIC dengan d yang ditetapkan lebih
        # dulu via uji ADF (lihat determine_d & select_order) — bukan auto_arima milik
        # pmdarima — sehingga label metode tidak lagi bergantung pada terpasang/tidaknya
        # pustaka pmdarima di lingkungan.
        "using_pmdarima": False,
        "metode_identifikasi": "Grid-search AIC (d ditetapkan via uji ADF)",
        "peringatan": None,
        "daftar_pengembang": daftar_pengembang,
        "pengembang_aktif": pengembang if pengembang in daftar_pengembang else None,
        "forecast_steps": forecast_steps,
    }