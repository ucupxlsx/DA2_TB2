import pandas as pd
from flask import Flask, render_template

# Inisialisasi aplikasi Flask
app = Flask(__name__)

def analyze_data():
    """
    Fungsi ini memuat dan menganalisis data untuk perbandingan Year-on-Year (YoY)
    antara 2020 dan 2019.
    """
    try:
        df_mobility = pd.read_csv('datasets/2020_ID_Region_Mobility_Report.csv')
        df_policy = pd.read_csv('datasets/OxCGRT_fullwithnotes_national_2020_v1.csv')
        df_aqi = pd.read_csv('datasets/ispu_dki_all.csv')

    except FileNotFoundError as e:
        print(f"ERROR: File tidak ditemukan! Pastikan file CSV ada di dalam folder 'datasets'. Detail: {e}")
        return get_sample_data()

    df_mobility.columns = df_mobility.columns.str.strip()
    df_policy.columns = df_policy.columns.str.strip()
    df_aqi.columns = df_aqi.columns.str.strip()
    
    # --- PROSES DATA 2020 (MOBILITAS & KEBIJAKAN) ---
    df_mobility = df_mobility[df_mobility['sub_region_1'].str.contains("Jakarta", na=False)].copy()
    df_mobility['date'] = pd.to_datetime(df_mobility['date'])
    df_mobility.set_index('date', inplace=True)
    monthly_mobility = df_mobility[['transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline']].mean(axis=1).resample('ME').mean()
    
    policy_column_name = 'StringencyIndex_WeightedAverage'
    df_policy['Date'] = pd.to_datetime(df_policy['Date'], format='%Y%m%d')
    df_policy.set_index('Date', inplace=True)
    monthly_policy = df_policy[policy_column_name].resample('ME').mean()

    # --- PROSES DATA KUALITAS UDARA UNTUK 2019 & 2020 ---
    date_col_aqi = 'tanggal' if 'tanggal' in df_aqi.columns else 'date'
    df_aqi[date_col_aqi] = pd.to_datetime(df_aqi[date_col_aqi])
    
    df_aqi = df_aqi[df_aqi[date_col_aqi].dt.year.isin([2019, 2020])].copy()
    
    pollutant_cols = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    for col in pollutant_cols:
        if col in df_aqi.columns:
            df_aqi[col] = pd.to_numeric(df_aqi[col], errors='coerce')
    df_aqi['max_pollutant_value'] = df_aqi[pollutant_cols].max(axis=1)
    df_aqi.dropna(subset=['max_pollutant_value'], inplace=True)

    df_aqi['year'] = df_aqi[date_col_aqi].dt.year
    df_aqi['month'] = df_aqi[date_col_aqi].dt.month
    monthly_aqi_raw = df_aqi.groupby(['year', 'month'])['max_pollutant_value'].median()
    
    aqi_yoy = monthly_aqi_raw.unstack(level='year')
    if 2019 in aqi_yoy.columns and 2020 in aqi_yoy.columns:
        aqi_yoy.columns = ['pollution_2019', 'pollution_2020']
    else:
        # Handle jika salah satu tahun tidak ada datanya
        if 2019 not in aqi_yoy.columns:
            aqi_yoy['pollution_2019'] = pd.NA
        if 2020 not in aqi_yoy.columns:
            aqi_yoy['pollution_2020'] = pd.NA
        aqi_yoy = aqi_yoy[['pollution_2019', 'pollution_2020']]


    # --- GABUNGKAN SEMUA DATA ---
    df_final = pd.DataFrame(index=pd.date_range(start='2020-01-01', end='2020-12-31', freq='ME'))
    df_final = df_final.join(monthly_policy.rename('stringency'))
    df_final = df_final.join(monthly_mobility.rename('mobility'))
    df_final = df_final.join(aqi_yoy.reset_index(drop=True).set_index(df_final.index))

    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    df_final.fillna(0, inplace=True)

    # --- HITUNG KPI BERDASARKAN YoY ---
    df_final['yoy_improvement_pct'] = ((df_final['pollution_2020'] - df_final['pollution_2019']) / df_final['pollution_2019']) * 100
    best_yoy_improvement = df_final['yoy_improvement_pct'].min()

    # --- SIAPKAN HASIL AKHIR ---
    return {
        "labels": [d.strftime('%b') for d in df_final.index],
        "stringency_data": list(df_final['stringency'].round(1)),
        "mobility_data": list(df_final['mobility'].round(1)),
        "pollution_data_2020": list(df_final['pollution_2020'].round(1)),
        "pollution_data_2019": list(df_final['pollution_2019'].round(1)),
        "kpi_mobility": f"{df_final['mobility'].min():.0f}",
        "kpi_yoy_pollution": f"{best_yoy_improvement:.0f}" if pd.notna(best_yoy_improvement) else "N/A"
    }

def get_sample_data():
    # Fallback jika terjadi error
    return {
        "labels": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        "stringency_data": [5, 5, 46, 82, 84, 76, 70, 68, 74, 65, 65, 68],
        "mobility_data": [-2, -1, -22, -61, -65, -42, -37, -35, -43, -36, -34, -31],
        "pollution_data_2020": [85, 80, 65, 45, 42, 58, 65, 68, 75, 78, 80, 82],
        "pollution_data_2019": [90, 88, 82, 75, 70, 72, 78, 80, 85, 88, 92, 95],
        "kpi_mobility": "-65",
        "kpi_yoy_pollution": "-40"
    }

@app.route("/")
def index():
    chart_data = analyze_data()
    return render_template("index.html", chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)
