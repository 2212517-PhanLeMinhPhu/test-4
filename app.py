import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px
import io

# ==============================================================================
# --- CẤU HÌNH TRANG & THÔNG SỐ TỐI ƯU ---
# ==============================================================================
st.set_page_config(page_title="JSON Data Pro (Optimized with VPD Table)", layout="wide", page_icon="🌱")
st.title("🌱 Công cụ Phân tích Dữ liệu Nông Nghiệp & Áp Suất VPD")

# Khoảng tối ưu sinh học thực tế của cây trồng để lọc nhiễu cực đoan
KHOANG_TOI_UU = {
    'TEMPKK': (15.0, 32.0),       # Ngưỡng nhiệt độ không khí an toàn (15°C - 32°C)
    'HUMIKK': (50.0, 85.0),       # Ngưỡng độ ẩm không khí lý tưởng (50% - 85%)
    'SOIL_ASKK': (0.0, 200000.0),
    'AS': (0.0, 200000.0),         
    'NHIỆT ĐỘ': (15.0, 32.0),     # Ngưỡng nhiệt độ an toàn (15°C - 32°C)
    'ĐỘ ẨM': (50.0, 85.0),        # Ngưỡng độ ẩm an toàn (50% - 85%)
    'PH': (4.5, 8.5),             # Ngưỡng pH thích hợp cho cây
    'TBPH': (4.5, 8.5),        
    'EC': (0.0, 10000.0),         
    'TBEC': (0.0, 10000.0),       
    'N': (0.0, 2000.0),       
    'P': (0.0, 2000.0),            
    'K': (0.0, 2000.0),              
    'VPD': (0.4, 1.6)             # Khoảng áp suất VPD lý tưởng (kPa)
}

PATTERN_DATETIME = re.compile(r'(\d{2}-\d{2}-\d{2})/([-+]?\d*\.?\d+)')
PATTERN_NUMBER = re.compile(r'[-+]?\d*\.?\d+')

# ==============================================================================
# 1. CÁC HÀM XỬ LÝ LÕI (CÓ CACHE)
# ==============================================================================
@st.cache_data(show_spinner=False)
def normalize_keys(data):
    if isinstance(data, list):
        return [normalize_keys(item) for item in data]
    elif isinstance(data, dict):
        return {str(k).strip().lower(): normalize_keys(v) for k, v in data.items()}
    return data

@st.cache_data(show_spinner=False)
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '.')
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

@st.cache_data(show_spinner=False)
def load_and_process_data(file_bytes):
    try:
        raw_data = json.loads(file_bytes)
    except json.JSONDecodeError:
        raise ValueError("File tải lên không đúng định dạng JSON hợp lệ.")
        
    if isinstance(raw_data, dict):
        raw_data = [raw_data]
    
    clean_json = normalize_keys(raw_data)
    df = pd.DataFrame([flatten_json(row) for row in clean_json])
    df = df.dropna(axis=1, how='all').loc[:, ~df.columns.duplicated()]
    df.replace("", np.nan, inplace=True)
    
    time_col = next((col for col in df.columns if 'time' in col.lower() or 'thời gian' in col.lower()), None)
    if time_col:
        mask = df[time_col].notna()
        df.loc[mask, '_parsed_time'] = pd.to_datetime(
            df.loc[mask, time_col].astype(str).str.replace('-', ':').str.replace(':', '-', 2),
            errors='coerce'
        )
    return df, time_col

# ==============================================================================
# 2. CÁC HÀM TIỆN ÍCH CHO BIỂU ĐỒ & BỘ LỌC
# ==============================================================================
def render_date_filter(min_date, max_date, key_prefix):
    if not min_date: return None, None
    mode = st.radio("⏳ Kiểu lọc thời gian:", ["Tùy chọn", "Theo Tuần (7 ngày)", "Theo Tháng", "Theo Quý"], horizontal=True, key=f"mode_{key_prefix}")
    if mode == "Tùy chọn":
        sel_date = st.date_input("📅 Chọn khoảng ngày:", value=(min_date, max_date), min_value=min_date, max_value=max_date, key=f"date_{key_prefix}")
        start_d = sel_date[0] if sel_date else None
        end_d = sel_date[1] if sel_date and len(sel_date) == 2 else start_d
    else:
        start_d = st.date_input("📅 Chọn ngày bắt đầu:", value=min_date, min_value=min_date, max_value=max_date, key=f"start_{key_prefix}")
        if start_d:
            if mode == "Theo Tuần (7 ngày)": end_d = (pd.to_datetime(start_d) + pd.Timedelta(days=6)).date()
            elif mode == "Theo Tháng": end_d = (pd.to_datetime(start_d) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
            elif mode == "Theo Quý": end_d = (pd.to_datetime(start_d) + pd.DateOffset(months=3) - pd.Timedelta(days=1)).date()
            if end_d > max_date: end_d = max_date
            st.success(f"🎯 Sẽ lọc từ: **{start_d.strftime('%d/%m/%Y')}** đến **{end_d.strftime('%d/%m/%Y')}**")
        else: end_d = None
    return start_d, end_d

@st.cache_data(show_spinner=False)
def extract_sensor_data(df, selected_cols):
    has_vpd = any(c.upper() == 'VPD' for c in selected_cols)
    cols_to_run = list(selected_cols)
    if has_vpd:
        t_col = next((c for c in df.columns if c.upper() in ['NHIỆT ĐỘ', 'TEMPKK']), None)
        rh_col = next((c for c in df.columns if c.upper() in ['ĐỘ ẨM', 'HUMIKK']), None)
        if t_col and t_col not in cols_to_run: cols_to_run.append(t_col)
        if rh_col and rh_col not in cols_to_run: cols_to_run.append(rh_col)
        
    actual_cols = [c for c in cols_to_run if c.upper() != 'VPD']
    records = []
    cols_to_extract = ['_parsed_time'] + actual_cols
    working_df = df[cols_to_extract].dropna(subset=['_parsed_time'])
    
    for row in working_df.itertuples(index=False):
        main_time = row[0]
        date_str = main_time.strftime('%Y-%m-%d')
        for i, col_name in enumerate(actual_cols, start=1):
            val = str(row[i]).strip()
            if not val or val.lower() == 'nan': continue
            col_upper = col_name.upper()
            
            def process_val(v_str):
                v = float(v_str)
                if col_upper in ['PH', 'TBPH'] and v > 14: return v / 100.0
                if col_upper in ['NHIỆT ĐỘ', 'TEMPKK'] and v > 100: return v / 10.0
                return v
                
            matches = PATTERN_DATETIME.findall(val)
            if matches:
                for t_str, v_str in matches:
                    try:
                        full_t_str = f"{date_str} {t_str.replace('-', ':')}"
                        records.append({'TG': pd.to_datetime(full_t_str), 'Giá trị': process_val(v_str), 'Chỉ số': col_upper})
                    except Exception: pass
            else:
                num_match = PATTERN_NUMBER.search(val)
                if num_match:
                    try: records.append({'TG': main_time, 'Giá trị': process_val(num_match.group()), 'Chỉ số': col_upper})
                    except Exception: pass
                        
    if not records: return pd.DataFrame(columns=['TG', 'Giá trị', 'Chỉ số'])
    res_df = pd.DataFrame(records)
    
    # THUẬT TOÁN TỰ ĐỘNG TÍNH TOÁN ÁP SUẤT VPD THEO THỜI GIAN
    if has_vpd:
        t_key = next((c.upper() for c in actual_cols if c.upper() in ['NHIỆT ĐỘ', 'TEMPKK']), None)
        rh_key = next((c.upper() for c in actual_cols if c.upper() in ['ĐỘ ẨM', 'HUMIKK']), None)
        if t_key and rh_key:
            pivot_df = res_df[res_df['Chỉ số'].isin([t_key, rh_key])].pivot_table(index='TG', columns='Chỉ số', values='Giá trị', aggfunc='mean').reset_index()
            if t_key in pivot_df.columns and rh_key in pivot_df.columns:
                T = pivot_df[t_key]
