import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px

# ==============================================================================
# --- CẤU HÌNH TRANG & THÔNG SỐ TỐI ƯU ---
# ==============================================================================
st.set_page_config(page_title="🌱 JSON Data Pro (Khuyến Nghị Sinh Học)", layout="wide", page_icon="🌱")
st.title("🌱 Công cụ Phân tích Dữ liệu Nông Nghiệp & Áp Suất VPD")

# Khoảng tối ưu sinh học thực tế tiêu chuẩn của cây trồng (Có thể điều chỉnh tùy loại cây)
KHOANG_TOI_UU = {
    'TEMPKK': (15.0, 32.0),       # Ngưỡng nhiệt độ không khí lý tưởng (15°C - 32°C)
    'HUMIKK': (50.0, 85.0),       # Ngưỡng độ ẩm không khí lý tưởng (50% - 85%)
    'SOIL_ASKK': (0.0, 200000.0),
    'AS': (0.0, 200000.0),         
    'NHIỆT ĐỘ': (15.0, 32.0),     
    'ĐỘ ẨM': (50.0, 85.0),        
    'PH': (4.5, 8.5),             
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
            for a in x: flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            for i, a in enumerate(x): flatten(a, name + str(i) + '.')
        else: out[name[:-1]] = x
    flatten(y)
    return out

@st.cache_data(show_spinner=False)
def load_and_process_data(file_bytes):
    try:
        raw_data = json.loads(file_bytes)
    except json.JSONDecodeError:
        raise ValueError("File tải lên không đúng định dạng JSON hợp lệ.")
        
    if isinstance(raw_data, dict): raw_data = [raw_data]
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
    
    if has_vpd:
        t_key = next((c.upper() for c in actual_cols if c.upper() in ['NHIỆT ĐỘ', 'TEMPKK']), None)
        rh_key = next((c.upper() for c in actual_cols if c.upper() in ['ĐỘ ẨM', 'HUMIKK']), None)
        if t_key and rh_key:
            pivot_df = res_df[res_df['Chỉ số'].isin([t_key, rh_key])].pivot_table(index='TG', columns='Chỉ số', values='Giá trị', aggfunc='mean').reset_index()
            if t_key in pivot_df.columns and rh_key in pivot_df.columns:
                T = pivot_df[t_key]
                RH = pivot_df[rh_key]
                vp_sat = 0.61078 * np.exp((17.27 * T) / (T + 237.3))
                vpd_values = vp_sat * (1.0 - (RH / 100.0))
                vpd_df = pd.DataFrame({'TG': pivot_df['TG'], 'Giá trị': vpd_values, 'Chỉ số': 'VPD'}).dropna()
                res_df = pd.concat([res_df, vpd_df], ignore_index=True)
                
    user_indices = [c.upper() for c in selected_cols]
    return res_df[res_df['Chỉ số'].isin(user_indices)].reset_index(drop=True)

def generate_chart(df, title, is_multi=False):
    num_points = len(df)
    use_webgl = 'webgl' if num_points > 1000 else 'svg'
    show_markers = num_points <= 500
    if is_multi:
        fig = px.line(df, x='TG', y='Giá trị', color='Chỉ số', markers=show_markers, render_mode=use_webgl, color_discrete_sequence=px.colors.qualitative.Set1)
    else:
        fig = px.line(df, x='TG', y='Giá trị', markers=show_markers, render_mode=use_webgl)
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="Thời gian", yaxis_title="Giá trị", hovermode="x unified", dragmode='pan', xaxis=dict(rangeslider=dict(visible=False), type="date"))
    return fig, num_points

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
uploaded_file = st.file_uploader("📥 Bước 1: Hãy tải lên tệp tin dữ liệu JSON của bạn tại đây:", type=['json'])

st.markdown("---")
# Đã cấu hình cố định 5 Tab hiển thị (Tích hợp thêm Tab 5 phân tích chuyên sâu Nhiệt/Ẩm phù hợp)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗂️ Bảng dữ liệu thô", "📈 Biểu đồ Đơn", "📊 Biểu đồ Lồng nhau", "📊 Bảng tra cứu VPD", "💡 Khuyến Nghị Nhiệt/Ẩm"])

if uploaded_file is None:
    with tab1: st.info("👋 Vui lòng tải file JSON ở phía trên lên để xem bảng dữ liệu chi tiết.")
    with tab2: st.info("👋 Vui lòng tải file JSON ở phía trên lên để thiết lập đồ thị đơn lẻ.")
    with tab3: st.info("👋 Vui lòng tải file JSON ở phía trên lên để thiết lập đồ thị đối chiếu lồng nhau.")
    with tab4: st.info("👋 Vui lòng tải file JSON ở phía trên lên để hệ thống trích xuất bảng tra cứu Áp suất VPD.")
    with tab5: st.info("👋 Vui lòng tải file JSON ở phía trên lên để phân tích khoảng Nhiệt độ, Độ ẩm phù hợp.")
else:
    try:
        with st.spinner("Đang xử lý dữ liệu siêu tốc..."):
            file_bytes = uploaded_file.getvalue().decode("utf-8")
            df, time_col = load_and_process_data(file_bytes)

        # Cấu hình thanh bên bộ lọc (Sidebar)
        st.sidebar.markdown("### 🔍 BỘ LỌC TÙY CHỈNH")
        filterable_cols = [col for col in df.columns if col not in ['_parsed_time']]
        selected_key = st.sidebar.selectbox("Chọn trường dữ liệu muốn lọc:", options=["-- Không lọc --"] + filterable_cols)
        if selected_key != "-- Không lọc --":
            unique_values = df[selected_key].dropna().astype(str).unique()
            selected_value = st.sidebar.selectbox(f"Chọn giá trị cho '{selected_key.upper()}':", options=["Tất cả"] + list(unique_values))
            if selected_value != "Tất cả":
                df = df[df[selected_key].astype(str) == selected_value].reset_index(drop=True)

        exclude = [time_col, 'stt', 'tên khu', 'trạng thái', 'phương thức hoạt động', 'người điều khiển', '_parsed_time']
        numeric_options = [c for c in df.columns if c not in exclude and '_id' not in c]

        # Kiểm tra sự tồn tại của các trường Nhiệt độ / Độ ẩm trong file
        t_col_name = next((c for c in df.columns if c.upper() in ['NHIỆT ĐỘ', 'TEMPKK']), None)
        rh_col_name = next((c for c in df.columns if c.upper() in ['ĐỘ ẨM', 'HUMIKK']), None)
        
        has_t = t_col_name is not None
        has_rh = rh_col_name is not None
        if has_t and has_rh:
            numeric_options.append('VPD')

        min_d, max_d = None, None
        if '_parsed_time' in df.columns:
            valid_ts = df['_parsed_time'].dropna()
            if not valid_ts.empty:
                min_d, max_d = valid_ts.min().date(), valid_ts.max().date()

        # TAB 1, 2, 3, 4 hoạt động ổn định...
        with tab1:
            st.subheader("🌾 Bảng dữ liệu chi tiết")
            display_df = df.drop(columns=['_parsed_time'], errors='ignore').fillna("")
            st.dataframe(display_df, use_container_width=True)

        with tab2:
            st.write("⚙️ Thiết lập biểu đồ đơn lẻ")
            col1, col2 = st.columns([1, 2])
            with col1:
                start_d_2, end_d_2 = render_date_filter(min_d, max_d, "tab2")
                filter_data_2 = st.checkbox("✅ Lọc Sạch Dữ Liệu", value=True, key="filter_tab2")
            with col2:
                selected_keys_2 = [k for k in numeric_options if st.checkbox(k.upper(), key=f"c_tab2_{k}")]
            if st.button("🚀 TẠO BIỂU ĐỒ ĐƠN", type="primary"):
                if selected_keys_2 and start_d_2 and end_d_2:
                    mask = (df['_parsed_time'].dt.date >= start_d_2) & (df['_parsed_time'].dt.date <= end_d_2)
                    chart_df = extract_sensor_data(df[mask], selected_keys_2)
                    for col in selected_keys_2:
                        sub_df = chart_df[chart_df['Chỉ số'] == col.upper()]
                        if filter_data_2 and col.upper() in KHOANG_TOI_UU:
                            sub_df = sub_df[(sub_df['Giá trị'] >= KHOANG_TOI_UU[col.upper()][0]) & (sub_df['Giá trị'] <= KHOANG_TOI_UU[col.upper()][1])]
                        plot_data = sub_df.groupby('TG')['Giá trị'].mean().reset_index().sort_values('TG')
                        fig, pts = generate_chart(plot_data, f"Chỉ số: {col.upper()}")
                        st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.write("⚙️ Thiết lập biểu đồ lồng nhau")
            col1_m, col2_m = st.columns([1, 2])
            with col1_m: selected_keys_3 = [k for k in numeric_options if st.checkbox(k.upper(), key=f"c_multi_{k}")]
            with col2_m:
                start_d_3, end_d_3 = render_date_filter(min_d, max_d, "tab3")
                filter_data_3 = st.checkbox("✅ Lọc Sạch Dữ Liệu", value=True, key="filter_tab3")
            if st.button("🚀 TẠO BIỂU ĐỒ ĐỐI CHIẾU", type="primary"):
                if len(selected_keys_3) >= 2 and start_d_3 and end_d_3:
                    mask = (df['_parsed_time'].dt.date >= start_d_3) & (df['_parsed_time'].dt.date <= end_d_3)
                    multi_chart_df = extract_sensor_data(df[mask], selected_keys_3)
                    clean_dfs = []
                    for col in selected_keys_3:
                        sub_df = multi_chart_df[multi_chart_df['Chỉ số'] == col.upper()]
                        if filter_data_3 and col.upper() in KHOANG_TOI_UU:
                            sub_df = sub_df[(sub_df['Giá trị'] >= KHOANG_TOI_UU[col.upper()][0]) & (sub_df['Giá trị'] <= KHOANG_TOI_UU[col.upper()][1])]
                        clean_dfs.append(sub_df)
                    plot_data = pd.concat(clean_dfs).groupby(['TG', 'Chỉ số'])['Giá trị'].mean().reset_index().sort_values('TG')
                    fig, pts = generate_chart(plot_data, "Biểu đồ Đối chiếu", is_multi=True)
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("📊 Bảng dữ liệu Áp suất hơi nước (VPD) chi tiết")
            if not (has_t and has_rh):
                st.error("❌ Không thể lập bảng VPD! File thiếu dữ liệu Nhiệt độ hoặc Độ ẩm.")
            else:
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1:
                    start_d_4, end_d_4 = render_date_filter(min_d, max_d, "tab4")
                    filter_data_4 = st.checkbox("🎯 Chỉ hiện dải VPD an toàn (0.4 - 1.6 kPa)", value=False, key="filter_tab4")
                with col_t2: st.info("💡 Chỉ số áp suất VPD lý tưởng từ **0.4 kPa đến 1.6 kPa**.")
                
                if start_d_4 and end_d_4:
                    mask_4 = (df['_parsed_time'].dt.date >= start_d_4) & (df['_parsed_time'].dt.date <= end_d_4)
                    raw_extracted = extract_sensor_data(df[mask_4], ['VPD'])
                    vpd_only = raw_extracted[raw_extracted['Chỉ số'] == 'VPD'].copy()
                    if not vpd_only.empty:
                        vpd_only['Ngày'] = vpd_only['TG'].dt.strftime('%d/%m/%Y')
                        vpd_only['Giờ'] = vpd_only['TG'].dt.strftime('%H:%M:%S')
                        vpd_only['Áp suất VPD (kPa)'] = vpd_only['Giá trị'].round(3)
                        if filter_data_4:
                            vpd_only = vpd_only[(vpd_only['Áp suất VPD (kPa)'] >= 0.4) & (vpd_only['Áp suất VPD (kPa)'] <= 1.6)]
                        final_table = vpd_only[['Ngày', 'Giờ', 'Áp suất VPD (kPa)']].reset_index(drop=True)
                        st.dataframe(final_table, use_container_width=True)

        # ==============================================================================
        # TAB 5: VIẾT THÊM - TỰ ĐỘNG TÌM VÀ ĐÁNH GIÁ THÔNG SỐ PHÙ HỢP (TRÁNH QUÁ CAO/THẤP)
        # ==============================================================================
        with tab5:
            st.subheader("💡 Chẩn Đoán & Khuyến Nghị Chỉ Số Môi Trường Phù Hợp")
            st.markdown("Hệ thống kiểm tra tự động dải thông số hiện tại dựa trên ngưỡng tối ưu sinh học thực tế.")

            if not (has_t or has_rh):
                st.warning("⚠️ File dữ liệu không chứa thông tin cơ bản về Nhiệt độ hoặc Độ ẩm để phân tích.")
            else:
                col_sel1, col_sel2 = st.columns([1, 2])
                with col_sel1:
                    start_d_5, end_d_5 = render_date_filter(min_d, max_d, "tab5")
                with col_sel2:
                    st.markdown("""
                    **Ngưỡng cài đặt an toàn (Có thể tùy biến trong code):**
                    * **Nhiệt độ tối ưu:** `15.0°C` đến `32.0°C` (Dưới 15°C: Cây thun rễ; Trên 32°C: Đóng khí khổng, cháy lá).
                    * **Độ ẩm tối ưu:** `50.0%` đến `85.0%` (Dưới 50%: Thoát nước quá nhanh gây héo; Trên 85%: Dễ sinh nấm bệnh hại).
                    """)

                if start_d_5 and end_d_5:
                    mask_5 = (df['_parsed_time'].dt.date >= start_d_5) & (df['_parsed_time'].dt.date <= end_d_5)
                    active_df = df[mask_5]

                    # Mảng chứa các cặp cột để quét tự động
                    targets = []
                    if has_t: targets.append((t_col_name, 'NHIỆT ĐỘ', '°C', 15.0, 32.0))
                    if has_rh: targets.append((rh_col_name, 'ĐỘ ẨM', '%', 50.0, 85.0))

                    for raw_col, label, unit, low_bound, high_bound in targets:
                        # Trích xuất chuỗi/số thô ra mảng số thực để tính toán thống kê chính xác
                        extracted = extract_sensor_data(active_df, [raw_col])
                        extracted_vals = extracted[extracted['Chỉ số'] == raw_col.upper()]['Giá trị']

                        if not extracted_vals.empty:
                            st.markdown(f"### 📊 Phân tích chuyên sâu về: **{label}**")
                            
                            # Tính toán phân phối số liệu thực tế
                            total_pts = len(extracted_vals)
                            avg_val = extracted_vals.mean()
                            max_val = extracted_vals.max()
                            min_val = extracted_vals.min()

                            low_pts = len(extracted_vals[extracted_vals < low_bound])
                            high_pts = len(extracted_vals[extracted_vals > high_bound])
                            normal_pts = total_pts - low_pts - high_pts

                            pct_low = (low_pts / total_pts) * 100
                            pct_high = (high_pts / total_pts) * 100
                            pct_normal = (normal_pts / total_pts) * 100

                            # Giao diện hiển thị Thẻ đo lường nhanh (Metrics)
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Trung bình thực tế", f"{avg_val:.1f} {unit}")
                            m2.metric("Thấp nhất ghi nhận", f"{min_val:.1f} {unit}")
                            m3.metric("Cao nhất ghi nhận", f"{max_val:.1f} {unit}")
                            m4.metric("Tổng số điểm quét", f"{total_pts} điểm")

                            # Đưa ra cảnh báo trực quan dựa trên tỷ lệ phần trăm phân bố dữ liệu
                            st.write("**Biểu đồ phân phối trạng thái an toàn sinh học:**")
                            progress_df = pd.DataFrame({
                                'Trạng thái': ['Quá thấp (Cần tăng)', 'Phù hợp (Tối ưu)', 'Quá cao (Cần giảm)'],
                                'Tỷ lệ %': [pct_low, pct_normal, pct_high],
                                'Màu sắc': ['Thấp', 'Tối ưu', 'Cao']
                            })
                            fig_bar = px.bar(progress_df, x='Tỷ lệ %', y='Trạng thái', color='Màu sắc', 
                                             orientation='h', text_auto='.1f',
                                             color_discrete_map={'Thấp': '#FFA07A', 'Tối ưu': '#2ECC71', 'Cao': '#E74C3C'})
                            fig_bar.update_layout(height=180, showlegend=False, yaxis_title="")
                            st.plotly_chart(fig_bar, use_container_width=True)

                            # Đưa ra chẩn đoán bằng văn bản thông minh giúp nông dân hành động
                            if pct_normal >= 75.0:
                                st.success(f"✅ Đánh giá: Môi trường **{label}** rất lý tưởng! Có đến **{pct_normal:.1f}%** thời gian nằm trong dải phù hợp. Hãy tiếp tục duy trì chế độ vận hành này.")
                            else:
                                st.error(f"⚠️ Cảnh báo điều tiết: Chỉ có **{pct_normal:.1f}%** thời gian đạt mức phù hợp. Hãy chú ý kiểm soát hệ thống kỹ thuật:")
                                
                                # Chi tiết giải pháp kỹ thuật cụ thể cho từng trường hợp lỗi
                                details = []
                                if pct_high > 15.0:
                                    if label == 'NHIỆT ĐỘ':
                                        details.append(f"* **Hiện tượng Nhiệt độ Quá cao ({pct_high:.1f}% thời gian):** Cần bật quạt thông gió, kéo lưới cắt nắng hoặc phun sương làm mát mái che.")
                                    else:
                                        details.append(f"* **Hiện tượng Độ ẩm Quá cao ({pct_high:.1f}% thời gian):** Cần ngừng phun sương, mở bạt thông gió đáy/mái hoặc bật quạt đối lưu để giảm nguy cơ nấm bệnh.")
                                
                                if pct_low > 15.0:
                                    if label == 'NHIỆT ĐỘ':
                                        details.append(f"* **Hiện tượng Nhiệt độ Quá thấp ({pct_low:.1f}% thời gian):** Chú ý đóng kín màng nhà kính vào ban đêm, hạn chế thông gió cưỡng bức.")
                                    else:
                                        details.append(f"* **Hiện tượng Độ ẩm Quá thấp ({pct_low:.1f}% thời gian):** Cây có nguy cơ héo rũ, hãy kích hoạt hệ thống phun sương hạt mịn hoặc tưới bù ẩm nền.")
                                
                                if details:
                                    st.markdown("\n".join(details))
                            st.markdown("---")
                        else:
                            st.info(f"Không có số liệu đo đạc cho {label} trong khoảng ngày này.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")
