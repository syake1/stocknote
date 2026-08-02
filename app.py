import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="銘柄分析ノート PRO (逆張り特化)", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+1p:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'M PLUS 1p', sans-serif; }
div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1E88E5; }
div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #666; }
.price-panel { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #E91E63; margin-bottom: 20px;}
.price-label { font-size: 0.9rem; color: #555; margin-bottom: 5px;}
.price-value { font-size: 1.5rem; font-weight: bold; color: #333;}
.buy-target { color: #E91E63 !important; }
.sell-target { color: #4CAF50 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📋 銘柄分析ノート PRO")
st.caption("最新のテクニカル分析とファンダメンタルズ情報を網羅する逆張り特化型ツール")

@st.cache_data(ttl=3600)
def scrape_kabutan(code):
    info = {"name": "", "sector": "", "summary": ""}
    url = f"https://kabutan.jp/stock/?code={code}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            h2 = soup.find('h2', id='stockinfo_i1')
            if h2: info['name'] = h2.get_text().split(' ')[-1]
            sector_a = soup.select('div.company_block a')
            if sector_a: info['sector'] = sector_a[0].get_text()
            summary_div = soup.find('div', class_='company_block')
            if summary_div:
                text = summary_div.get_text()
                info['summary'] = ' '.join(text.split()).replace('特色:', '\n【特色】').replace('連結事業:', '\n【連結事業】')
    except Exception:
        pass
    return info

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    macd = data['Close'].ewm(span=fast, adjust=False).mean() - data['Close'].ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=signal, adjust=False).mean()

def calculate_bb(data, window=25, num_std=2):
    mid = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default

def fmt_pct(x, digits=1): return "—" if x is None or np.isnan(x) else f"{x*100:.{digits}f}%"
def fmt_num(x, digits=1): return "—" if x is None or np.isnan(x) else f"{x:,.{digits}f}"

def create_candlestick_chart(hist):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='株価', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA25'], line=dict(color='#FFA726', width=1.5), name='25日線'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA75'], line=dict(color='#42A5F5', width=1.5), name='75日線'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), name='+2σ'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), name='-2σ', fill='tonexty', fillcolor='rgba(200,200,200,0.05)'))
    fig.update_layout(template='plotly_white', margin=dict(l=20, r=20, t=30, b=20), height=500, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_radar_chart(scores):
    categories = ['財務健全性', '収益性', '割安性', '安定性', '逆張りチャンス']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[scores['financial'], scores['profitability'], scores['value'], scores['stability'], scores['momentum']],
        theta=categories,
        fill='toself',
        name='企業スコア',
        line=dict(color='#E91E63'),
        fillcolor='rgba(233, 30, 99, 0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=50, r=50, t=30, b=30),
        height=320
    )
    return fig

def calc_score(val, v_min, v_max, reverse=False):
    if val is None or np.isnan(val): return 5
    if reverse:
        s = 10 - (val - v_min) / (v_max - v_min) * 10
    else:
        s = (val - v_min) / (v_max - v_min) * 10
    return max(0, min(10, s))

def analyze_ticker(code):
    result = {"code": code}
    tk = yf.Ticker(f"{code}.T")
    try: info = tk.info
    except Exception: info = {}

    kb_info = scrape_kabutan(code)
    result["name"]      = kb_info.get('name') or safe_get(info, "shortName") or code
    result["sector"]    = kb_info.get('sector') or safe_get(info, "sector", "—")
    result["summary"]   = kb_info.get('summary') or safe_get(info, "longBusinessSummary", "")
    result["employees"] = safe_get(info, "fullTimeEmployees", "—")
    result["target_price_analyst"] = safe_get(info, "targetMeanPrice")

    try:
        hist = tk.history(period="2y")
    except Exception as e:
        result["error"] = f"Yahooファイナンスからのデータ取得に失敗しました（エラー: {str(e)}）"
        return result

    if hist is None or hist.empty or len(hist) < 60:
        result["error"] = "株価データが十分に取得できませんでした。"
        return result

    hist['MA25']  = hist['Close'].rolling(25).mean()
    hist['MA75']  = hist['Close'].rolling(75).mean()
    hist['MA200'] = hist['Close'].rolling(200).mean()
    hist['RSI']   = calculate_rsi(hist)
    macd, sig     = calculate_macd(hist)
    hist['MACD'], hist['Signal'] = macd, sig
    bb_up, bb_mid, bb_lo = calculate_bb(hist)
    hist['BB_upper'], hist['BB_mid'], hist['BB_lower'] = bb_up, bb_mid, bb_lo

    latest = hist.iloc[-1]
    current_price = float(latest['Close'])
    result["hist"] = hist
    result["current_price"] = current_price
    result["rsi"] = float(latest['RSI'])
    
    # 目標株価シミュレーション（逆張り用）
    buy_target = float(latest['BB_lower'])
    sell_target = float(latest['BB_mid'])
    final_target = result["target_price_analyst"] if result["target_price_analyst"] else float(latest['BB_upper'])
    
    result["sim_buy"] = buy_target if current_price > buy_target else current_price
    result["sim_sell"] = sell_target if sell_target > current_price else current_price * 1.05
    result["sim_target"] = final_target if final_target > current_price else current_price * 1.10

    # スコアリング（10点満点）
    per = safe_get(info, "trailingPE")
    pbr = safe_get(info, "priceToBook")
    roe = safe_get(info, "returnOnEquity")
    mcap = safe_get(info, "marketCap")
    
    sc_val = calc_score(per, 5, 30, reverse=True) # 割安性 (PER低いほど高得点)
    sc_prof = calc_score(roe, 0, 0.20) # 収益性 (ROE 0〜20%)
    sc_stab = calc_score(mcap, 1e10, 1e12) # 安定性 (時価総額)
    
    # 逆張りチャンス (RSI低いほど高得点。30以下で10点、70以上で0点)
    sc_mom = calc_score(result["rsi"], 30, 70, reverse=True)
    if current_price <= buy_target * 1.02: 
        sc_mom = 10.0 # ボリンジャー-2σ接近でチャンスMAX
        
    # 財務健全性（簡易的に自己資本比率またはPBR/PERから推測）
    sc_fin = calc_score(1/pbr if pbr else None, 0.2, 2.0)
    
    result["scores"] = {
        "financial": sc_fin, "profitability": sc_prof, 
        "value": sc_val, "stability": sc_stab, "momentum": sc_mom
    }
    
    result["per"], result["pbr"], result["roe"] = per, pbr, roe
    result["volatility"] = float(hist['Close'].pct_change().dropna().tail(60).std() * (252 ** 0.5))
    return result

# ================================================================
# UI
# ================================================================
col_input, col_btn, _ = st.columns([2, 1, 3])
with col_input:
    code = st.text_input("銘柄コードを入力", placeholder="例：7203", max_chars=6)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 分析を実行", type="primary", use_container_width=True)

if run and code:
    with st.spinner(f"「{code}」の最新データを収集中..."):
        r = analyze_ticker(code.strip())

    if "error" in r:
        st.error(r["error"])
    else:
        st.markdown("---")
        
        # ヘッダー領域（左：基本情報と価格シミュレーション、右：レーダーチャート）
        h_col1, h_col2 = st.columns([3, 2])
        
        with h_col1:
            st.subheader(f"🏢 {r['name']} ({r['code']})")
            st.caption(f"**業種:** {r['sector']}")
            
            st.markdown(f"""
            <div class="price-panel">
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <div class="price-label">現在価格</div>
                        <div class="price-value">¥ {r['current_price']:,.0f}</div>
                    </div>
                    <div>
                        <div class="price-label">推奨買い価格 (-2σ等)</div>
                        <div class="price-value buy-target">¥ {r['sim_buy']:,.0f}以下</div>
                    </div>
                    <div>
                        <div class="price-label">予想売り価格 (反発目標)</div>
                        <div class="price-value sell-target">¥ {r['sim_sell']:,.0f}</div>
                    </div>
                    <div>
                        <div class="price-label">最終目標価格 (+2σ等)</div>
                        <div class="price-value">¥ {r['sim_target']:,.0f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("逆張りチャンススコア", f"{r['scores']['momentum']:.1f} / 10")
            c2.metric("現在RSI (14)", f"{r['rsi']:.1f}")
            c3.metric("PER / PBR", f"{fmt_num(r['per'])} / {fmt_num(r['pbr'])}")

        with h_col2:
            st.markdown("#### 企業スコア (逆張り評価)")
            st.plotly_chart(create_radar_chart(r['scores']), use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")
        st.markdown("### 📈 テクニカルチャート (日足)")
        st.plotly_chart(create_candlestick_chart(r["hist"].tail(150)), use_container_width=True)

        st.markdown("---")
        tab1, tab2 = st.tabs(["📋 事業概要", "💡 テクニカル詳細"])
        with tab1:
            st.markdown("#### 事業内容・特色")
            st.write(r["summary"] if r["summary"] else "情報が取得できませんでした。")
        with tab2:
            st.write("ボリンジャーバンドの-2σ（青点線の下限）や、RSIが30を下回るタイミングが逆張りの狙い目となります。")
            
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}  |  提供元: 株探 (kabutan.jp) / Yahoo Finance")

elif run and not code:
    st.warning("⚠️ 銘柄コードを入力してください。")
