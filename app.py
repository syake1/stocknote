import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="銘柄分析ノート PRO", layout="wide", initial_sidebar_state="collapsed")

# プロフェッショナル向けカスタムCSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=M+PLUS+1p:wght@400;700&display=swap');
html, body, [class*="css"] {
    font-family: 'M PLUS 1p', sans-serif;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1E88E5;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.9rem;
    color: #666;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📋 銘柄分析ノート PRO")
st.caption("最新のテクニカル分析とファンダメンタルズ情報を網羅するプロフェッショナル向けツール")

# ================================================================
# データ取得・スクレイピング
# ================================================================
@st.cache_data(ttl=3600)
def scrape_kabutan(code):
    """株探から日本語の会社情報と業種を取得する"""
    info = {"name": "", "sector": "", "summary": ""}
    url = f"https://kabutan.jp/stock/?code={code}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # 会社名
            h2 = soup.find('h2', id='stockinfo_i1')
            if h2:
                info['name'] = h2.get_text().split(' ')[-1]
                
            # 業種・テーマ
            sector_a = soup.select('div.company_block a')
            if sector_a:
                info['sector'] = sector_a[0].get_text()
                
            # 事業内容
            summary_div = soup.find('div', class_='company_block')
            if summary_div:
                text = summary_div.get_text()
                # 余分な改行やスペースを削除し、大まかにテキストを抽出
                info['summary'] = ' '.join(text.split()).replace('特色:', '\n【特色】').replace('連結事業:', '\n【連結事業】')
                
    except Exception as e:
        pass
    return info

# ================================================================
# 指標計算
# ================================================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calculate_bb(data, window=20, num_std=2):
    mid = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default

def fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{digits}f}%"

def fmt_num(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{digits}f}"

# ================================================================
# チャート生成 (Plotly)
# ================================================================
def create_candlestick_chart(hist):
    fig = go.Figure()
    # ローソク足
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
        name='株価', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    # 移動平均線
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA25'], line=dict(color='#FFA726', width=1.5), name='25日線'))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA75'], line=dict(color='#42A5F5', width=1.5), name='75日線'))
    # ボリンジャーバンド
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), name='+2σ'))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['BB_lower'], line=dict(color='rgba(200,200,200,0.5)', width=1, dash='dash'), 
        name='-2σ', fill='tonexty', fillcolor='rgba(200,200,200,0.05)'
    ))
    fig.update_layout(
        template='plotly_white', 
        margin=dict(l=20, r=20, t=30, b=20), 
        height=500, 
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ================================================================
# メイン分析処理
# ================================================================
def analyze_ticker(code):
    result = {"code": code}
    tk = yf.Ticker(f"{code}.T")

    # 基本情報の取得 (yfinance)
    try:
        info = tk.info
    except Exception:
        info = {}

    # 日本語情報への置き換え (kabutanから取得)
    kb_info = scrape_kabutan(code)
    
    result["name"]      = kb_info.get('name') or safe_get(info, "shortName") or code
    result["sector"]    = kb_info.get('sector') or safe_get(info, "sector", "—")
    result["summary"]   = kb_info.get('summary') or safe_get(info, "longBusinessSummary", "")
    result["employees"] = safe_get(info, "fullTimeEmployees", "—")

    # --- 株価・テクニカル用の日足データ ---
    try:
        hist = tk.history(period="2y")
    except Exception as e:
        result["error"] = f"Yahooファイナンスからのデータ取得に失敗しました。少し時間をおいて再度お試しください（エラー詳細: {str(e)}）"
        return result

    if hist is None or hist.empty or len(hist) < 60:
        result["error"] = "株価データが十分に取得できませんでした。銘柄コードが正しいか確認してください。"
        return result

    hist['MA25']  = hist['Close'].rolling(25).mean()
    hist['MA75']  = hist['Close'].rolling(75).mean()
    hist['MA200'] = hist['Close'].rolling(200).mean()
    hist['RSI']   = calculate_rsi(hist)
    macd, sig     = calculate_macd(hist)
    hist['MACD']  = macd
    hist['Signal']= sig
    bb_up, bb_mid, bb_lo = calculate_bb(hist)
    hist['BB_upper'] = bb_up
    hist['BB_mid']   = bb_mid
    hist['BB_lower'] = bb_lo

    latest = hist.iloc[-1]
    current_price = float(latest['Close'])
    result["hist"] = hist
    result["current_price"] = current_price
    result["rsi"]  = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
    result["ma25"]  = float(latest['MA25'])  if not pd.isna(latest['MA25'])  else None
    result["ma75"]  = float(latest['MA75'])  if not pd.isna(latest['MA75'])  else None
    result["ma200"] = float(latest['MA200']) if not pd.isna(latest['MA200']) else None
    result["macd_val"] = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
    result["macd_sig"] = float(latest['Signal']) if not pd.isna(latest['Signal']) else None
    result["bb_pos"] = None
    if not pd.isna(latest['BB_upper']) and not pd.isna(latest['BB_lower']):
        rng = float(latest['BB_upper']) - float(latest['BB_lower'])
        if rng > 0:
            result["bb_pos"] = (current_price - float(latest['BB_lower'])) / rng * 100

    # 52週レンジ・ボラティリティ
    last_252 = hist.tail(252)
    result["week52_high"] = float(last_252['High'].max())
    result["week52_low"]  = float(last_252['Low'].min())
    daily_ret = hist['Close'].pct_change().dropna()
    result["volatility"] = float(daily_ret.tail(60).std() * (252 ** 0.5))

    # --- 財務・バリュエーション ---
    result["per"]  = safe_get(info, "trailingPE")
    result["pbr"]  = safe_get(info, "priceToBook")
    result["roe"]  = safe_get(info, "returnOnEquity")
    result["eps"]  = safe_get(info, "trailingEps")
    result["dividend_yield"] = safe_get(info, "dividendYield")
    result["market_cap"] = safe_get(info, "marketCap")
    
    # --- 業績推移（売上・純利益） ---
    growth_df = None
    try:
        fin = tk.financials
        if fin is not None and not fin.empty:
            rev_row = profit_row = None
            for idx in fin.index:
                if 'Total Revenue' in str(idx): rev_row = fin.loc[idx]
                if str(idx) == 'Net Income': profit_row = fin.loc[idx]
            if rev_row is not None:
                growth_df = pd.DataFrame({"売上高": rev_row})
                if profit_row is not None: growth_df["純利益"] = profit_row
                growth_df = growth_df.iloc[:, ::1].sort_index(axis=0)
                growth_df.index = [str(d)[:10] for d in growth_df.index]
    except Exception:
        pass
    result["growth_df"] = growth_df

    result["revenue_growth"] = result["profit_growth"] = None
    if growth_df is not None and len(growth_df) >= 2:
        try:
            rev_vals = growth_df["売上高"].dropna().values
            if len(rev_vals) >= 2: result["revenue_growth"] = (rev_vals[-1] - rev_vals[-2]) / abs(rev_vals[-2])
            if "純利益" in growth_df.columns:
                pr_vals = growth_df["純利益"].dropna().values
                if len(pr_vals) >= 2: result["profit_growth"] = (pr_vals[-1] - pr_vals[-2]) / abs(pr_vals[-2])
        except Exception:
            pass

    # --- 総合判定 ---
    score = 0
    reasons = []
    if result["ma200"] and current_price > result["ma200"]:
        score += 1; reasons.append("200日線の上（中期トレンド良好）")
    elif result["ma200"]:
        score -= 1; reasons.append("200日線の下（中期トレンド軟調）")

    if result["rsi"]:
        if result["rsi"] < 35: score += 1; reasons.append(f"RSI {result['rsi']:.0f}（売られすぎ圏）")
        elif result["rsi"] > 70: score -= 1; reasons.append(f"RSI {result['rsi']:.0f}（買われすぎ圏）")

    if result["macd_val"] is not None and result["macd_sig"] is not None:
        if result["macd_val"] > result["macd_sig"]: score += 1; reasons.append("MACDが上向き")
        else: score -= 1; reasons.append("MACDが下向き")

    if score >= 2: judgement = "🔥 買い推奨"
    elif score >= 1: judgement = "🙂 やや買い"
    elif score <= -2: judgement = "🔻 売り推奨"
    elif score <= -1: judgement = "🤔 やや弱気"
    else: judgement = "⏳ 中立・様子見"

    result["score"] = score
    result["judgement"] = judgement
    result["reasons"] = reasons

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
        
        # ヘッダー
        st.subheader(f"🏢 {r['name']} ({r['code']})")
        st.caption(f"**業種:** {r['sector']}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在値", f"¥ {r['current_price']:,.0f}")
        c2.metric("総合判定", r["judgement"])
        c3.metric("52週レンジ", f"{r['week52_low']:,.0f} - {r['week52_high']:,.0f}")
        c4.metric("PER / PBR", f"{fmt_num(r['per'])} 倍 / {fmt_num(r['pbr'])} 倍")

        # テクニカルチャート
        st.markdown("### 📈 テクニカルチャート (日足)")
        fig = create_candlestick_chart(r["hist"].tail(150))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 ファンダメンタルズ・業績", "💡 テクニカル詳細", "📋 事業概要"])

        with tab1:
            st.markdown("#### 企業価値・財務")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("時価総額", f"¥ {r['market_cap']/1e8:,.0f} 億円" if r["market_cap"] else "—")
            fc2.metric("ROE (自己資本利益率)", fmt_pct(r["roe"]))
            fc3.metric("配当利回り", fmt_pct(r["dividend_yield"]))
            fc4.metric("従業員数", f"{r['employees']:,} 人" if r["employees"] != "—" else "—")
            
            st.markdown("#### 直近の業績推移")
            if r["growth_df"] is not None and not r["growth_df"].empty:
                st.dataframe(r["growth_df"], use_container_width=True)
            else:
                st.info("業績データを取得できませんでした。")

        with tab2:
            st.markdown("#### テクニカル判定根拠")
            if r["reasons"]:
                for reason in r["reasons"]:
                    st.write(f"✅ {reason}")
            else:
                st.write("明確なシグナルはありません。")
                
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("RSI (14)", f"{fmt_num(r['rsi'])}")
            tc2.metric("MACD トレンド", "上向き ⤴" if (r["macd_val"] or 0) > (r["macd_sig"] or 0) else "下向き ⤵")
            tc3.metric("ボラティリティ", fmt_pct(r["volatility"]))

        with tab3:
            st.markdown("#### 事業内容・特色")
            if r["summary"]:
                st.write(r["summary"])
            else:
                st.info("この銘柄の事業概要テキストは取得できませんでした。")

        st.markdown("---")
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}  |  提供元: 株探 (kabutan.jp) / Yahoo Finance")

elif run and not code:
    st.warning("⚠️ 銘柄コードを入力してください。")
