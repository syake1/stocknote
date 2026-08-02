import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
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

# ============================================================
# 事業内容・特色のスクレイピングで無効な文言（JS警告等）を弾くためのブラックリスト
# ============================================================
_INVALID_SUMMARY_PATTERNS = [
    "javascript", "java script", "スクリプト", "cookie", "クッキー",
    "設定を変更する方法", "ブラウザの設定", "有効にしてください", "無効になっています",
    "ページが見つかりません", "アクセスが集中", "only", "403", "404",
]

def _is_valid_summary(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 15:
        return False
    low = t.lower()
    for pat in _INVALID_SUMMARY_PATTERNS:
        if pat.lower() in low:
            return False
    return True


@st.cache_data(ttl=3600)
def scrape_japanese_info(code):
    info = {"name": "", "sector": "", "summary": ""}

    # --- 1. 株探 (kabutan.jp) からの取得 ---
    try:
        url = f"https://kabutan.jp/stock/?code={code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3'
        }
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        res = requests.get(url, headers=headers, timeout=8, verify=False)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')

            # 銘柄名（複数パターンでトライ）
            h2 = soup.find('h2', id='stockinfo_i1')
            if h2:
                info['name'] = h2.get_text().split(' ')[-1].strip()
            if not info['name']:
                title = soup.find('title')
                if title:
                    # 例：「トヨタ自動車【7203】株の基本情報｜株探」
                    m = re.match(r'^(.*?)【', title.get_text())
                    if m:
                        info['name'] = m.group(1).strip()

            # 業種
            sector_a = soup.select('div.company_block a')
            if sector_a:
                info['sector'] = sector_a[0].get_text().strip()

            # 特色・連結事業（company_blockが見つからない場合はページ全文から正規表現で抽出）
            summary_text = ""
            summary_div = soup.find('div', class_='company_block')
            if summary_div:
                text = summary_div.get_text()
                summary_text = ' '.join(text.split())
                summary_text = summary_text.replace('特色:', '\n【特色】').replace('連結事業:', '\n【連結事業】')

            if not _is_valid_summary(summary_text):
                # フォールバック：ページ全体のテキストから「特色」「連結事業」を含む箇所を正規表現で拾う
                full_text = ' '.join(soup.get_text().split())
                m = re.search(r'特色[:：]?\s*(.{10,300}?)(?:連結事業|株探ポイント|【|$)', full_text)
                if m:
                    candidate = '【特色】' + m.group(1).strip()
                    m2 = re.search(r'連結事業[:：]?\s*(.{5,200}?)(?:【|株探ポイント|$)', full_text)
                    if m2:
                        candidate += '\n【連結事業】' + m2.group(1).strip()
                    summary_text = candidate

            if _is_valid_summary(summary_text):
                info['summary'] = summary_text
                return info
    except Exception:
        pass

    # --- 2. フォールバック: みんかぶ (minkabu.jp) の企業概要 ---
    try:
        url = f"https://minkabu.jp/stock/{code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3'
        }
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        res = requests.get(url, headers=headers, timeout=8, verify=False)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            if not info['name']:
                title = soup.find('title')
                if title:
                    info['name'] = title.get_text().split('【')[0].strip()

            # 概要らしき段落を探す（JS警告等は除外）
            candidates = []
            for p in soup.find_all(['p', 'div']):
                txt = p.get_text(" ", strip=True)
                if _is_valid_summary(txt) and 30 <= len(txt) <= 400:
                    candidates.append(txt)
            if candidates:
                # 最も長いものを採用
                info['summary'] = max(candidates, key=len)
                if info['summary']:
                    return info
    except Exception:
        pass

    # --- 3. さらにフォールバック: Yahoo!ファイナンス（日本） ---
    #     ※現在JSレンダリング必須のため取得できないことが多い。
    #       JS警告文などの無効テキストは _is_valid_summary で確実に弾く。
    try:
        url = f"https://finance.yahoo.co.jp/quote/{code}.T/profile"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        res = requests.get(url, headers=headers, timeout=8, verify=False)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            title = soup.find('title')
            if title and not info['name']:
                info['name'] = title.text.split('【')[0].strip()

            longest_p = ""
            for p in soup.find_all('p'):
                txt = p.get_text().strip()
                if _is_valid_summary(txt) and len(txt) > len(longest_p):
                    longest_p = txt

            if _is_valid_summary(longest_p):
                info['summary'] = longest_p
    except Exception:
        pass

    # 上記すべてで有効な情報が得られなかった場合は空のまま返す
    # → 呼び出し側 (analyze_ticker) で yfinance の longBusinessSummary に自然にフォールバックする
    if not _is_valid_summary(info.get('summary', '')):
        info['summary'] = ""

    return info

@st.cache_data(ttl=3600)
def scrape_minkabu(code):
    info = {"target_price": "—", "analyst_trend": "—"}
    try:
        url = f"https://minkabu.jp/stock/{code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3'
        }
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        res = requests.get(url, headers=headers, timeout=5, verify=False)
        res.encoding = 'utf-8'
        if res.status_code == 200:
            text = res.text
            # 簡易的な正規表現で目標株価を抽出（みんかぶの構造が複雑なため）
            # 例: みんかぶ目標株価 <span>3,500</span>円
            m = re.search(r'みんかぶ目標株価.*?([0-9,.]+)円', text, re.DOTALL)
            if m:
                info["target_price"] = m.group(1) + " 円"
            
            # アナリスト予想（買い・売り）の抽出
            if "買い予想" in text and "売り予想" not in text:
                info["analyst_trend"] = "🔥 買い"
            elif "売り予想" in text and "買い予想" not in text:
                info["analyst_trend"] = "🔻 売り"
            else:
                info["analyst_trend"] = "中立"
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

    kb_info = scrape_japanese_info(code)
    result["name"]      = kb_info.get('name') or safe_get(info, "shortName") or code
    result["sector"]    = kb_info.get('sector') or safe_get(info, "sector", "—")
    result["summary"]   = kb_info.get('summary') or safe_get(info, "longBusinessSummary", "")
    result["employees"] = safe_get(info, "fullTimeEmployees", "—")
    result["target_price_analyst"] = safe_get(info, "targetMeanPrice")

    # みんかぶ情報の取得
    result["minkabu"] = scrape_minkabu(code)

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
    
    buy_target = float(latest['BB_lower'])
    sell_target = float(latest['BB_mid'])
    final_target = result["target_price_analyst"] if result["target_price_analyst"] else float(latest['BB_upper'])
    
    result["sim_buy"] = buy_target if current_price > buy_target else current_price
    result["sim_sell"] = sell_target if sell_target > current_price else current_price * 1.05
    result["sim_target"] = final_target if final_target > current_price else current_price * 1.10

    per = safe_get(info, "trailingPE")
    pbr = safe_get(info, "priceToBook")
    roe = safe_get(info, "returnOnEquity")
    mcap = safe_get(info, "marketCap")
    
    sc_val = calc_score(per, 5, 30, reverse=True)
    sc_prof = calc_score(roe, 0, 0.20)
    sc_stab = calc_score(mcap, 1e10, 1e12)
    
    sc_mom = calc_score(result["rsi"], 30, 70, reverse=True)
    if current_price <= buy_target * 1.02: 
        sc_mom = 10.0
        
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
                <div style="margin-top: 10px; font-size: 0.9rem; color: #555;">
                    ※ 📊 <b>みんかぶ目標株価:</b> {r['minkabu']['target_price']} (予想: {r['minkabu']['analyst_trend']})
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
            
            st.markdown("---")
            st.markdown("#### 🔗 関連リンク (IR・企業情報)")
            st.markdown(f"- [Yahoo!ファイナンスで企業情報・IRを見る](https://finance.yahoo.co.jp/quote/{r['code']}.T/profile)")
            st.markdown(f"- [株探で業績推移・ニュースを見る](https://kabutan.jp/stock/?code={r['code']})")
            st.markdown(f"- [みんなの株式（みんかぶ）でアナリスト予想を見る](https://minkabu.jp/stock/{r['code']})")
            st.markdown(f"- [JPX（日本取引所グループ）適時開示情報検索](https://www.release.tdnet.info/inbs/I_main_00.html)")
            
        with tab2:
            st.write("ボリンジャーバンドの-2σ（青点線の下限）や、RSIが30を下回るタイミングが逆張りの狙い目となります。")
            
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}  |  情報提供元: 株探 / Yahoo Finance / みんかぶ")

elif run and not code:
    st.warning("⚠️ 銘柄コードを入力してください。")
