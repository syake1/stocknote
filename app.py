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

def _contains_japanese(text: str) -> bool:
    """テキストに日本語（ひらがな/カタカナ/漢字）が含まれているか判定"""
    if not text:
        return False
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))


@st.cache_data(ttl=86400)
def translate_to_japanese(text: str) -> str:
    """
    yfinanceのlongBusinessSummary等、英語しか取得できなかった場合に日本語へ自動翻訳する。
    deep-translatorが使えない/翻訳に失敗した場合は、原文に注記を付けてそのまま返す。
    """
    if not text:
        return text
    if _contains_japanese(text):
        return text
    try:
        from deep_translator import GoogleTranslator
        # 長すぎる場合は分割して翻訳（GoogleTranslatorは1回あたりの文字数制限があるため）
        chunks = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        translated_chunks = [GoogleTranslator(source='en', target='ja').translate(c) for c in chunks]
        translated = ' '.join(t for t in translated_chunks if t)
        if translated:
            return translated + "\n\n※英語の企業情報（yfinance提供）を自動翻訳しています。ニュアンスが不正確な場合があります。"
    except Exception:
        pass
    # 翻訳できなかった場合はそのまま返す（原文が英語であることを明記）
    return text + "\n\n※日本語の企業情報が取得できなかったため、英語の原文をそのまま表示しています。"


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


# ============================================================
# 銀行株など、株探の「特色」欄が【資金】【資産】【融資】のような
# 構成比データになっているケース向けの解析・用語集
# ============================================================
BANK_TERM_GLOSSARY = {
    "定期": "定期預金（一定期間は原則引き出せない代わりに金利が高めの預金）",
    "普通": "普通預金（自由に出し入れできる一般的な預金）",
    "当座": "当座預金（手形・小切手の決済に使う無利息の預金。主に企業が利用）",
    "通知": "通知預金（引き出す際に事前通知が必要な、まとまった資金向けの預金）",
    "現・預け金": "現金及び日銀・他の金融機関への預け金（すぐに使える手元資金）",
    "有価証券": "国債・地方債・株式など、銀行が保有する有価証券",
    "貸出金": "企業や個人への融資残高（銀行の収益の柱）",
    "中小企業等向け": "中小企業向けの事業性融資",
    "住宅・消費者向け": "住宅ローンやカードローンなど、個人向け融資",
    "他": "その他の項目",
}


def parse_bracket_breakdown(text):
    """
    「【資金】定期24、普通66、当座4、他6【資産】現・預け金10、有価証券14…」
    のような株探の銀行株特有フォーマットを
    {"資金": [("定期",24.0), ("普通",66.0), ...], "資産": [...], ...} の形に変換する。
    パースできなければ空dictを返す。
    """
    if not text:
        return {}
    sections = {}
    for m in re.finditer(r'【([^】]{1,10})】\s*([^【]*)', text):
        label = m.group(1).strip()
        content = m.group(2).strip()
        items = []
        for im in re.finditer(
            r'([一\u4e00-\u9fffぁ-んァ-ヶー・]{1,10}?)(\d{1,3}(?:\.\d+)?)(?:\([^)]*\))?(?=[、,]|$)',
            content
        ):
            name = im.group(1).strip(' 、,・')
            try:
                pct = float(im.group(2))
            except ValueError:
                continue
            if name and 0 < pct <= 100:
                items.append((name, pct))
        if items:
            sections[label] = items
    return sections


def extract_intro_text(text, bracket_start_label="【"):
    """括弧構成データが始まる前の、通常の説明文（あれば）だけを取り出す"""
    if not text:
        return ""
    idx = text.find(bracket_start_label)
    intro = text[:idx].strip() if idx > 0 else text.strip()
    intro = re.sub(r'^【特色】\s*', '', intro).strip()
    return intro


def parse_segments(segment_text):
    """
    株探の「連結事業」テキスト（例：'自動車74(19)、金融9(20)、住宅1(3)'）から
    [(セグメント名, 構成比%), ...] のリストを抽出する。
    パースできない場合は空リストを返す（呼び出し側は生テキストを表示すればよい）。
    """
    if not segment_text:
        return []
    results = []
    # 株探の表記は「自動車74(19)、金融9(20)」のように%記号が付かない場合が多いため、
    # 「セグメント名＋構成比の数字＋(利益率など、任意)」のパターンで抽出する（%記号があっても対応）
    for m in re.finditer(
        r'([一\u4e00-\u9fffぁ-んァ-ヶーA-Za-z・]{1,14})(\d{1,3}(?:\.\d+)?)[%％]?(?:\(\-?\d+(?:\.\d+)?\))?',
        segment_text
    ):
        name = m.group(1).strip(' 、,・')
        try:
            pct = float(m.group(2))
        except ValueError:
            continue
        if name and 0 < pct <= 100:
            results.append((name, pct))
    return results


@st.cache_data(ttl=3600)
def scrape_japanese_info(code):
    info = {"name": "", "sector": "", "summary": "", "segments_raw": "", "segments": []}

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
            tokusyoku_text = ""
            segment_text = ""
            summary_div = soup.find('div', class_='company_block')
            if summary_div:
                text = ' '.join(summary_div.get_text().split())
                m_t = re.search(r'特色[:：]?\s*(.{5,300}?)(?:連結事業|$)', text)
                if m_t:
                    tokusyoku_text = m_t.group(1).strip()
                m_s = re.search(r'連結事業[:：]?\s*(.{3,250}?)$', text)
                if m_s:
                    segment_text = m_s.group(1).strip()

            if not _is_valid_summary(tokusyoku_text):
                # フォールバック：ページ全体のテキストから「特色」「連結事業」を含む箇所を正規表現で拾う
                full_text = ' '.join(soup.get_text().split())
                m = re.search(r'特色[:：]?\s*(.{10,300}?)(?:連結事業|株探ポイント|【|$)', full_text)
                if m:
                    tokusyoku_text = m.group(1).strip()
                m2 = re.search(r'連結事業[:：]?\s*(.{5,200}?)(?:【|株探ポイント|$)', full_text)
                if m2:
                    segment_text = m2.group(1).strip()

            if _is_valid_summary(tokusyoku_text):
                info['summary'] = '【特色】' + tokusyoku_text
                if segment_text:
                    info['summary'] += '\n【連結事業】' + segment_text
                    info['segments_raw'] = segment_text
                    info['segments'] = parse_segments(segment_text)
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
        if v is None:
            return default
        try:
            # pandasのNA/NaNや、np.isnanが使えない型を安全に判定する
            if pd.isna(v):
                return default
        except (TypeError, ValueError):
            pass
        return v
    except Exception:
        return default

def _to_finite_float(x):
    """xを安全にfloatへ変換する。None/NaN/変換不可なら None を返す。"""
    if x is None:
        return None
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(fx):
        return None
    return fx

def fmt_pct(x, digits=1):
    fx = _to_finite_float(x)
    return "—" if fx is None else f"{fx*100:.{digits}f}%"

def fmt_num(x, digits=1):
    fx = _to_finite_float(x)
    return "—" if fx is None else f"{fx:,.{digits}f}"

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

def get_financial_highlights(info):
    """時価総額・PER・PBR・ROE・配当利回り・自己資本比率などをまとめて返す"""
    h = {}
    h["market_cap"] = safe_get(info, "marketCap")
    h["per"] = safe_get(info, "trailingPE")
    h["pbr"] = safe_get(info, "priceToBook")
    h["roe"] = safe_get(info, "returnOnEquity")
    h["roa"] = safe_get(info, "returnOnAssets")
    div_yield = safe_get(info, "dividendYield")
    # yfinanceはdividendYieldを「%」の数値（例: 2.5）で返す場合と小数（0.025）で返す場合が混在するため補正
    if div_yield is not None:
        div_yield = div_yield / 100 if div_yield > 1 else div_yield
    h["dividend_yield"] = div_yield
    h["equity_ratio"] = None  # balance sheetから別途計算
    h["op_margin"] = safe_get(info, "operatingMargins")
    h["revenue_growth"] = safe_get(info, "revenueGrowth")
    return h


def fmt_market_cap(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    oku = x / 1e8  # 億円換算
    if oku >= 10000:
        return f"{oku/10000:,.1f} 兆円"
    return f"{oku:,.0f} 億円"


# ============================================================
# 財務指標の説明文（st.metricのhelpツールチップに表示）
# ============================================================
METRIC_HELP = {
    "market_cap": "時価総額＝株価×発行済株式数。会社の規模・市場評価額の目安です。",
    "per": "PER（株価収益率）＝株価 ÷ 1株当たり利益。株価が利益の何年分かを示し、低いほど「割安」とされます（業種により適正水準は異なります）。",
    "pbr": "PBR（株価純資産倍率）＝株価 ÷ 1株当たり純資産。1倍が「解散価値」の目安で、1倍未満は資産価値より株価が安い状態です。",
    "roe": "ROE（自己資本利益率）＝当期純利益 ÷ 自己資本。株主が出したお金でどれだけ効率的に利益を稼いでいるかを示します。高いほど資本効率が良いとされます。",
    "roa": "ROA（総資産利益率）＝当期純利益 ÷ 総資産。会社の持つ全資産をどれだけ効率的に使って利益を出しているかを示します。",
    "dividend_yield": "配当利回り＝年間配当金 ÷ 株価。株価に対して配当でどれだけ還元されるかを示します。",
    "equity_ratio": "自己資本比率＝自己資本 ÷ 総資産。高いほど借金が少なく財務が健全（倒産しにくい）とされます。",
    "op_margin": "営業利益率＝営業利益 ÷ 売上高。本業でどれだけ効率よく稼げているかを示す収益力の指標です。",
    "revenue_growth": "売上高成長率＝前年同期比の売上高の伸び率。会社が成長しているかどうかの目安です。",
}


def _eval_badge(value, thresholds, labels):
    """
    value: 評価対象の数値（None可）
    thresholds: 昇順の閾値リスト 例 [0.08, 0.15] -> 3区分
    labels: (低評価, 中間評価, 高評価) のラベルタプル（thresholdsの数+1個）
    戻り値: 評価バッジ文字列（例 "🟢 優良水準"）。valueがNoneなら空文字。
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    idx = 0
    for t in thresholds:
        if value >= t:
            idx += 1
        else:
            break
    return labels[idx]


def evaluate_financials(fin):
    """各指標について簡易評価バッジを付与して返す（あくまで一般的な目安）"""
    ev = {}
    ev["per"] = _eval_badge(fin.get("per"), [15, 25], ("🟢 割安水準", "🟡 標準的", "🔴 割高水準")) if fin.get("per") else ""
    if fin.get("per") and fin["per"] < 0:
        ev["per"] = "⚫ 赤字（PERマイナス）"
    ev["pbr"] = _eval_badge(fin.get("pbr"), [1, 3], ("🟢 割安（解散価値割れ）", "🟡 標準的", "🔴 割高水準")) if fin.get("pbr") else ""
    ev["roe"] = _eval_badge(fin.get("roe"), [0.08, 0.15], ("🔴 資本効率は低め", "🟡 標準的", "🟢 高い資本効率")) if fin.get("roe") is not None else ""
    ev["roa"] = _eval_badge(fin.get("roa"), [0.02, 0.05], ("🔴 やや低め", "🟡 標準的", "🟢 効率良好")) if fin.get("roa") is not None else ""
    ev["dividend_yield"] = _eval_badge(fin.get("dividend_yield"), [0.02, 0.04], ("🔴 低め", "🟡 標準的", "🟢 高配当")) if fin.get("dividend_yield") is not None else ""
    ev["equity_ratio"] = _eval_badge(fin.get("equity_ratio"), [0.3, 0.5], ("🔴 財務やや不安定", "🟡 標準的", "🟢 財務健全")) if fin.get("equity_ratio") is not None else ""
    ev["op_margin"] = _eval_badge(fin.get("op_margin"), [0.05, 0.15], ("🔴 収益力は低め", "🟡 標準的", "🟢 高収益体質")) if fin.get("op_margin") is not None else ""
    ev["revenue_growth"] = _eval_badge(fin.get("revenue_growth"), [0, 0.10], ("🔴 減収", "🟡 横ばい〜微増", "🟢 高成長")) if fin.get("revenue_growth") is not None else ""
    return ev


@st.cache_data(ttl=3600)
def get_performance_trend(code):
    """
    年次の売上高・営業利益・純利益の推移（直近最大4期）を取得する。
    yfinanceのfinancials（income statement）が取得できない銘柄もあるため、
    取得できた分だけ返す。戻り値はDataFrame（列=決算期、行=売上高/営業利益/純利益）または None。
    """
    try:
        tk = yf.Ticker(f"{code}.T")
        fin = tk.financials  # annual income statement, columns=決算期（降順）
        if fin is None or fin.empty:
            return None

        def pick_row(candidates):
            for name in candidates:
                if name in fin.index:
                    return fin.loc[name]
            return None

        revenue = pick_row(["Total Revenue", "TotalRevenue", "Operating Revenue"])
        op_income = pick_row(["Operating Income", "OperatingIncome"])
        net_income = pick_row(["Net Income", "NetIncome", "Net Income Common Stockholders"])

        if revenue is None and net_income is None:
            return None

        cols = fin.columns[:4]  # 直近4期
        data = {}
        if revenue is not None:
            data["売上高"] = (revenue[cols] / 1e8).round(0)
        if op_income is not None:
            data["営業利益"] = (op_income[cols] / 1e8).round(0)
        if net_income is not None:
            data["純利益"] = (net_income[cols] / 1e8).round(0)

        df = pd.DataFrame(data).T
        df.columns = [c.strftime("%Y/%m") for c in cols]
        df = df[df.columns[::-1]]  # 古い順に並べ替え
        return df
    except Exception:
        return None


def calc_score(val, v_min, v_max, reverse=False):
    val = _to_finite_float(val)
    if val is None: return 5
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
    raw_summary = kb_info.get('summary') or safe_get(info, "longBusinessSummary", "")

    # 銀行株など「【資金】定期24、普通66…」のような構成比データを解析
    # （日本語の生テキストに対してのみ解析。翻訳前の raw_summary を使う）
    result["bracket_breakdown"] = parse_bracket_breakdown(raw_summary) if _contains_japanese(raw_summary) else {}
    result["summary_intro"] = extract_intro_text(raw_summary) if result["bracket_breakdown"] else ""

    result["summary"]   = translate_to_japanese(raw_summary) if raw_summary else ""
    result["segments"]  = kb_info.get('segments', [])
    result["segments_raw"] = kb_info.get('segments_raw', "")
    result["employees"] = safe_get(info, "fullTimeEmployees", "—")
    result["target_price_analyst"] = safe_get(info, "targetMeanPrice")

    # 財務ハイライトと業績推移
    result["financials"] = get_financial_highlights(info)
    result["perf_trend"] = get_performance_trend(code)
    # 自己資本比率（balance sheetから計算。取得できない場合はNoneのまま）
    try:
        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            equity_row = None
            for name in ["Stockholders Equity", "Total Stockholder Equity", "StockholdersEquity"]:
                if name in bs.index:
                    equity_row = bs.loc[name]
                    break
            assets_row = None
            for name in ["Total Assets", "TotalAssets"]:
                if name in bs.index:
                    assets_row = bs.loc[name]
                    break
            if equity_row is not None and assets_row is not None:
                latest_col = bs.columns[0]
                eq = equity_row.get(latest_col)
                asset = assets_row.get(latest_col)
                if eq and asset:
                    result["financials"]["equity_ratio"] = float(eq) / float(asset)
    except Exception:
        pass

    # みんかぶ情報の取得
    result["minkabu"] = scrape_minkabu(code)

    try:
        hist = tk.history(period="2y")
    except Exception as e:
        result["error"] = f"Yahooファイナンスからのデータ取得に失敗しました（エラー: {str(e)}）"
        return result

    if hist is None or hist.empty:
        result["error"] = "株価データが取得できませんでした。"
        return result

    # 直近データがまだ確定しておらずOpen/High/Low/CloseがNaNの場合があるため除外する
    # （取引時間中〜終値確定前にYahoo Financeから取得すると最終行がNaNになることがある）
    hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close'])

    if hist.empty or len(hist) < 60:
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

if 'analyze_code' not in st.session_state:
    st.session_state['analyze_code'] = None
if 'watchlist_df' not in st.session_state:
    st.session_state['watchlist_df'] = None
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = None

# ---------------- CSV買い銘柄リスト ----------------
st.markdown("### 📂 買い銘柄リストから選択（CSV）")
st.caption("1列目に銘柄コード、2列目に銘柄名があるCSVをアップロードしてください（例: code,name）。銘柄をクリックすると、その場ですぐ分析します。")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        wl_df = pd.read_csv(uploaded_file, dtype=str)
        wl_df = wl_df.dropna(how="all")

        # 列名の揺れに対応（code列・name列を自動推定）
        code_col, name_col = None, None
        for c in wl_df.columns:
            cl = str(c).strip().lower()
            if code_col is None and cl in ["code", "コード", "銘柄コード", "証券コード"]:
                code_col = c
            if name_col is None and cl in ["name", "名称", "銘柄名", "会社名"]:
                name_col = c
        if code_col is None:
            code_col = wl_df.columns[0]
        if name_col is None and len(wl_df.columns) > 1:
            name_col = wl_df.columns[1]

        # コードを文字列として正規化（4091.0のような表記を防ぐ）
        wl_df[code_col] = wl_df[code_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

        st.session_state['watchlist_df'] = wl_df
        st.session_state['watchlist_code_col'] = code_col
        st.session_state['watchlist_name_col'] = name_col
        st.success(f"✅ {len(wl_df)}銘柄を読み込みました")
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")

if st.session_state.get('watchlist_df') is not None:
    wl_df = st.session_state['watchlist_df']
    code_col = st.session_state['watchlist_code_col']
    name_col = st.session_state.get('watchlist_name_col')

    n_cols = 4
    rows = list(wl_df.iterrows())
    for i in range(0, len(rows), n_cols):
        cols = st.columns(n_cols)
        for j, (idx, row) in enumerate(rows[i:i + n_cols]):
            c_code = str(row[code_col]).strip()
            c_name = str(row[name_col]).strip() if name_col else ""
            label = f"🔍 {c_code} {c_name}".strip()
            with cols[j]:
                if st.button(label, key=f"wl_btn_{idx}", use_container_width=True):
                    st.session_state['analyze_code'] = c_code

    st.markdown("")
    if st.button("📊 一括分析してランキング表示", type="secondary", use_container_width=True):
        results = []
        progress = st.progress(0, text="分析を開始します...")
        total = len(rows)
        for i, (idx, row) in enumerate(rows):
            c_code = str(row[code_col]).strip()
            c_name = str(row[name_col]).strip() if name_col else ""
            progress.progress((i + 1) / total, text=f"分析中: {c_code} {c_name} ({i + 1}/{total})")
            try:
                res = analyze_ticker(c_code)
                if "error" in res:
                    results.append({
                        "code": c_code, "name": c_name, "momentum": None,
                        "current_price": None, "sim_buy": None, "rsi": None,
                        "error": res["error"]
                    })
                else:
                    results.append({
                        "code": c_code,
                        "name": res.get("name") or c_name,
                        "momentum": res["scores"]["momentum"],
                        "current_price": res["current_price"],
                        "sim_buy": res["sim_buy"],
                        "rsi": res["rsi"],
                        "error": None
                    })
            except Exception as e:
                results.append({
                    "code": c_code, "name": c_name, "momentum": None,
                    "current_price": None, "sim_buy": None, "rsi": None,
                    "error": f"分析エラー: {e}"
                })
        progress.empty()
        st.session_state['batch_results'] = results

if st.session_state.get('batch_results'):
    st.markdown("### 🏆 逆張りチャンス ランキング")
    st.caption("逆張りチャンススコアが高い順に並んでいます。銘柄をクリックすると詳細分析に切り替わります。")

    valid = [r for r in st.session_state['batch_results'] if r["momentum"] is not None]
    invalid = [r for r in st.session_state['batch_results'] if r["momentum"] is None]
    valid_sorted = sorted(valid, key=lambda r: r["momentum"], reverse=True)

    for rank, r in enumerate(valid_sorted, start=1):
        rc1, rc2, rc3, rc4, rc5 = st.columns([0.6, 2.2, 1, 1, 1.2])
        with rc1:
            st.markdown(f"**#{rank}**")
        with rc2:
            if st.button(f"{r['code']} {r['name']}", key=f"rank_btn_{r['code']}", use_container_width=True):
                st.session_state['analyze_code'] = r['code']
        with rc3:
            st.markdown(f"スコア **{r['momentum']:.1f}**/10")
        with rc4:
            st.markdown(f"RSI {r['rsi']:.1f}")
        with rc5:
            st.markdown(f"¥{r['current_price']:,.0f}")

    if invalid:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(invalid)}件）"):
            for r in invalid:
                st.caption(f"{r['code']} {r['name']}: {r['error']}")

st.markdown("---")

# ---------------- 手動入力 ----------------
col_input, col_btn, _ = st.columns([2, 1, 3])
with col_input:
    manual_code = st.text_input("銘柄コードを入力（手動）", placeholder="例：7203", max_chars=6)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 分析を実行", type="primary", use_container_width=True)

if run and manual_code:
    st.session_state['analyze_code'] = manual_code.strip()
elif run and not manual_code:
    st.warning("⚠️ 銘柄コードを入力してください。")

code = st.session_state.get('analyze_code')

if code:
    with st.spinner(f"「{code}」の最新データを収集中..."):
        r = analyze_ticker(code)

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
            c1.metric("逆張りチャンススコア", f"{r['scores']['momentum']:.1f} / 10", help="RSIの低さやボリンジャーバンド-2σ近辺かどうかから算出した、逆張り買いタイミングとしての狙い目度（10が最も狙い目）。")
            c2.metric("現在RSI (14)", f"{r['rsi']:.1f}", help="RSI（相対力指数）＝直近の値上がり幅と値下がり幅の比率から算出する指標。一般に30以下は「売られすぎ」、70以上は「買われすぎ」の目安とされます。")
            c3.metric("PER / PBR", f"{fmt_num(r['per'])} / {fmt_num(r['pbr'])}", help="PER＝株価収益率（低いほど割安の目安）／PBR＝株価純資産倍率（1倍が解散価値の目安）。詳しい説明は下部「財務ハイライト」を参照してください。")

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
            if r.get("bracket_breakdown"):
                # 銀行株など、構成比データ形式の場合は説明文＋構成比を分けて表示
                if r.get("summary_intro"):
                    st.write(r["summary_intro"])
                else:
                    st.caption("株探の「特色」欄が構成比データのみのため、下記に読みやすく整理しています。")
            else:
                st.write(r["summary"] if r["summary"] else "情報が取得できませんでした。")

            # --- 銀行株など：資金・資産・融資の構成比 ---
            if r.get("bracket_breakdown"):
                st.markdown("#### 🏦 資金・資産・融資の構成比")
                st.caption("株探の特色データを表＋グラフに変換したものです。用語名にカーソルを合わせる（スマホは長押し）と説明が出ます。")
                bracket_cols = st.columns(len(r["bracket_breakdown"]))
                for col, (label, items) in zip(bracket_cols, r["bracket_breakdown"].items()):
                    with col:
                        st.markdown(f"**【{label}】**")
                        bdf = pd.DataFrame(items, columns=["項目", "構成比(%)"])
                        for _, row in bdf.iterrows():
                            term = row["項目"]
                            help_text = BANK_TERM_GLOSSARY.get(term, "")
                            if help_text:
                                st.metric(term, f"{row['構成比(%)']:.0f}%", help=help_text, label_visibility="visible")
                            else:
                                st.metric(term, f"{row['構成比(%)']:.0f}%")
                        fig_b = go.Figure(data=[go.Pie(
                            labels=bdf["項目"], values=bdf["構成比(%)"], hole=0.45, textinfo="percent"
                        )])
                        fig_b.update_layout(margin=dict(l=5, r=5, t=5, b=5), height=200, showlegend=True,
                                             legend=dict(orientation="h", font=dict(size=9)))
                        st.plotly_chart(fig_b, use_container_width=True, config={'displayModeBar': False})

            # --- 主力事業セグメント ---
            if r.get("segments"):
                st.markdown("#### 🏭 主力事業セグメント（連結売上構成比）")
                seg_df = pd.DataFrame(r["segments"], columns=["セグメント", "構成比(%)"]).sort_values(
                    "構成比(%)", ascending=False
                )
                seg_col1, seg_col2 = st.columns([1, 1])
                with seg_col1:
                    st.dataframe(seg_df, hide_index=True, use_container_width=True)
                with seg_col2:
                    fig_seg = go.Figure(data=[go.Pie(
                        labels=seg_df["セグメント"], values=seg_df["構成比(%)"],
                        hole=0.45, textinfo="label+percent"
                    )])
                    fig_seg.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260, showlegend=False)
                    st.plotly_chart(fig_seg, use_container_width=True, config={'displayModeBar': False})
            elif r.get("segments_raw"):
                st.markdown("#### 🏭 主力事業セグメント")
                st.caption(r["segments_raw"])

            st.markdown("---")

            # --- 財務ハイライト ---
            st.markdown("#### 💰 財務ハイライト")
            st.caption("各指標名にカーソルを合わせる（スマホは長押し）と説明が表示されます。評価は一般的な目安であり、業種や成長ステージにより適正水準は異なります。")
            fin = r.get("financials", {})
            ev = evaluate_financials(fin)

            fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
            fc1.metric("時価総額", fmt_market_cap(fin.get("market_cap")), help=METRIC_HELP["market_cap"])

            fc2.metric("PER", (fmt_num(fin.get("per")) + " 倍") if fin.get("per") else "—", help=METRIC_HELP["per"])
            if ev.get("per"): fc2.caption(ev["per"])

            fc3.metric("PBR", (fmt_num(fin.get("pbr")) + " 倍") if fin.get("pbr") else "—", help=METRIC_HELP["pbr"])
            if ev.get("pbr"): fc3.caption(ev["pbr"])

            fc4.metric("ROE", fmt_pct(fin.get("roe")), help=METRIC_HELP["roe"])
            if ev.get("roe"): fc4.caption(ev["roe"])

            fc5.metric("配当利回り", fmt_pct(fin.get("dividend_yield")), help=METRIC_HELP["dividend_yield"])
            if ev.get("dividend_yield"): fc5.caption(ev["dividend_yield"])

            fc6.metric("自己資本比率", fmt_pct(fin.get("equity_ratio")), help=METRIC_HELP["equity_ratio"])
            if ev.get("equity_ratio"): fc6.caption(ev["equity_ratio"])

            fc7, fc8, fc9 = st.columns(3)
            fc7.metric("ROA", fmt_pct(fin.get("roa")), help=METRIC_HELP["roa"])
            if ev.get("roa"): fc7.caption(ev["roa"])

            fc8.metric("営業利益率", fmt_pct(fin.get("op_margin")), help=METRIC_HELP["op_margin"])
            if ev.get("op_margin"): fc8.caption(ev["op_margin"])

            fc9.metric("売上高成長率", fmt_pct(fin.get("revenue_growth")), help=METRIC_HELP["revenue_growth"])
            if ev.get("revenue_growth"): fc9.caption(ev["revenue_growth"])

            st.markdown("---")

            # --- 業績推移 ---
            st.markdown("#### 📊 業績推移（単位：億円）")
            perf = r.get("perf_trend")
            if perf is not None and not perf.empty:
                perf_col1, perf_col2 = st.columns([1, 2])
                with perf_col1:
                    st.dataframe(perf, use_container_width=True)
                    # 直近期の前年比（売上高・純利益）
                    if perf.shape[1] >= 2:
                        latest_p, prev_p = perf.columns[-1], perf.columns[-2]
                        if "売上高" in perf.index and prev_p in perf.columns:
                            rev_prev, rev_latest = perf.loc["売上高", prev_p], perf.loc["売上高", latest_p]
                            if rev_prev:
                                st.caption(f"売上高 前期比: {((rev_latest / rev_prev) - 1) * 100:+.1f}%")
                        if "純利益" in perf.index:
                            ni_prev, ni_latest = perf.loc["純利益", prev_p], perf.loc["純利益", latest_p]
                            if ni_prev:
                                st.caption(f"純利益 前期比: {((ni_latest / ni_prev) - 1) * 100:+.1f}%")
                with perf_col2:
                    fig_perf = go.Figure()
                    for idx in perf.index:
                        fig_perf.add_trace(go.Bar(x=perf.columns, y=perf.loc[idx], name=idx))
                    fig_perf.update_layout(
                        barmode="group", template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=20), height=300,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_perf, use_container_width=True, config={'displayModeBar': False})
            else:
                st.caption("業績データを取得できませんでした。")

            st.markdown("---")
            st.markdown("#### 🔗 関連リンク (IR・企業情報)")
            st.markdown(f"- [Yahoo!ファイナンスで企業情報・IRを見る](https://finance.yahoo.co.jp/quote/{r['code']}.T/profile)")
            st.markdown(f"- [株探で業績推移・ニュースを見る](https://kabutan.jp/stock/?code={r['code']})")
            st.markdown(f"- [みんなの株式（みんかぶ）でアナリスト予想を見る](https://minkabu.jp/stock/{r['code']})")
            st.markdown(f"- [JPX（日本取引所グループ）適時開示情報検索](https://www.release.tdnet.info/inbs/I_main_00.html)")
            
        with tab2:
            st.write("ボリンジャーバンドの-2σ（青点線の下限）や、RSIが30を下回るタイミングが逆張りの狙い目となります。")
            
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}  |  情報提供元: 株探 / Yahoo Finance / みんかぶ")
