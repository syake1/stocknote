import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="銘柄分析ノート", layout="wide")

st.title("📋 銘柄分析ノート（簡易版）")
st.caption("銘柄コードを入れると、基本情報・業績・テクニカル・簡易理論株価をまとめて表示します。")

# ================================================================
# 指標計算（既存kabujijiと共通ロジック）
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
        if v is None:
            return default
        return v
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
# メイン分析処理
# ================================================================
def analyze_ticker(code):
    result = {"code": code}
    tk = yf.Ticker(f"{code}.T")

    # --- 基本情報 ---
    try:
        info = tk.info
    except Exception:
        info = {}

    result["name"]        = safe_get(info, "longName") or safe_get(info, "shortName") or code
    result["sector"]      = safe_get(info, "sector", "—")
    result["industry"]    = safe_get(info, "industry", "—")
    result["summary"]     = safe_get(info, "longBusinessSummary", "")
    result["employees"]   = safe_get(info, "fullTimeEmployees", "—")

    # --- 株価・テクニカル用の日足データ ---
    hist = tk.history(period="3y")
    if len(hist) < 60:
        result["error"] = "株価データが十分に取得できませんでした。コードが正しいか確認してください。"
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
    result["volatility"] = float(daily_ret.tail(60).std() * (252 ** 0.5))  # 年率換算ボラティリティ

    # --- 財務・バリュエーション ---
    result["per"]  = safe_get(info, "trailingPE")
    result["pbr"]  = safe_get(info, "priceToBook")
    result["roe"]  = safe_get(info, "returnOnEquity")
    result["eps"]  = safe_get(info, "trailingEps")
    result["bps"]  = None
    if result["pbr"] and current_price:
        try:
            result["bps"] = current_price / result["pbr"]
        except Exception:
            pass
    result["dividend_yield"] = safe_get(info, "dividendYield")
    result["market_cap"] = safe_get(info, "marketCap")
    result["equity_ratio"] = None  # yfinanceでは直接取れないため balance_sheet から概算

    try:
        bs = tk.balance_sheet
        if bs is not None and not bs.empty:
            total_assets = None
            total_equity = None
            for idx in bs.index:
                if 'Total Assets' in str(idx):
                    total_assets = bs.loc[idx].iloc[0]
                if 'Stockholders Equity' in str(idx) or 'Total Equity' in str(idx):
                    total_equity = bs.loc[idx].iloc[0]
            if total_assets and total_equity and total_assets != 0:
                result["equity_ratio"] = float(total_equity) / float(total_assets)
    except Exception:
        pass

    # --- 業績推移（売上・純利益） ---
    growth_df = None
    try:
        fin = tk.financials
        if fin is not None and not fin.empty:
            rev_row = None
            profit_row = None
            for idx in fin.index:
                if 'Total Revenue' in str(idx):
                    rev_row = fin.loc[idx]
                if str(idx) == 'Net Income':
                    profit_row = fin.loc[idx]
            if rev_row is not None:
                growth_df = pd.DataFrame({"売上高": rev_row})
                if profit_row is not None:
                    growth_df["純利益"] = profit_row
                growth_df = growth_df.iloc[:, ::1]
                growth_df = growth_df.sort_index(axis=0)
                growth_df.index = [str(d)[:10] for d in growth_df.index]
    except Exception:
        pass
    result["growth_df"] = growth_df

    # 増収率・増益率（直近2期比較、新しい列が先頭想定なのでreverseで比較）
    result["revenue_growth"] = None
    result["profit_growth"] = None
    if growth_df is not None and len(growth_df) >= 2:
        try:
            rev_vals = growth_df["売上高"].dropna().values
            if len(rev_vals) >= 2:
                result["revenue_growth"] = (rev_vals[-1] - rev_vals[-2]) / abs(rev_vals[-2])
            if "純利益" in growth_df.columns:
                pr_vals = growth_df["純利益"].dropna().values
                if len(pr_vals) >= 2:
                    result["profit_growth"] = (pr_vals[-1] - pr_vals[-2]) / abs(pr_vals[-2])
        except Exception:
            pass

    # --- 為替感応度（ドル円との相関） ---
    result["fx_corr"] = None
    try:
        fx = yf.Ticker("JPY=X").history(period="1y")['Close'].pct_change().dropna()
        stock_ret = hist['Close'].pct_change().dropna().tail(len(fx))
        merged = pd.concat([stock_ret.tail(len(fx)), fx.tail(len(stock_ret))], axis=1).dropna()
        if len(merged) > 30:
            merged.columns = ["stock", "fx"]
            result["fx_corr"] = float(merged["stock"].corr(merged["fx"]))
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
        if result["rsi"] < 35:
            score += 1; reasons.append(f"RSI {result['rsi']:.0f}（売られすぎ圏）")
        elif result["rsi"] > 70:
            score -= 1; reasons.append(f"RSI {result['rsi']:.0f}（買われすぎ圏）")

    if result["macd_val"] is not None and result["macd_sig"] is not None:
        if result["macd_val"] > result["macd_sig"]:
            score += 1; reasons.append("MACDが上向き")
        else:
            score -= 1; reasons.append("MACDが下向き")

    if result["revenue_growth"] and result["profit_growth"]:
        if result["revenue_growth"] > 0 and result["profit_growth"] > 0:
            score += 1; reasons.append("増収増益")
        elif result["revenue_growth"] < 0 and result["profit_growth"] < 0:
            score -= 1; reasons.append("減収減益")

    if score >= 3:
        judgement = "🔥 買い推奨"
    elif score >= 1:
        judgement = "🙂 やや買い"
    elif score <= -3:
        judgement = "🔻 売り推奨"
    elif score <= -1:
        judgement = "🤔 やや弱気"
    else:
        judgement = "⏳ 中立・様子見"

    result["score"] = score
    result["judgement"] = judgement
    result["reasons"] = reasons

    return result


# ================================================================
# UI
# ================================================================
col_input, col_btn = st.columns([3, 1])
with col_input:
    code = st.text_input("銘柄コードを入力（例：7203、6501）", value="", max_chars=6)
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 分析する", type="primary", use_container_width=True)

if run and code:
    with st.spinner(f"{code} を分析中..."):
        r = analyze_ticker(code.strip())

    if "error" in r:
        st.error(r["error"])
    else:
        st.markdown("---")

        # ヘッダー
        st.subheader(f"{r['code']} {r['name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在値", f"{r['current_price']:,.0f}円")
        c2.metric("総合判定", r["judgement"])
        c3.metric("52週高値/安値", f"{r['week52_high']:,.0f} / {r['week52_low']:,.0f}")
        c4.metric("年率ボラティリティ", fmt_pct(r["volatility"]))

        st.markdown("**📌 判定の根拠**")
        if r["reasons"]:
            for reason in r["reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("- 明確なシグナルなし")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🏢 事業概要", "📊 業績・成長性", "💹 バリュエーション", "📈 テクニカル", "💰 財務・為替"]
        )

        with tab1:
            st.write(f"**業種:** {r['sector']} / {r['industry']}")
            if r.get("employees") and r["employees"] != "—":
                st.write(f"**従業員数:** {r['employees']:,}人")
            if r["summary"]:
                st.write(r["summary"])
            else:
                st.info("この銘柄の事業概要テキストは取得できませんでした（yfinanceの提供元データに依存するため、日本株では未収録の場合があります）。")
            st.caption("※ IR資料の要約・株主構成の詳細は、このバージョンでは未対応です。")

        with tab2:
            if r["growth_df"] is not None and not r["growth_df"].empty:
                st.dataframe(r["growth_df"], use_container_width=True)
                gc1, gc2 = st.columns(2)
                gc1.metric("直近の増収率", fmt_pct(r["revenue_growth"]))
                gc2.metric("直近の増益率", fmt_pct(r["profit_growth"]))
            else:
                st.info("業績データを取得できませんでした。")

        with tab3:
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("PER", fmt_num(r["per"]))
            vc2.metric("PBR", fmt_num(r["pbr"]))
            vc3.metric("ROE", fmt_pct(r["roe"]))
            st.caption("PER・PBRは市場平均や同業他社と比べることで割安・割高の目安になります。")

        with tab4:
            st.line_chart(r["hist"][["Close", "MA25", "MA75", "MA200"]].tail(250))
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("RSI(14)", fmt_num(r["rsi"]))
            tc2.metric("MACD", "↑上" if (r["macd_val"] or 0) > (r["macd_sig"] or 0) else "↓下")
            tc3.metric("BB位置", f"{r['bb_pos']:.0f}%" if r["bb_pos"] is not None else "—")

        with tab5:
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("自己資本率", fmt_pct(r["equity_ratio"]))
            fc2.metric("配当利回り", fmt_pct(r["dividend_yield"]))
            fc3.metric("時価総額", f"{r['market_cap']/1e8:,.0f}億円" if r["market_cap"] else "—")
            st.markdown("**ドル円との連動性（過去1年・簡易相関）**")
            if r["fx_corr"] is not None:
                st.metric("相関係数", f"{r['fx_corr']:.2f}")
                if r["fx_corr"] > 0.3:
                    st.write("→ 円安で株価が上がりやすい傾向（輸出関連の可能性）")
                elif r["fx_corr"] < -0.3:
                    st.write("→ 円高で株価が上がりやすい傾向（輸入・内需関連の可能性）")
                else:
                    st.write("→ 為替との明確な連動性は見られません")
            else:
                st.info("為替相関を計算するデータが不足しています。")

        st.markdown("---")
        st.caption(f"分析日時: {datetime.now().strftime('%Y/%m/%d %H:%M')} ※ このアプリは投資助言ではありません。参考情報としてご利用ください。")

elif run and not code:
    st.warning("銘柄コードを入力してください。")
