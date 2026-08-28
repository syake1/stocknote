import os
import re
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from stocknote_tracking import load_active, load_history

st.set_page_config(page_title="Stocknote 追跡・保有管理", layout="wide")
st.title("📌 Stocknote 追跡・保有管理")
st.caption("候補の経過追跡・実保有株・現在資金・損益をStocknote内で一括管理します。固定300万円は使いません。")

DB_PATH = os.getenv("STOCKNOTE_DB_PATH", "data/stocknote.sqlite3")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)


def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT,
            side TEXT NOT NULL DEFAULT '買い',
            trade_date TEXT NOT NULL,
            shares REAL NOT NULL,
            entry_price REAL NOT NULL,
            status TEXT NOT NULL DEFAULT '保有中',
            exit_date TEXT,
            exit_price REAL,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT,
            side TEXT NOT NULL DEFAULT '買い',
            signal_date TEXT NOT NULL,
            signal_price REAL NOT NULL,
            source TEXT,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('cash',0)")
        con.commit()


init_db()

# ======================== 自動監視候補 ========================
st.markdown("## 🔄 自動監視中の買い候補")
st.caption("新規検索が0件でも消えず、初回検出から14日間、相場中に15分ごとに再評価されます。")
auto_active = load_active()
if auto_active:
    auto_rows = []
    for item in auto_active:
        auto_rows.append({
            "コード": item.get("code"), "銘柄名": item.get("name"),
            "現在値": item.get("current_price"), "騰落率%": item.get("return_pct"),
            "状態": item.get("status"), "経過日数": item.get("elapsed_days"),
            "RSI14": item.get("rsi"), "BB位置σ": item.get("bb_position"),
            "出来高倍率": item.get("volume_ratio"), "買いスコア": item.get("score"),
            "初回検出日時": item.get("first_seen_at"), "最終更新日時": item.get("updated_at"),
        })
    st.dataframe(pd.DataFrame(auto_rows), hide_index=True, use_container_width=True)
else:
    st.info("現在、自動監視中の買い候補はありません。新規候補が検出されるとここへ追加されます。")

with st.expander("📚 14日間の監視を終了した検証履歴"):
    completed = load_history()
    if completed:
        history_rows = []
        for item in completed:
            snapshots = item.get("snapshots", [])
            checkpoints = {}
            for day in (1, 3, 5, 7, 14):
                choices = [s for s in snapshots if (s.get("elapsed_days") or 0) >= day]
                checkpoints[f"{day}日後%"] = choices[0].get("return_pct") if choices else None
            history_rows.append({
                "コード": item.get("code"), "銘柄名": item.get("name"),
                "初回価格": item.get("first_price"), **checkpoints,
                "最終状態": item.get("status"), "初回検出日時": item.get("first_seen_at"),
                "監視終了日時": item.get("updated_at"),
            })
        st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("監視終了済みの履歴はまだありません。")


def normalize_code(value):
    text = str(value).strip().upper().replace('.T', '')
    text = re.sub(r'\.0$', '', text)
    m = re.search(r'([0-9A-Z]{4})', text)
    return m.group(1) if m else ''


def get_cash():
    with db() as con:
        row = con.execute("SELECT value FROM settings WHERE key='cash'").fetchone()
    return float(row['value']) if row else 0.0


def set_cash(value):
    with db() as con:
        con.execute("INSERT INTO settings(key,value) VALUES('cash',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (float(value),))
        con.commit()


@st.cache_data(ttl=300, show_spinner=False)
def latest_prices(codes):
    codes = [normalize_code(c) for c in codes]
    codes = [c for c in dict.fromkeys(codes) if c]
    out = {}
    for code in codes:
        try:
            hist = yf.download(f"{code}.T", period="5d", progress=False, auto_adjust=False, threads=False)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            close = pd.to_numeric(hist.get('Close'), errors='coerce').dropna()
            if not close.empty:
                out[code] = float(close.iloc[-1])
        except Exception:
            pass
    return out


@st.cache_data(ttl=900, show_spinner=False)
def history_from(code, start_date):
    try:
        hist = yf.download(
            f"{code}.T",
            start=(pd.Timestamp(start_date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d'),
            end=(date.today() + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist.dropna(subset=['Close']).copy()
    except Exception:
        return pd.DataFrame()


def nth_close(hist, signal_date, n):
    if hist.empty:
        return None
    after = hist[hist.index >= pd.Timestamp(signal_date)]
    if len(after) <= n:
        return None
    return float(after['Close'].iloc[n])


def return_pct(current, base, side='買い'):
    if current is None or not base:
        return None
    r = (float(current) / float(base) - 1) * 100
    return r if side == '買い' else -r


# ======================== 資金サマリー ========================
with db() as con:
    holdings_df = pd.read_sql_query("SELECT * FROM holdings ORDER BY id DESC", con)
    tracking_df = pd.read_sql_query("SELECT * FROM tracking ORDER BY id DESC", con)

open_df = holdings_df[holdings_df['status'] == '保有中'].copy() if not holdings_df.empty else pd.DataFrame()
closed_df = holdings_df[holdings_df['status'] == '決済済'].copy() if not holdings_df.empty else pd.DataFrame()
all_codes = open_df['code'].tolist() if not open_df.empty else []
prices = latest_prices(tuple(all_codes))

market_value = 0.0
unrealized = 0.0
if not open_df.empty:
    for _, r in open_df.iterrows():
        cur = prices.get(r['code'])
        if cur is None:
            continue
        qty = float(r['shares'])
        entry = float(r['entry_price'])
        if r['side'] == '買い':
            market_value += cur * qty
            unrealized += (cur - entry) * qty
        else:
            market_value += entry * qty
            unrealized += (entry - cur) * qty

realized = 0.0
if not closed_df.empty:
    for _, r in closed_df.iterrows():
        if pd.isna(r['exit_price']):
            continue
        qty = float(r['shares'])
        entry = float(r['entry_price'])
        exit_price = float(r['exit_price'])
        realized += ((exit_price - entry) if r['side'] == '買い' else (entry - exit_price)) * qty

cash = get_cash()
total_assets = cash + market_value

st.markdown("## 💰 現在資金")
c1, c2, c3, c4 = st.columns(4)
c1.metric("現金", f"¥{cash:,.0f}")
c2.metric("保有株評価額", f"¥{market_value:,.0f}")
c3.metric("総資産", f"¥{total_assets:,.0f}")
c4.metric("含み損益", f"¥{unrealized:,.0f}")
st.caption(f"実現損益累計: ¥{realized:,.0f}")

with st.expander("⚙️ 現在の現金残高を設定"):
    new_cash = st.number_input("現在の現金残高", min_value=0.0, value=float(cash), step=10000.0)
    if st.button("現金残高を保存"):
        set_cash(new_cash)
        st.success("現金残高を更新しました。")
        st.rerun()

# ======================== 実保有 ========================
st.markdown("## 🧾 実際に買った・空売りした銘柄")
with st.form("holding_form", clear_on_submit=True):
    a, b, c, d = st.columns(4)
    code = normalize_code(a.text_input("証券コード", placeholder="例 7203"))
    name = b.text_input("銘柄名")
    side = c.selectbox("区分", ["買い", "空売り"])
    trade_date = d.date_input("取引日", value=date.today())
    e, f, g = st.columns(3)
    shares = e.number_input("株数", min_value=1.0, step=100.0, value=100.0)
    entry_price = f.number_input("買値 / 売建値", min_value=0.0, step=1.0)
    note = g.text_input("メモ")
    submitted = st.form_submit_button("登録")
    if submitted:
        if not code or entry_price <= 0:
            st.error("証券コードと価格を入力してください。")
        else:
            with db() as con:
                con.execute(
                    "INSERT INTO holdings(code,name,side,trade_date,shares,entry_price,status,note,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (code, name, side, trade_date.isoformat(), shares, entry_price, '保有中', note, datetime.now().isoformat(timespec='seconds')),
                )
                con.commit()
            # 約定を入力したら現金残高も自動調整
            if side == '買い':
                set_cash(max(0, get_cash() - shares * entry_price))
            else:
                set_cash(get_cash() + shares * entry_price)
            st.success("保有銘柄を登録しました。")
            st.rerun()

if not open_df.empty:
    rows = []
    for _, r in open_df.iterrows():
        cur = prices.get(r['code'])
        pnl = None
        pnl_pct = None
        if cur is not None:
            pnl = ((cur - r['entry_price']) if r['side'] == '買い' else (r['entry_price'] - cur)) * r['shares']
            pnl_pct = return_pct(cur, r['entry_price'], r['side'])
        rows.append({
            'ID': int(r['id']), '区分': r['side'], 'コード': r['code'], '銘柄名': r['name'], '株数': r['shares'],
            '取得/建値': r['entry_price'], '現在値': cur, '含み損益': pnl, '損益率%': pnl_pct, '取引日': r['trade_date']
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### 決済・損切り・利確")
    ids = [int(x) for x in open_df['id'].tolist()]
    sell_id = st.selectbox("決済するID", ids)
    exit_price = st.number_input("決済価格", min_value=0.0, step=1.0)
    exit_date = st.date_input("決済日", value=date.today(), key='exit_date')
    if st.button("決済を記録"):
        row = open_df[open_df['id'] == sell_id].iloc[0]
        if exit_price <= 0:
            st.error("決済価格を入力してください。")
        else:
            with db() as con:
                con.execute("UPDATE holdings SET status='決済済', exit_date=?, exit_price=? WHERE id=?", (exit_date.isoformat(), exit_price, sell_id))
                con.commit()
            qty = float(row['shares'])
            if row['side'] == '買い':
                set_cash(get_cash() + qty * exit_price)
            else:
                # 空売りは売建時に現金へ加算しているため買戻し分を減算
                set_cash(max(0, get_cash() - qty * exit_price))
            st.success("決済を記録し、現金残高へ反映しました。")
            st.rerun()
else:
    st.info("現在登録されている保有銘柄はありません。")

# ======================== 候補追跡 ========================
st.markdown("## 📈 買い候補・空売り候補の経過追跡")
with st.form("tracking_form", clear_on_submit=True):
    a, b, c, d = st.columns(4)
    t_code = normalize_code(a.text_input("候補コード", placeholder="例 7203"))
    t_name = b.text_input("候補銘柄名")
    t_side = c.selectbox("候補区分", ["買い", "空売り"], key='track_side')
    t_date = d.date_input("シグナル日", value=date.today())
    e, f, g = st.columns(3)
    t_price = e.number_input("シグナル価格", min_value=0.0, step=1.0)
    source = f.text_input("出所", value="Stocknote")
    t_note = g.text_input("メモ", key='track_note')
    track_submit = st.form_submit_button("追跡に追加")
    if track_submit:
        if not t_code or t_price <= 0:
            st.error("コードとシグナル価格を入力してください。")
        else:
            with db() as con:
                con.execute(
                    "INSERT INTO tracking(code,name,side,signal_date,signal_price,source,note,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (t_code, t_name, t_side, t_date.isoformat(), t_price, source, t_note, datetime.now().isoformat(timespec='seconds')),
                )
                con.commit()
            st.success("追跡対象へ追加しました。")
            st.rerun()

if not tracking_df.empty:
    track_rows = []
    for _, r in tracking_df.iterrows():
        hist = history_from(r['code'], r['signal_date'])
        p1 = nth_close(hist, r['signal_date'], 1)
        p3 = nth_close(hist, r['signal_date'], 3)
        p5 = nth_close(hist, r['signal_date'], 5)
        p10 = nth_close(hist, r['signal_date'], 10)
        current = float(hist['Close'].iloc[-1]) if not hist.empty else None
        track_rows.append({
            '区分': r['side'], 'コード': r['code'], '銘柄名': r['name'], 'シグナル日': r['signal_date'],
            'シグナル価格': r['signal_price'], '1日後%': return_pct(p1, r['signal_price'], r['side']),
            '3日後%': return_pct(p3, r['signal_price'], r['side']), '5日後%': return_pct(p5, r['signal_price'], r['side']),
            '10日後%': return_pct(p10, r['signal_price'], r['side']), '現在値': current,
            '現在損益%': return_pct(current, r['signal_price'], r['side']), '出所': r['source']
        })
    st.dataframe(pd.DataFrame(track_rows), hide_index=True, use_container_width=True)
else:
    st.info("まだ追跡候補はありません。統合スキャナーの上位候補から登録して使えます。")

st.caption("※ 株価はYahoo Finance経由。Streamlit CloudではローカルSQLiteが再デプロイで消えることがあるため、本番サーバーではSTOCKNOTE_DB_PATHを永続ディスクへ向けてください。")
