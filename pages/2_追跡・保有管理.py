import os
import re
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

from stocknote_holding_analysis import analyze_holding, market_score
from stocknote_tracking import holding_performance, load_active, load_history

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


@st.cache_data(ttl=3600, show_spinner=False)
def company_name_from_code(code):
    """Resolve a real company name from the saved SBI universe, then Yahoo."""
    code = normalize_code(code)
    if not code:
        return ""
    universe_path = os.getenv("STOCKNOTE_UNIVERSE", "data/saved_universe.csv")
    try:
        universe = pd.read_csv(universe_path, dtype=str, encoding="utf-8-sig")
        if "コード" in universe.columns and "銘柄名" in universe.columns:
            codes = universe["コード"].astype(str).map(normalize_code)
            matched = universe.loc[codes == code, "銘柄名"].dropna()
            if not matched.empty and str(matched.iloc[0]).strip():
                return str(matched.iloc[0]).strip()
    except Exception:
        pass
    try:
        info = yf.Ticker(f"{code}.T").info or {}
        return str(info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def get_cash():
    with db() as con:
        row = con.execute("SELECT value FROM settings WHERE key='cash'").fetchone()
    return float(row['value']) if row else 0.0


def set_cash(value):
    with db() as con:
        con.execute("INSERT INTO settings(key,value) VALUES('cash',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (float(value),))
        con.commit()


def entry_cash_effect(side, shares, price):
    """Cash movement recorded when an open position is first registered."""
    amount = float(shares) * float(price)
    return -amount if side == '買い' else amount


def update_open_holding(holding_id, code, name, side, trade_date, shares, entry_price, note):
    """Correct one open entry and apply only its cash difference."""
    with db() as con:
        old = con.execute("SELECT * FROM holdings WHERE id=? AND status='保有中'", (int(holding_id),)).fetchone()
        if old is None:
            raise ValueError("編集対象の保有銘柄が見つかりません。")
        old_effect = entry_cash_effect(old['side'], old['shares'], old['entry_price'])
        new_effect = entry_cash_effect(side, shares, entry_price)
        con.execute(
            """UPDATE holdings
               SET code=?, name=?, side=?, trade_date=?, shares=?, entry_price=?, note=?
               WHERE id=? AND status='保有中'""",
            (code, name, side, trade_date, float(shares), float(entry_price), note, int(holding_id)),
        )
        con.commit()
    set_cash(max(0.0, get_cash() + new_effect - old_effect))


def delete_open_holding(holding_id):
    """Delete only the selected open entry and undo its original cash movement."""
    with db() as con:
        old = con.execute("SELECT * FROM holdings WHERE id=? AND status='保有中'", (int(holding_id),)).fetchone()
        if old is None:
            raise ValueError("削除対象の保有銘柄が見つかりません。")
        old_effect = entry_cash_effect(old['side'], old['shares'], old['entry_price'])
        con.execute("DELETE FROM holdings WHERE id=? AND status='保有中'", (int(holding_id),))
        con.commit()
    set_cash(max(0.0, get_cash() - old_effect))


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


@st.cache_data(ttl=600, show_spinner=False)
def current_holding_analysis(codes):
    """Analyze registered holdings directly; scanner ranking changes do not affect this."""
    current_market = market_score()
    return {code: analyze_holding(code, current_market) for code in codes}


normalized_open_codes = tuple(dict.fromkeys(normalize_code(c) for c in all_codes if normalize_code(c)))
holding_analysis = current_holding_analysis(normalized_open_codes) if normalized_open_codes else {}

market_value = 0.0
unrealized = 0.0
unrealized_loss = 0.0
unrealized_profit = 0.0
missing_price_count = 0
if not open_df.empty:
    for _, r in open_df.iterrows():
        cur = prices.get(normalize_code(r['code']))
        if cur is None:
            missing_price_count += 1
            continue
        qty = float(r['shares'])
        entry = float(r['entry_price'])
        performance = holding_performance(entry, cur, qty, r['side'])
        if r['side'] == '買い':
            market_value += cur * qty
        else:
            market_value += entry * qty
        if performance:
            unrealized += performance['pnl']
            unrealized_loss += performance['loss']
            unrealized_profit += performance['profit']

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

st.markdown("## 💰 現在資金・含み損益")
c1, c2, c3, c4 = st.columns(4)
c1.metric("現金", f"¥{cash:,.0f}")
c2.metric("保有株評価額", f"¥{market_value:,.0f}")
c3.metric("総資産", f"¥{total_assets:,.0f}")
c4.metric("含み損益合計", f"¥{unrealized:,.0f}")
u1, u2, u3 = st.columns(3)
u1.metric("🔴 含み損合計", f"-¥{abs(unrealized_loss):,.0f}" if unrealized_loss < 0 else "¥0")
u2.metric("🟢 含み益合計", f"+¥{unrealized_profit:,.0f}" if unrealized_profit > 0 else "¥0")
u3.metric("現在値取得済み", f"{len(open_df) - missing_price_count}/{len(open_df)}銘柄")
if missing_price_count:
    st.warning(f"現在値を取得できない{missing_price_count}銘柄は、含み損益合計の計算対象外です。推測値は使用していません。")
st.caption(f"実現損益累計: ¥{realized:,.0f}")

with st.expander("⚙️ 現在の現金残高を設定"):
    new_cash = st.number_input("現在の現金残高", min_value=0.0, value=float(cash), step=10000.0)
    if st.button("現金残高を保存"):
        set_cash(new_cash)
        st.success("現金残高を更新しました。")
        st.rerun()

# ======================== 実保有 ========================
st.markdown("## 🧾 実際に買った・空売りした銘柄")
register_code_raw = st.text_input("証券コードを入力", placeholder="例 7203", key="register_code_lookup")
code = normalize_code(register_code_raw)
resolved_name = company_name_from_code(code) if code else ""
if resolved_name:
    st.success(f"銘柄名：{resolved_name}")
elif code:
    st.warning("銘柄名を自動取得できませんでした。名称を確認して手入力してください。")
with st.form("holding_form", clear_on_submit=True):
    b, c, d = st.columns(3)
    name = b.text_input("銘柄名", value=resolved_name, key=f"register_name_{code or 'empty'}")
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
        normalized_code = normalize_code(r['code'])
        cur = prices.get(normalized_code)
        performance = holding_performance(r['entry_price'], cur, r['shares'], r['side'])
        analysis = holding_analysis.get(normalized_code, {})
        rows.append({
            'ID': int(r['id']), '区分': r['side'], 'コード': r['code'], '銘柄名': r['name'], '株数': r['shares'],
            '取得/建値': r['entry_price'], '現在値': cur,
            '建値からの差/1株': performance['per_share'] if performance else None,
            '含み損益': performance['pnl'] if performance else None,
            '買値からの騰落率%': performance['pnl_pct'] if performance else None,
            '判定': ('含み損' if performance['pnl'] < 0 else '含み益' if performance['pnl'] > 0 else 'トントン') if performance else '現在値未取得',
            '現在の総合得点': analysis.get('total_score'),
            'テクニカル点': analysis.get('technical_score'),
            'ファンダメンタル点': analysis.get('fundamental_score'),
            '市場環境点': analysis.get('market_score'),
            'RSI14': analysis.get('rsi'), 'PER': analysis.get('per'),
            '一目位置': analysis.get('cloud_position'), '現在の評価理由': analysis.get('trend_reason') or analysis.get('error'),
            '取引日': r['trade_date']
        })
    holdings_view = pd.DataFrame(rows)
    st.markdown("### 📊 保有銘柄の現在評価")
    st.caption("再スキャンのランキングとは別に、登録済み銘柄を現在データで直接再分析しています。")
    st.dataframe(holdings_view, hide_index=True, use_container_width=True,
                 column_config={
                     "現在の総合得点": st.column_config.NumberColumn(format="%.1f / 100"),
                     "テクニカル点": st.column_config.NumberColumn(format="%.1f"),
                     "ファンダメンタル点": st.column_config.NumberColumn(format="%.1f"),
                     "市場環境点": st.column_config.NumberColumn(format="%.1f"),
                     "買値からの騰落率%": st.column_config.NumberColumn(format="%.2f%%"),
                 })

    score_rows = [r for r in rows if r.get('現在の総合得点') is not None]
    if score_rows:
        selected_score_code = st.selectbox(
            "詳細を見る保有銘柄",
            [r['コード'] for r in score_rows],
            format_func=lambda c: next(f"{r['コード']} {r['銘柄名']}" for r in score_rows if r['コード'] == c),
        )
        selected = next(r for r in score_rows if r['コード'] == selected_score_code)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("現在の総合得点", f"{selected['現在の総合得点']:.1f}/100")
        s2.metric("テクニカル", f"{selected['テクニカル点']:.1f}/100")
        s3.metric("ファンダメンタル", f"{selected['ファンダメンタル点']:.1f}/100")
        s4.metric("市場環境", f"{selected['市場環境点']:.1f}/100")
        st.caption("総合得点 = テクニカル50% + ファンダメンタル35% + 市場環境15%")

    st.markdown("### ✏️ 登録内容の編集・削除")
    edit_ids = [int(x) for x in open_df['id'].tolist()]
    edit_id = st.selectbox("編集・削除するID", edit_ids, key="edit_holding_id")
    edit_row = open_df[open_df['id'] == edit_id].iloc[0]
    try:
        initial_date = date.fromisoformat(str(edit_row['trade_date']))
    except ValueError:
        initial_date = date.today()
    edit_code_raw = st.text_input("編集する証券コード", value=str(edit_row['code']), key=f"edit_code_{edit_id}")
    edit_code_normalized = normalize_code(edit_code_raw)
    resolved_edit_name = company_name_from_code(edit_code_normalized) if edit_code_normalized else ""
    if resolved_edit_name:
        st.caption(f"自動取得した銘柄名：{resolved_edit_name}")
    with st.form("edit_holding_form"):
        eb, ec, ed = st.columns(3)
        default_edit_name = resolved_edit_name or str(edit_row['name'] or '')
        edit_name = eb.text_input("銘柄名", value=default_edit_name,
                                  key=f"edit_name_{edit_id}_{edit_code_normalized or 'empty'}")
        side_options = ["買い", "空売り"]
        edit_side = ec.selectbox("区分", side_options,
                                 index=side_options.index(edit_row['side']) if edit_row['side'] in side_options else 0,
                                 key="edit_side")
        edit_date = ed.date_input("取引日", value=initial_date, key="edit_trade_date")
        ee, ef, eg = st.columns(3)
        edit_shares = ee.number_input("株数", min_value=1.0, step=100.0,
                                      value=float(edit_row['shares']), key="edit_shares")
        edit_price = ef.number_input("買値 / 売建値", min_value=0.0, step=1.0,
                                     value=float(edit_row['entry_price']), key="edit_entry_price")
        edit_note = eg.text_input("メモ", value=str(edit_row['note'] or ''), key="edit_note")
        edit_submit = st.form_submit_button("変更を保存")
        if edit_submit:
            edit_code = edit_code_normalized
            if not edit_code or edit_price <= 0:
                st.error("証券コードと価格を正しく入力してください。")
            else:
                update_open_holding(edit_id, edit_code, edit_name, edit_side,
                                    edit_date.isoformat(), edit_shares, edit_price, edit_note)
                st.success("登録内容を修正し、差額を現金残高へ反映しました。")
                st.rerun()

    with st.expander("🗑️ 間違って登録した銘柄を削除"):
        st.warning(f"ID {edit_id}：{edit_row['code']} {edit_row['name'] or ''} だけを削除します。決済ではなく、誤登録の取り消しです。")
        confirm_delete = st.checkbox("この1件を削除することを確認しました", key=f"confirm_delete_{edit_id}")
        if st.button("この登録を削除", type="secondary", disabled=not confirm_delete,
                     key=f"delete_holding_{edit_id}"):
            delete_open_holding(edit_id)
            st.success("誤登録を削除し、登録時の現金増減を元に戻しました。")
            st.rerun()

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
tracking_code_raw = st.text_input("候補コードを入力", placeholder="例 7203", key="tracking_code_lookup")
t_code = normalize_code(tracking_code_raw)
resolved_tracking_name = company_name_from_code(t_code) if t_code else ""
if resolved_tracking_name:
    st.caption(f"銘柄名：{resolved_tracking_name}")
with st.form("tracking_form", clear_on_submit=True):
    b, c, d = st.columns(3)
    t_name = b.text_input("候補銘柄名", value=resolved_tracking_name,
                          key=f"tracking_name_{t_code or 'empty'}")
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
