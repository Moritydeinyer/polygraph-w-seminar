# streamlit_app.py
"""
Streamlit dashboard for Polygraph lab.
Features:
 - Register / Login (simple, SQLite)
 - Create / revoke API tokens for sensor nodes
 - Live view of incoming measurements (polling DB)
 - Baseline capture (per device), manual input of weight/humidity metadata per trial
 - Recording control (start/stop) to tag saved trials
 - CSV export and PDF report generation (basic)
"""

import streamlit as st
import time, os, hashlib, secrets, io
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import plotly.express as px
from passlib.hash import bcrypt
import json
from streamlit_autorefresh import st_autorefresh


# DB config (same DB as API)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///polygraph.db")
engine = sa.create_engine(DATABASE_URL)
metadata = sa.MetaData()
SessionLocal = sessionmaker(bind=engine)


# Reflect tables (created by API server if used)
metadata.reflect(bind=engine)
users = metadata.tables.get("users")
api_tokens = metadata.tables.get("api_tokens")
measurements = metadata.tables.get("measurements")

# Create tables if not present (in case UI launched first)
if users is None:
    users = sa.Table(
        "users", metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("username", sa.String, unique=True, nullable=False),
        sa.Column("password_hash", sa.String, nullable=False),
    )
if api_tokens is None:
    api_tokens = sa.Table(
        "api_tokens", metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("token", sa.String, unique=True, nullable=False),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("name", sa.String, nullable=True),
        sa.Column("created_at", sa.DateTime, default=datetime.utcnow),
        sa.Column("config", sa.JSON, nullable=True, default={}),
    )
if measurements is None:
    measurements = sa.Table(
        "measurements", metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("device_id", sa.String, nullable=False),
        sa.Column("gsr", sa.Float),
        sa.Column("pulse", sa.Float),
        sa.Column("humidity", sa.Float),
        sa.Column("pressure", sa.Float),
        sa.Column("metadata", sa.String, nullable=True),
        sa.Column("timestamp", sa.DateTime, default=datetime.utcnow),
    )
if "config" not in api_tokens.c:
    api_tokens.append_column(
        sa.Column("config", sa.String, nullable=True)  
    )
metadata.create_all(engine)

# ---------- Helpers ----------


import io, ast, json
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime


def generate_experiment_report(df: pd.DataFrame, baseline_df: pd.DataFrame, device_filter: str = None) -> bytes:
    """
    Generates a PDF report for GSR sensor experiments comparing recorded data to baseline.
    Now shows only mean GSR per X-value with ±1σ error bars to reduce Y-axis spread.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    # --- parse metadata ---
    df['metadata'] = df['metadata'].apply(parse_metadata)
    baseline_df['metadata'] = baseline_df['metadata'].apply(parse_metadata)

    # --- optional filter ---
    if device_filter:
        df = df[df['device_id'] == device_filter]
        baseline_df = baseline_df[baseline_df['device_id'] == device_filter]

    devices = df['device_id'].unique()

    # --- header ---
    content.append(Paragraph("Polygraph Lab — Automatic Experiment Report", styles['Title']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()} UTC", styles['Normal']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(
        "Experiment Objective: How do physical factors—in particular contact pressure and "
        "moisture—affect the measurement accuracy of a lie detector based on galvanic skin response (GSR)?",
        styles['Normal']
    ))
    content.append(Spacer(1, 24))


    # --- device loop ---
    for device in devices:
        content.append(Paragraph(f"Device: {device}", styles['Heading2']))

        device_df = df[df['device_id'] == device]
        device_baseline = baseline_df[baseline_df['device_id'] == device]

        if device_baseline.empty:
            content.append(Paragraph("No baseline data available.", styles['Normal']))
            content.append(Spacer(1, 24))
            continue

        # === BASELINE SECTION ===
        baseline_mean = device_baseline['gsr'].mean(skipna=True)
        baseline_std = device_baseline['gsr'].std(skipna=True)
        base_press = device_baseline['pressure'].mean(skipna=True)
        base_hum = device_baseline['humidity'].mean(skipna=True)

        content.append(Paragraph(f"GSR Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}", styles['Normal']))
        content.append(Paragraph(f"Baseline Pressure: {base_press:.1f} g | Baseline Humidity: {base_hum:.1f} %", styles['Normal']))
        content.append(Spacer(1, 12))

        # --- plot baseline over time ---
        base_sorted = device_baseline.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(base_sorted['timestamp'], base_sorted['gsr'], label='Baseline GSR', color='r')
        ax.set_title(f"Baseline GSR vs Time - {device}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("GSR")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='PNG', bbox_inches='tight')
        img_buf.seek(0)
        content.append(Image(img_buf, width=450, height=200))
        plt.close(fig)
        content.append(Spacer(1, 24))

        # === HUMIDITY ANALYSIS (Mean + σ) ===
        hum_levels = sorted(device_df['humidity'].dropna().unique())
        for hum in hum_levels:
            hum_df = device_df[device_df['humidity'] == hum]
            if hum_df.empty:
                continue

            # Gruppieren nach Pressure
            grouped = hum_df.groupby('pressure')['gsr']
            x_vals = []
            y_means = []
            y_errs = []
            for press, group in grouped:
                x_vals.append(press)
                y_means.append(group.mean())
                y_errs.append(group.std())

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.errorbar(x_vals, y_means, yerr=y_errs, fmt='o', color='blue',
                        ecolor='gray', capsize=5, label='Mean ±1σ')
            ax.axhline(y=baseline_mean, color='r', linestyle='--', label='Baseline')
            ax.set_xlabel("Pressure (g)")
            ax.set_ylabel("GSR")
            ax.set_title(f"GSR vs Pressure @ {hum:.1f}% Humidity (Mean ± σ)")
            ax.legend()
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='PNG', bbox_inches='tight')
            img_buf.seek(0)
            content.append(Image(img_buf, width=450, height=200))
            plt.close(fig)
            content.append(Spacer(1, 24))

        # === PRESSURE ANALYSIS (Mean + σ) ===
        pres_levels = sorted(device_df['pressure'].dropna().unique())
        for pres in pres_levels:
            pres_df = device_df[device_df['pressure'] == pres]
            if pres_df.empty:
                continue

            grouped = pres_df.groupby('humidity')['gsr']
            x_vals = []
            y_means = []
            y_errs = []
            for hum_val, group in grouped:
                x_vals.append(hum_val)
                y_means.append(group.mean())
                y_errs.append(group.std())

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.errorbar(x_vals, y_means, yerr=y_errs, fmt='o', color='green',
                        ecolor='gray', capsize=5, label='Mean ±1σ')
            ax.axhline(y=baseline_mean, color='r', linestyle='--', label='Baseline')
            ax.set_xlabel("Humidity (%)")
            ax.set_ylabel("GSR")
            ax.set_title(f"GSR vs Humidity @ {pres:.1f} g Pressure (Mean ± σ)")
            ax.legend()
            plt.tight_layout()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='PNG', bbox_inches='tight')
            img_buf.seek(0)
            content.append(Image(img_buf, width=450, height=200))
            plt.close(fig)
            content.append(Spacer(1, 24))

        # === HISTOGRAM ===
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(device_df['gsr'].dropna(), bins=20, color='gray', alpha=0.7)
        ax.set_title(f"GSR Distribution - {device}")
        ax.set_xlabel("GSR")
        ax.set_ylabel("Count")
        plt.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='PNG', bbox_inches='tight')
        img_buf.seek(0)
        content.append(Image(img_buf, width=450, height=200))
        plt.close(fig)
        content.append(Spacer(1, 24))

        # === OVERALL OPTIMAL PRESSURE ===
        pressure_means = device_df.groupby('pressure')['gsr'].mean()
        if not pressure_means.empty:
            opt_pressure = pressure_means.idxmax()
            content.append(Paragraph(f"Overall optimal pressure: {opt_pressure:.1f} g", styles['Normal']))
        else:
            content.append(Paragraph("Overall optimal pressure: N/A", styles['Normal']))
        content.append(Spacer(1, 36))

    # === finish ===
    doc.build(content)
    buf.seek(0)
    return buf.read()







def parse_metadata(x):
    """Safe parsing of metadata: JSON or Python dict string."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except:
                return {}
    elif isinstance(x, dict):
        return x
    return {}


def get_db():
    return SessionLocal()

import hashlib
from passlib.hash import bcrypt

def hash_password(pw: str):
    pw_hash = hashlib.sha256(pw.encode('utf-8')).hexdigest()
    return pw_hash

def verify_password(pw: str, hashed: str):
    pw_hash = hashlib.sha256(pw.encode('utf-8')).hexdigest()
    if pw_hash == hashed:
        return True
    return False

def create_user(username, password):
    db = get_db()
    try:
        exists = db.execute(sa.select(users.c.id).where(users.c.username == username)).fetchone()
        if exists:
            return False, "User exists"
        ins = users.insert().values(username=username, password_hash=hash_password(password))
        db.execute(ins)
        db.commit()
        return True, "ok"
    finally:
        db.close()

def authenticate_user(username, password):
    db = get_db()
    try:
        row = db.execute(sa.select(users).where(users.c.username == username)).fetchone()
        if not row:
            return False
        return verify_password(password, row.password_hash)
    finally:
        db.close()

def create_api_token(user, name="default"):
    db = get_db()
    try:
        token = secrets.token_hex(32)
        ins = api_tokens.insert().values(token=token, user_id=user['id'], name=name, created_at=datetime.utcnow())
        db.execute(ins)
        db.commit()
        return token
    finally:
        db.close()

def list_tokens(user):
    db = get_db()
    try:
        rows = db.execute(sa.select(api_tokens).where(api_tokens.c.user_id == user['id']).order_by(api_tokens.c.created_at.desc())).fetchall()
        return rows
    finally:
        db.close()

def revoke_token(token_id):
    db = get_db()
    try:
        db.execute(api_tokens.delete().where(api_tokens.c.id == token_id))
        db.commit()
        return True
    finally:
        db.close()

def update_recording_flag(user_id, recording: bool, hum, press):
    db = get_db()
    try:
        tokens = db.execute(sa.select(api_tokens).where(api_tokens.c.user_id == user_id)).fetchall()
        for t in tokens:
            cfg = json.loads(t.config) if t.config else {}
            cfg['recording'] = recording
            cfg['hum'] = hum
            cfg['press'] = press
            db.execute(api_tokens.update().where(api_tokens.c.id == t.id).values(config=json.dumps(cfg)))
        db.commit()
    finally:
        db.close()

def get_user_devices(user_id):
    db = get_db()
    try:
        tokens = db.execute(sa.select(api_tokens).where(api_tokens.c.user_id == user_id)).fetchall()
        devices = [f"device_{t.id}" for t in tokens if t.id]
        return list(set(devices))
    finally:
        db.close()


def query_measurements(user_id, limit=1000, since_minutes=60, device_id=None):
    db = get_db()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        sel = sa.select(measurements).where(measurements.c.timestamp >= cutoff).order_by(measurements.c.timestamp.desc())
      
        user_devices = get_user_devices(user_id)
        if not user_devices:
            return pd.DataFrame()  

        sel = sel.where(measurements.c.device_id.in_(user_devices))

        if device_id and device_id in user_devices:
            sel = sel.where(measurements.c.device_id == device_id)

        rows = db.execute(sel.limit(limit)).fetchall()
        data = [dict(r._mapping) for r in rows]
        df = pd.DataFrame(data)
        if not df.empty and 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(parse_metadata) 
        return df
    finally:
        db.close()



# ---------- Streamlit UI ----------
st.set_page_config(page_title="Polygraph Lab", layout="wide")
st.title("Dashboard")

# Authentication simple flow (session_state)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# ---------- Auth  ----------
st.sidebar.title("Polygraph Lab")

# Login / Logout Info
if not st.session_state.get("authenticated", False):
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username", key="login_user")
    password = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.sidebar.error("Login failed")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sign up")
    new_user = st.sidebar.text_input("New Username", key="reg_user")
    new_pass = st.sidebar.text_input("New Password", type="password", key="reg_pass")
    if st.sidebar.button("Sign up"):
        ok, msg = create_user(new_user, new_pass)
        if ok:
            st.sidebar.success("User created — login now.")
        else:
            st.sidebar.error(msg)

else:
    st.sidebar.success(f"Signed in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


    
    db = get_db()
    user_row = db.execute(sa.select(users).where(users.c.username == st.session_state.username)).fetchone()
    db.close()
    user = dict(user_row._mapping) if user_row else None

    # --- Token management ---
    st.header("API Tokens (for Sensor-Clients)")
    col1, col2 = st.columns([2,1])
    with col1:
        token_name = st.text_input("Token-Name (e.g. pi_zero_2w)")
    with col2:
        if st.button("Create Token"):
            new_token = create_api_token(user, token_name or "unnamed")
            st.success("Token created — copy (just displyed once).")
            st.code(new_token)

    # list tokens
    tokens = list_tokens(user)
    if tokens:
        rows = []
        for t in tokens:
            rows.append([t.id, t.name, t.token[:8]+"..."+t.token[-8:], t.created_at])
        st.table(pd.DataFrame(rows, columns=["id","name","token","created_at"]))

        t_revoke = st.number_input("ID to delete", min_value=0, step=1, value=0)
        if st.button("Delete Token"):
            if t_revoke>0:
                revoke_token(t_revoke)
                st.rerun()

    st.markdown("---")


    from datetime import datetime
    from streamlit_autorefresh import st_autorefresh

    
    def get_node_status(user_id):
        """
        Returns a DataFrame of devices (nodes) owned by this user, with last seen, lag, and online status.
        """
        db = get_db()
        try:
            # Hol alle Tokens des Users
            tokens = db.execute(sa.select(api_tokens).where(api_tokens.c.user_id == user_id)).fetchall()
            devices = [f"device_{t.id}" for t in tokens if t.name]
            status_list = []

            now = datetime.utcnow()
            for device in devices:
                last_row = db.execute(
                    sa.select(measurements)
                    .where(measurements.c.device_id == device)
                    .order_by(measurements.c.timestamp.desc())
                    .limit(1)
                ).fetchone()

                if last_row:
                    last_seen = last_row.timestamp
                    lag = (now - last_seen).total_seconds()
                    online = lag < 5
                    display_lag = round(lag, 1) if lag < 120 else "-"
                else:
                    last_seen = None
                    lag = None
                    online = False
                    display_lag = "-"

                status_list.append({
                    "Device": device,
                    "Last Seen": last_seen.strftime("%H:%M:%S") if last_seen else "Never",
                    "Lag (s)": display_lag,
                    "Status": "🟢 Online" if online else "🔴 Offline"
                })

            return pd.DataFrame(status_list)
        finally:
            db.close()


    st.header("Node Overview")

    placeholder = st.empty()

    with placeholder.container():
        df_nodes = get_node_status(user['id'])
        if not df_nodes.empty:
            st.table(df_nodes)
        else:
            st.info("No nodes registered / measurements available yet.")

    
    st.header("Global Token Configuration")

    # --- Global configuration inputs ---
    interval = st.number_input("Interval (s)", min_value=1, max_value=60, value=2, step=1)
    gsr_min = st.number_input("GSR min (Simulation only)", value=0.1, step=0.1)
    gsr_max = st.number_input("GSR max (Simulation only)", value=10.0, step=0.1)
    pulse_min = st.number_input("Pulse min (Simulation only)", value=60, step=1)
    pulse_max = st.number_input("Pulse max (Simulation only)", value=100, step=1)
    pressure_val = st.number_input("Pressure (kPa)", min_value=0.0, step=1.0, value=st.session_state.get("pressure_value", 100.0))
    humidity_val = st.number_input("Humidity (%)", min_value=0.0, step=1.0, value=st.session_state.get("humidity_value", 50.0))

    if st.button("Apply Config to All Nodes"):
        db = get_db()
        for t in db.execute(sa.select(api_tokens)).fetchall():
            cfg = json.loads(t.config) if t.config else {}
            cfg.update({
                "interval": interval,
                "gsr_range": [gsr_min, gsr_max],
                "pulse_range": [pulse_min, pulse_max],
                "pressure": pressure_val,
                "humidity": humidity_val
            })
            db.execute(api_tokens.update().where(api_tokens.c.id == t.id).values(config=json.dumps(cfg)))
        db.commit()
        db.close()
        st.success("Configuration applied to all nodes.")




    # --- Live view & controls ---
    st.header("Live Sensor View & Recording")
    user_devices = get_user_devices(user['id'])
    selected_device = st.selectbox("Select Device (Live)", options=["all"]+user_devices)
    since_minutes = st.slider("Period (min)", 1, 10080, 60)
    refresh = st.slider("Refresh (s)", 1, 5, 2)

    # Baseline & experiment controls
    st.subheader("Experiment Controls")

    if "refresh_active" not in st.session_state:
        st.session_state.refresh_active = True

    refresh_col1, refresh_col2 = st.columns(2)
    if refresh_col1.button("Stop Live Refresh"):
        st.session_state.refresh_active = False
    if refresh_col2.button("Start Live Refresh"):
        st.session_state.refresh_active = True

    if st.session_state.refresh_active:
        count = st_autorefresh(interval=refresh*1000, limit=None, key="node_overview_refresh")


    if st.button("Record Baseline (10 s)"):
        db = get_db()
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=10)

            df_recent = pd.DataFrame([
                dict(r._mapping) for r in db.execute(
                    sa.select(measurements)
                    .where(measurements.c.timestamp >= cutoff)
                ).fetchall()
            ])
            
            if df_recent.empty:
                st.warning("No measurement data found for baseline capture.")
            else:
                for row in df_recent.itertuples():
                    meta = parse_metadata(row.metadata)
                    meta['baseline'] = True
                    db.execute(
                        measurements.update()
                        .where(measurements.c.id == row.id)
                        .values(metadata=json.dumps(meta))
                    )
                db.commit()

                baseline_mean = df_recent['gsr'].mean()
                st.session_state["baseline"] = baseline_mean
                st.success(f"Baseline recorded over last 10 s: mean GSR = {baseline_mean:.3f}")
        finally:
            db.close()


    # --- Experiment Controls ---
    if "recording" not in st.session_state:
        st.session_state.recording = False

    rec_col1, rec_col2 = st.columns(2)

    if rec_col1.button("Start Recording"):
        st.session_state.recording = True
        st.session_state.record_start = datetime.utcnow().isoformat()
        st.session_state.pressure_value = pressure_val
        st.session_state.humidity_value = humidity_val
        update_recording_flag(user['id'], True, humidity_val, pressure_val)

    if rec_col2.button("Stop Recording"):
        st.session_state.recording = False
        update_recording_flag(user['id'], False, humidity_val, pressure_val)



    st.write("Recording status:", st.session_state.recording)

    # -------- Live Table + Chart (partial refresh) --------
    placeholder = st.empty()

    with placeholder.container():
        df = query_measurements(user['id'], limit=1000, since_minutes=since_minutes,
                                device_id=None if selected_device=="all" else selected_device)

        if df is not None and not df.empty:
            st.subheader("Live Data")
            st.dataframe(df.head(200), use_container_width=True)
            fig = px.line(df.sort_values("timestamp"),
                        x="timestamp", y="gsr", color="device_id",
                        title="GSR over Time")
            st.plotly_chart(fig, use_container_width=True, key="live_chart")
        else:
            st.warning("No data available.")






    # Export CSV
    if st.button("Export CSV (recent period)"):
        if df.empty:
            st.warning("No data available for export.")
        else:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="measurements_export.csv", mime="text/csv")


    if st.button("Generate PDF Report (EXPERIMENTAL v.1.5)"):
        df_all = query_measurements(user['id'], limit=50000, since_minutes=7*24*60)         ### INPUT
        if df_all.empty:
            st.warning("No data available for report.")
        else:
            df_all = df_all[df_all['metadata'].notnull()]
            df_all['metadata'] = df_all['metadata'].apply(parse_metadata)  

            df_record = df_all[df_all['metadata'].apply(lambda m: m.get('recording') or m.get('baseline', False))]
            df_baseline = df_all[df_all['metadata'].apply(lambda m: m.get('baseline', False))]

            if df_record.empty or df_baseline.empty:
                st.warning("Not enough data for PDF report (need recordings + baseline).")
            else:
                pdf_bytes = generate_experiment_report(
                    df_record,
                    df_baseline,
                    device_filter=None                                          ### INPUT
                )
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="experiment_report.pdf",
                    mime="application/pdf"
                )






