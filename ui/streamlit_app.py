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
metadata.create_all(engine)

# ---------- Helpers ----------
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

def insert_measurement(d):
    db = get_db()
    try:
        ins = measurements.insert().values(
            device_id=d.get("device_id"),
            gsr=d.get("gsr"),
            pulse=d.get("pulse"),
            humidity=d.get("humidity"),
            pressure=d.get("pressure"),
            metadata=str(d.get("metadata")) if d.get("metadata") else None,
            timestamp=datetime.utcnow()
        )
        db.execute(ins)
        db.commit()
    finally:
        db.close()

def query_measurements(limit=1000, since_minutes=60, device_id=None):
    db = get_db()
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        sel = sa.select(measurements).where(measurements.c.timestamp >= cutoff).order_by(measurements.c.timestamp.desc())
        if device_id:
            sel = sel.where(measurements.c.device_id == device_id)
        rows = db.execute(sel.limit(limit)).fetchall()
        df = pd.DataFrame(rows, columns=rows[0].keys()) if rows else pd.DataFrame([])
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
                st.experimental_rerun()

    st.markdown("---")

    # --- Live view & controls ---
    st.header("Live Sensor View & Recording")
    devices = db.execute(sa.select(sa.distinct(measurements.c.device_id))).fetchall()
    devices = [d[0] for d in devices] if devices else []
    selected_device = st.selectbox("Select Device (Live)", options=["all"]+devices)
    since_minutes = st.slider("Period (min)", 1, 1440, 60)
    refresh = st.slider("Refresh (s)", 1, 5, 2)

    # Baseline & experiment controls
    st.subheader("Experiment-Parameter")
    weight = st.number_input("Contactpressure (g)", min_value=0.0, step=1.0, value=150.0)
    humidity_level = st.number_input("Moisture content (µL on Pad)", min_value=0.0, step=1.0, value=10.0)
    if st.button("Record Baseline (10 s)"):
        # capture baseline: read latest 10 s of measurements (from selected device)
        df = query_measurements(limit=1000, since_minutes=10, device_id=None if selected_device=="all" else selected_device)
        if df.empty:
            st.warning("No measurement data found for baseline capture.")
        else:
            # take last 10s
            cutoff = datetime.utcnow() - timedelta(seconds=10)
            df_recent = df[df['timestamp'] >= cutoff]
            if df_recent.empty:
                st.warning("No measurement data for the last 10s.")
            else:
                baseline = df_recent['gsr'].mean()
                st.success(f"Baseline (mean GSR over last 10s): {baseline:.3f}")
                st.session_state["baseline"] = baseline

    # Recording controls (tagging)
    if "recording" not in st.session_state:
        st.session_state.recording = False
    rec_col1, rec_col2 = st.columns(2)
    if rec_col1.button("Start Recording"):
        st.session_state.recording = True
        st.session_state.record_start = datetime.utcnow().isoformat()
    if rec_col2.button("Stop Recording"):
        st.session_state.recording = False

    st.write("Recording status:", st.session_state.recording)

    # Show live plot (poll DB)
    placeholder = st.empty()
    stop_button = st.button("Stop Live Refresh")
    keep_running = True

    def build_plot(df_plot):
        if df_plot.empty:
            placeholder.info("No available data.")
            return
        df_plot = df_plot.sort_values("timestamp")
        fig = px.line(df_plot, x="timestamp", y="gsr", color="device_id", title="Live GSR")
        st.plotly_chart(fig, use_container_width=True)

    # Single refresh cycle (non-blocking)
    df = query_measurements(limit=1000, since_minutes=since_minutes, device_id=None if selected_device=="all" else selected_device)
    build_plot(df)

    # Data table and analytics
    st.subheader("Recent messurements (Table)")
    if not df.empty:
        df_display = df.copy()
        df_display['timestamp'] = df_display['timestamp'].astype(str)
        st.dataframe(df_display.head(200))

        # compute simple features per-device
        agg = df.groupby('device_id').agg(
            mean_gsr = sa.func.avg(measurements.c.gsr)
        )
    else:
        st.info("No messurament data available.")

    # Export CSV
    if st.button("Export CSV (recent period)"):
        if df.empty:
            st.warning("No data available for export.")
        else:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="measurements_export.csv", mime="text/csv")

    # Generate PDF report
    if st.button("Generate PDF Report"):
        # build simple PDF with summary stats & plots
        df_all = query_measurements(limit=5000, since_minutes=24*60)
        if df_all.empty:
            st.warning("No data available for Report.")
        else:
            pdf_bytes = generate_report_pdf(df_all, st.session_state.get("baseline", None), weight, humidity_level)
            st.download_button("Download Report", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")


# ---------- Report generation ----------
def generate_report_pdf(df, baseline_val, weight_g, humidity_ul):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("Polygraph Lab — Report", styles['Title']))
    content.append(Spacer(1,12))
    content.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()}", styles['Normal']))
    content.append(Paragraph(f"Baseline (GSR): {baseline_val}", styles['Normal']))
    content.append(Paragraph(f"Weight set: {weight_g} g — Humidity: {humidity_ul} µL", styles['Normal']))
    content.append(Spacer(1,12))

    # Summary stats
    content.append(Paragraph("Summary Stats (last loaded dataset)", styles['Heading2']))
    try:
        stats = df['gsr'].describe().to_frame().to_html()
        content.append(Paragraph("Basic stats included", styles['Normal']))
    except Exception as e:
        content.append(Paragraph("No stats", styles['Normal']))

    # include a time-series plot
    try:
        fig, ax = plt.subplots(figsize=(6,3))
        df_sorted = df.sort_values("timestamp")
        plt.plot(df_sorted['timestamp'], df_sorted['gsr'])
        plt.xticks(rotation=30)
        plt.title("GSR over time")
        plt.tight_layout()
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='PNG')
        img_buf.seek(0)
        content.append(Image(img_buf, width=450, height=200))
    except Exception as e:
        content.append(Paragraph(f"Plot error: {e}", styles['Normal']))

    doc.build(content)
    buf.seek(0)
    return buf.read()
