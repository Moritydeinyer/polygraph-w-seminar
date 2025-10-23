# api_server.py
"""
Simple FastAPI server that accepts sensor uploads (JSON) authenticated by api_token.
Writes to SQLite DB file 'polygraph.db' (same DB used by Streamlit).
Run with:
    uvicorn api_server:app --host 0.0.0.0 --port 5000
"""
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import uuid

# DB config (same DB as API)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///polygraph.db")
engine = sa.create_engine(DATABASE_URL)
metadata = sa.MetaData()

# Tables (if not exist)
users = sa.Table(
    "users", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("username", sa.String, unique=True, nullable=False),
    sa.Column("password_hash", sa.String, nullable=False),
)

api_tokens = sa.Table(
    "api_tokens", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("token", sa.String, unique=True, nullable=False),
    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
    sa.Column("name", sa.String, nullable=True),
    sa.Column("created_at", sa.DateTime, default=datetime.utcnow),
)

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
SessionLocal = sessionmaker(bind=engine)

# FastAPI app
app = FastAPI(title="Polygraph Node API")

class UploadPayload(BaseModel):
    token: str
    device_id: str
    gsr: float | None = None
    pulse: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    metadata: dict | None = None

def verify_token(db, token_str: str):
    q = sa.select(api_tokens.c.id, api_tokens.c.user_id).where(api_tokens.c.token == token_str)
    r = db.execute(q).fetchone()
    return r

@app.post("/upload")
def upload(payload: UploadPayload):
    db = SessionLocal()
    try:
        tk = verify_token(db, payload.token)
        if not tk:
            raise HTTPException(status_code=401, detail="Invalid API token")
        ins = measurements.insert().values(
            device_id=payload.device_id,
            gsr=payload.gsr,
            pulse=payload.pulse,
            humidity=payload.humidity,
            pressure=payload.pressure,
            metadata=str(payload.metadata) if payload.metadata else None,
            timestamp=datetime.utcnow()
        )
        db.execute(ins)
        db.commit()
        return {"status":"ok"}
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status":"ok", "time": datetime.utcnow().isoformat()}
