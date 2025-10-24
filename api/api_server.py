# api_server.py
"""
FastAPI server for Polygraph sensor nodes.
Receives JSON uploads authenticated by api_token.
Writes to SQLite DB 'polygraph.db'.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import uuid
import json

# DB config
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///polygraph.db")
engine = sa.create_engine(DATABASE_URL)
metadata = sa.MetaData()

# Tables
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
    sa.Column("config", sa.JSON, nullable=True, default={}),
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
    q = sa.select(api_tokens.c.id, api_tokens.c.user_id, api_tokens.c.config).where(api_tokens.c.token == token_str)
    return db.execute(q).fetchone()

@app.post("/api/upload")
def upload(payload: UploadPayload):
    db = SessionLocal()
    try:
        tk = verify_token(db, payload.token)
        if not tk:
            raise HTTPException(status_code=401, detail="Invalid API token")

        cfg = json.loads(tk.config) if tk.config else {}
        recording = cfg.get("recording", False)  # Default: False

        meta = payload.metadata.copy() if payload.metadata else {}
        meta["recording"] = recording

        ins = measurements.insert().values(
            device_id=payload.device_id,
            gsr=payload.gsr,
            pulse=payload.pulse,
            humidity=payload.humidity,
            pressure=payload.pressure,
            metadata=str(meta) if meta else None,
            timestamp=datetime.utcnow()
        )
        db.execute(ins)
        db.commit()
        return {"status": "ok", "recording": recording}
    finally:
        db.close()

@app.get("/api/config")
def get_node_config(token: str):
    db = SessionLocal()
    try:
        tk = verify_token(db, token)
        if not tk:
            raise HTTPException(status_code=401, detail="Invalid API token")

        if tk.config:
            config = json.loads(tk.config)
            config.setdefault("device_id", f"device_{tk.id}")
        else:
            config = {
                "device_id": f"device_{tk.id}",
                "interval": 2,
                "gsr_range": [0.1, 10.0],
                "pulse_range": [60, 100],
                "pressure": 100.0,
                "humidity": 50.0,
            }
        return config
    finally:
        db.close()



@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

