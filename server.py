from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import os
import uuid
import jwt
import bcrypt
import io
import csv
import qrcode
import logging

# ------------------ Setup ------------------

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
JWT_SECRET = os.getenv("JWT_SECRET")
BASE_URL = os.getenv("BASE_URL", "http://localhost:3000")
ENV = os.getenv("ENV", "development")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer()

JWT_ALGORITHM = "HS256"

# ------------------ Models ------------------

class FeedbackCreate(BaseModel):
    institution: str
    mess: str
    meal_type: str
    food_quality: int = Field(ge=1, le=5)
    taste: int = Field(ge=1, le=5)
    hygiene: int = Field(ge=1, le=5)
    portion_size: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(default="", max_length=200)
    device_fingerprint: str
    interaction_time: int


class Feedback(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    institution: str
    mess: str
    meal_type: str
    food_quality: int
    taste: int
    hygiene: int
    portion_size: int
    comment: str
    device_fingerprint: str
    confidence_score: float
    validated: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdminUser(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    password_hash: str
    role: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdminLogin(BaseModel):
    email: EmailStr
    password: str


# ------------------ Helpers ------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_token(user_id: str, email: str, role: str):
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = await db.admin_users.find_one({"id": payload["user_id"]})
        if not user or not user.get("is_active"):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return user
    except:
        raise HTTPException(status_code=401, detail="Invalid token")


def calculate_confidence_score(feedback: FeedbackCreate) -> float:
    score = 1.0

    if feedback.interaction_time < 5:
        score -= 0.4
    elif feedback.interaction_time < 10:
        score -= 0.2

    ratings = [
        feedback.food_quality,
        feedback.taste,
        feedback.hygiene,
        feedback.portion_size
    ]

    if len(set(ratings)) == 1:
        score -= 0.3

    return max(0.1, min(1.0, score))


async def check_duplicate(device_fp: str, meal_type: str):
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    existing = await db.feedback.find_one({
        "device_fingerprint": device_fp,
        "meal_type": meal_type,
        "timestamp": {"$gte": start_of_day}
    })

    return existing is not None


# ------------------ Public Routes ------------------

@api_router.post("/feedback")
async def submit_feedback(feedback: FeedbackCreate):

    if await check_duplicate(feedback.device_fingerprint, feedback.meal_type):
        raise HTTPException(status_code=400, detail="Already submitted today")

    confidence = calculate_confidence_score(feedback)

    if confidence < 0.3:
        raise HTTPException(status_code=400, detail="Low confidence feedback")

    feedback_obj = Feedback(
        **feedback.model_dump(),
        confidence_score=confidence,
        validated=True
    )

    await db.feedback.insert_one(feedback_obj.model_dump())

    return {"success": True}


# ------------------ Auth ------------------

@api_router.post("/auth/login")
async def login(data: AdminLogin):
    user = await db.admin_users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(user["id"], user["email"], user["role"])
    return {"token": token}


# ------------------ Dashboard ------------------

@api_router.get("/dashboard/summary")
async def dashboard_summary(user=Depends(get_current_admin)):
    total = await db.feedback.count_documents({"validated": True})

    pipeline = [
        {"$match": {"validated": True}},
        {"$group": {
            "_id": None,
            "avg_food": {"$avg": "$food_quality"},
            "avg_taste": {"$avg": "$taste"},
            "avg_hygiene": {"$avg": "$hygiene"},
            "avg_portion": {"$avg": "$portion_size"}
        }}
    ]

    result = await db.feedback.aggregate(pipeline).to_list(1)

    if result:
        stats = result[0]
    else:
        stats = {"avg_food": 0, "avg_taste": 0, "avg_hygiene": 0, "avg_portion": 0}

    return {
        "total_feedback": total,
        "avg_food_quality": round(stats["avg_food"] or 0, 2),
        "avg_taste": round(stats["avg_taste"] or 0, 2),
        "avg_hygiene": round(stats["avg_hygiene"] or 0, 2),
        "avg_portion_size": round(stats["avg_portion"] or 0, 2),
    }


# ------------------ AI Insights (Internal Rule-Based) ------------------

@api_router.get("/ai-insights")
async def ai_insights(user=Depends(get_current_admin)):

    feedback_list = await db.feedback.find(
        {"validated": True, "confidence_score": {"$gte": 0.5}}
    ).limit(200).to_list(200)

    if len(feedback_list) < 5:
        return {"message": "Not enough data"}

    avg_hygiene = sum(f["hygiene"] for f in feedback_list) / len(feedback_list)

    recommendations = []

    if avg_hygiene < 3:
        recommendations.append("Improve cleaning and sanitation processes.")
    else:
        recommendations.append("Hygiene standards are stable.")

    return {
        "total_feedback_analyzed": len(feedback_list),
        "average_hygiene": round(avg_hygiene, 2),
        "recommendations": recommendations
    }


# ------------------ QR Code ------------------

@api_router.get("/qr/{mess_name}")
async def generate_qr(mess_name: str, user=Depends(get_current_admin)):

    url = f"{BASE_URL}/?mess={mess_name}"

    qr = qrcode.make(url)

    buffer = io.BytesIO()
    qr.save(buffer)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


# ------------------ Startup ------------------

@app.on_event("startup")
async def startup():

    await db.feedback.create_index("device_fingerprint")
    await db.feedback.create_index("meal_type")
    await db.feedback.create_index("timestamp")
    await db.admin_users.create_index("email", unique=True)

    if ENV == "development":
        owner_exists = await db.admin_users.find_one({"role": "owner"})
        if not owner_exists:
            admin = AdminUser(
                email="admin@mess.com",
                password_hash=hash_password("admin123"),
                role="owner"
            )
            await db.admin_users.insert_one(admin.model_dump())
            logging.info("Dev admin created")


# ------------------ CORS ------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
