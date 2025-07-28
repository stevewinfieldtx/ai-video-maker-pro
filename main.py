# main.py - Railway-Ready AI Video Maker with Auth & Payments
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from moviepy import ImageSequenceClip, AudioFileClip, ImageClip
from pydantic import BaseModel
from typing import List, Optional
import os
import base64
import uuid
import asyncio
import aiohttp
from pathlib import Path
import shutil
import random
import hashlib
import jwt
from datetime import datetime, timedelta
import stripe
import sqlalchemy
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt

# Runware imports
try:
    from runware import Runware
    from runware.types import IImageInference
    RUNWARE_AVAILABLE = True
except ImportError:
    RUNWARE_AVAILABLE = False
    print("⚠️ Runware not available - AI image generation disabled")

app = FastAPI(title="AI Video Maker Pro")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and Static Files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Environment Variables (Railway will provide these)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./video_maker.db")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "sk_test_...")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "pk_test_...")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "H2J4fDIX9bPHcr2jItoFJkS3yMqbgM1a")
MUSICAPI_AI_KEY = os.getenv("MUSICAPI_AI_KEY", "f8e3be01ef37a0b3ceff3ea0a389d4da")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBX4Q3dZXzdHIoHDSHX6i5Qhp8jS01wUds")

# Stripe configuration
stripe.api_key = STRIPE_SECRET_KEY

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String, default="none")  # none, standard
    stripe_customer_id = Column(String, nullable=True)
    videos_this_month = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    subscription_expires = Column(DateTime, nullable=True)

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    job_id = Column(String, unique=True)
    title = Column(String)
    status = Column(String)  # processing, completed, failed
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    ai_prompt = Column(String, nullable=True)
    music_genre = Column(String, nullable=True)
    enhanced_prompts_used = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)

# Store job status in memory (you might want to move this to Redis in production)
jobs = {}

# Subscription tiers - SINGLE TIER ONLY
SUBSCRIPTION_TIERS = {
    "standard": {"videos_per_month": 2, "price": 99}  # $0.99
}

# Pydantic models
class VideoRequest(BaseModel):
    images: Optional[List[str]] = None
    ai_prompt: Optional[str] = None
    num_images: Optional[int] = 5
    audio: Optional[str] = None
    seconds_per_image: float = 2.0
    title: str = "My Video"
    add_effects: bool = False
    enhance_prompts: bool = False
    generate_music: bool = False
    music_genre: Optional[str] = None
    music_mood: Optional[str] = None
    music_bpm: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    download_url: str = None
    error: str = None
    static_video_url: str = None
    generated_music_url: str = None

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

# Authentication
security = HTTPBearer()

def get_database():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_database)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def check_video_quota(user: User) -> bool:
    """Check if user can create another video this month"""
    if user.subscription_tier == "none":
        return False  # Must have paid subscription
    
    # Only one tier - check against limit
    limit = SUBSCRIPTION_TIERS["standard"]["videos_per_month"]
    return user.videos_this_month < limit

# Copy all your existing Gemini and video processing functions here
async def enhance_prompts_with_gemini(original_prompt: str, music_genre: str, music_mood: str, seconds_per_image: float, music_bpm: str, image_style: str, num_images: int) -> List[str]:
    """Use Gemini 2.0-Flash to enhance and create varied prompts"""
    try:
        print(f"🧠 Enhancing prompts with Gemini 2.0-Flash...")
        
        context = f"""
        Original user prompt: "{original_prompt}"
        Video style: {image_style}
        Music: {music_genre} {music_mood} at {music_bpm} speed
        Image duration: {seconds_per_image} seconds each
        Number of images needed: {num_images}
        
        GENDER CONSISTENCY RULE:
        - If original prompt mentions "woman", "girl", "female", "lady", "she", "her" → ALL prompts must feature ONLY females
        - If original prompt mentions "man", "boy", "male", "guy", "he", "him" → ALL prompts must feature ONLY males  
        - If no gender specified → choose ONE gender and use consistently throughout ALL prompts

        Your task: Create {num_images} unique image prompts that tell a cohesive visual story.

        REQUIREMENTS:
        1. Each prompt must be EXACTLY 125 tokens (approximately 100-110 words)
        2. Keep the same time period and art style from the original prompt
        3. **CRITICAL: Maintain consistent gender throughout ALL prompts**
        4. Create variety in: poses, clothing, colors, locations, actions (but same gender/person)
        5. No two images should be similar in composition
        6. All images should clearly connect to the original concept
        7. Match the music mood and energy level
        8. Ensure all prompts are SFW (Safe For Work)
        9. Avoid misshapen or unrealistic anatomy descriptions
        10. Use rich, detailed descriptions with cinematic language
        11. Include lighting, composition, and atmospheric details

        Return ONLY a JSON array of prompts, no other text:
        ["prompt 1", "prompt 2", "prompt 3", ...]
        """
        
        payload = {
            "contents": [{
                "parts": [{"text": context}]
            }],
            "generationConfig": {
                "temperature": 0.8,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        async with aiohttp.ClientSession() as session:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
            
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Gemini API error: {response.status} - {error_text}")
                    raise Exception(f"Gemini API error: {error_text}")
                
                data = await response.json()
                
                if 'candidates' not in data or not data['candidates']:
                    raise Exception("No response from Gemini")
                
                gemini_text = data['candidates'][0]['content']['parts'][0]['text']
                
                import json
                try:
                    clean_text = gemini_text.strip()
                    if clean_text.startswith('```json'):
                        clean_text = clean_text.replace('```json', '').replace('```', '').strip()
                    
                    prompts = json.loads(clean_text)
                    
                    if not isinstance(prompts, list):
                        raise Exception("Gemini did not return a list")
                    
                    print(f"✅ Generated {len(prompts)} enhanced prompts")
                    return prompts[:num_images]
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error: {e}")
                    return [f"{original_prompt} variation {i+1}" for i in range(num_images)]
                
    except Exception as e:
        print(f"💥 Gemini enhancement failed: {e}")
        return [f"{original_prompt} variation {i+1}" for i in range(num_images)]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page"""
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "stripe_publishable_key": STRIPE_PUBLISHABLE_KEY
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_database)):
    """User dashboard - main video creation interface"""
    
    # Get user's recent videos
    recent_videos = db.query(Video).filter(Video.user_id == current_user.id).order_by(Video.created_at.desc()).limit(10).all()
    
    # Check quota
    can_create_video = check_video_quota(current_user)
    quota_info = SUBSCRIPTION_TIERS[current_user.subscription_tier]
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user,
        "recent_videos": recent_videos,
        "can_create_video": can_create_video,
        "videos_remaining": quota_info["videos_per_month"] - current_user.videos_this_month,
        "quota_info": quota_info
    })

@app.post("/api/register")
async def register(user_data: UserCreate, db: Session = Depends(get_database)):
    """User registration"""
    
    # Check if user exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = hash_password(user_data.password)
    
    # Create Stripe customer
    stripe_customer = stripe.Customer.create(
        email=user_data.email,
        name=user_data.full_name
    )
    
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        stripe_customer_id=stripe_customer.id
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": db_user.id})
    
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": db_user.email, "full_name": db_user.full_name}}

@app.post("/api/login")
async def login(user_data: UserLogin, db: Session = Depends(get_database)):
    """User login"""
    
    user = db.query(User).filter(User.email == user_data.email).first()
    
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account deactivated")
    
    access_token = create_access_token(data={"sub": user.id})
    
    return {"access_token": access_token, "token_type": "bearer", "user": {"email": user.email, "full_name": user.full_name}}

@app.post("/api/create-video")
async def create_video_api(request: VideoRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_database)):
    """Create video - protected route"""
    
    # Check quota
    if not check_video_quota(current_user):
        limit = SUBSCRIPTION_TIERS[current_user.subscription_tier]["videos_per_month"]
        raise HTTPException(status_code=403, detail=f"Monthly video limit reached ({limit} videos). Please upgrade your subscription.")
    
    # Validate request
    if not request.audio and not request.generate_music:
        raise HTTPException(status_code=400, detail="Either provide audio or request music generation")
    
    if not request.images and not request.ai_prompt:
        raise HTTPException(status_code=400, detail="Either upload images or provide AI prompt")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Create video record
    db_video = Video(
        user_id=current_user.id,
        job_id=job_id,
        title=request.title,
        status="processing",
        ai_prompt=request.ai_prompt,
        music_genre=request.music_genre,
        enhanced_prompts_used=request.enhance_prompts
    )
    
    db.add(db_video)
    
    # Update user's video count
    current_user.videos_this_month += 1
    
    db.commit()
    
    # Initialize job status
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="starting",
        progress=0,
        message="Starting..."
    )
    
    # Start background task (copy your existing video processing code here)
    asyncio.create_task(process_video_with_music(job_id, request, current_user.id, db))
    
    return {"job_id": job_id, "status": "started"}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get job status - protected route"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.post("/api/create-checkout-session")
async def create_checkout_session(tier: str, current_user: User = Depends(get_current_user)):
    """Create Stripe checkout session"""
    
    if tier != "standard":
        raise HTTPException(status_code=400, detail="Invalid subscription tier")
    
    price = SUBSCRIPTION_TIERS["standard"]["price"]
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': 'AI Video Maker',
                        'description': '2 AI-generated videos per month'
                    },
                    'unit_amount': price,
                    'recurring': {
                        'interval': 'month'
                    }
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{request.base_url}dashboard?success=true",
            cancel_url=f"{request.base_url}dashboard?canceled=true",
            metadata={
                'user_id': current_user.id,
                'tier': 'standard'
            }
        )
        
        return {"checkout_url": checkout_session.url}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_database)):
    """Handle Stripe webhooks"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            user_id = session['metadata']['user_id']
            
            # Update user subscription (always standard tier)
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                user.subscription_tier = "standard"
                user.subscription_expires = datetime.utcnow() + timedelta(days=30)
                user.videos_this_month = 0  # Reset quota
                db.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Video Maker Pro on Railway...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
