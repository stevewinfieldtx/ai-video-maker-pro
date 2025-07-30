from __future__ import annotations

# main.py - Complete AI Video Maker with PayPal Integration (NO STRIPE!)
from fastapi import FastAPI, HTTPException, Request, Depends, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
import json

# MoviePy imports - FIXED for version 2.0
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.VideoClip import ImageClip
    MOVIEPY_AVAILABLE = True
    print("✅ MoviePy available")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"⚠️ MoviePy not available - video processing disabled: {e}")

# Runware imports - FIXED for version 0.4.16
try:
    from runware import Runware
    from runware.types import IImageInference
    RUNWARE_AVAILABLE = True
    print("✅ Runware available")
except ImportError as e:
    RUNWARE_AVAILABLE = False
    print(f"⚠️ Runware not available - AI image generation disabled: {e}")

# Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini available")
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"⚠️ Gemini not available - prompt enhancement disabled: {e}")

# Audio processing imports
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print("✅ Librosa available")
except ImportError as e:
    LIBROSA_AVAILABLE = False
    print(f"⚠️ Librosa not available - beat sync disabled: {e}")

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

# Create directories first
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")

for dir_path in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Mount static files conditionally
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files mounted")
except Exception as e:
    print(f"⚠️ Could not mount static files: {e}")

# Environment Variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./video_maker.db")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_ENVIRONMENT = os.getenv("PAYPAL_ENVIRONMENT", "sandbox")  # sandbox or live
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY", "H2J4fDIX9bPHcr2jItoFJkS3yMqbgM1a")
MUSICAPI_AI_KEY = os.getenv("MUSICAPI_AI_KEY", "f8e3be01ef37a0b3ceff3ea0a389d4da")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBX4Q3dZXzdHIoHDSHX6i5Qhp8jS01wUds")

# PayPal Configuration
PAYPAL_BASE_URL = "https://api.sandbox.paypal.com" if PAYPAL_ENVIRONMENT == "sandbox" else "https://api.paypal.com"

print(f"💰 PayPal Environment: {PAYPAL_ENVIRONMENT}")
print(f"💰 PayPal Client ID: {PAYPAL_CLIENT_ID[:12] if PAYPAL_CLIENT_ID else 'Not set'}...")

if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyBX4Q3dZXzdHIoHDSHX6i5Qhp8jS01wUds":
    genai.configure(api_key=GEMINI_API_KEY)

# Database Setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)
    video_credits = Column(Integer, default=0)  # NO FREE CREDITS!
    paypal_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Create tables
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")
except Exception as e:
    print(f"💥 Error creating database tables: {str(e)}")

# Pydantic Models
class VideoRequest(BaseModel):
    ai_prompt: str = ""
    images: List[str] = []
    music_genre: str = "electronic"
    music_mood: str = "energetic"
    music_bpm: str = "medium"
    image_style: str = "Studio Fashion Photography"
    seconds_per_image: float = 4.0
    enhance_prompts: bool = True
    uploaded_audio: bool = False
    
    @property
    def num_images(self) -> int:
        if self.images:
            return len(self.images)
        else:
            return max(1, int(60 / self.seconds_per_image))

class VideoJob:
    def __init__(self, id: str, user_id: int, status: str = "pending", message: str = "", progress: int = 0):
        self.id = id
        self.user_id = user_id
        self.status = status
        self.message = message
        self.progress = progress
        self.error = None
        self.video_url = None
        self.created_at = datetime.utcnow()

# Job storage
jobs = {}

# PayPal Functions
async def get_paypal_access_token():
    """Get PayPal access token for API calls"""
    try:
        auth = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode()).decode()
        
        headers = {
            "Accept": "application/json",
            "Accept-Language": "en_US",
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = "grant_type=client_credentials"

        
        print(f"📁 Saved original audio: {original_audio_path}")
        
        # Load and check duration
        if MOVIEPY_AVAILABLE:
            audio_clip = AudioFileClip(str(original_audio_path))
            original_duration = audio_clip.duration
            
            print(f"🎵 Original audio duration: {original_duration:.1f} seconds")
            
            # If longer than 60 seconds, trim it
            if original_duration > 60:
                print(f"✂️ Trimming audio to 60 seconds...")
                trimmed_audio_clip = audio_clip.subclipped(0, 60)
                trimmed_path = job_dir / "uploaded_audio.wav"
                trimmed_audio_clip.write_audiofile(
                    str(trimmed_path),
                    verbose=False,
                    logger=None
                )
                trimmed_audio_clip.close()
                audio_clip.close()
                
                print(f"✅ Audio trimmed and saved: {trimmed_path}")
                return str(trimmed_path)
            else:
                # Audio is already 60 seconds or less
                final_path = job_dir / "uploaded_audio.wav"
                audio_clip.write_audiofile(
                    str(final_path),
                    verbose=False,
                    logger=None
                )
                audio_clip.close()
                
                print(f"✅ Audio duration OK, saved as: {final_path}")
                return str(final_path)
        else:
            # Fallback if MoviePy not available
            final_path = job_dir / "uploaded_audio.wav"
            shutil.copy(original_audio_path, final_path)
            print(f"⚠️ MoviePy not available, using original audio: {final_path}")
            return str(final_path)
            
    except Exception as e:
        print(f"💥 Error processing uploaded audio: {e}")
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

# Authentication Functions
def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

def create_password_reset_token(email: str):
    to_encode = {"email": email, "purpose": "password_reset"}
    expire = datetime.utcnow() + timedelta(hours=1)  # Reset token valid for 1 hour
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm="HS256")
    return encoded_jwt

def verify_password_reset_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if payload.get("purpose") != "password_reset":
            return None
        return payload.get("email")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return {
                "id": user.id,
                "email": user.email,
                "video_credits": user.video_credits,
                "paypal_customer_id": user.paypal_customer_id
            }
        finally:
            db.close()
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Music Generation Function
async def generate_music_with_musicapi(prompt: str, genre: str, mood: str, duration: int = 60) -> str:
    """Generate music using MusicAPI.ai"""
    try:
        print(f"🎼 Generating music: {prompt}")
        
        payload = {
            "type": "bgm",
            "description": f"{prompt} instrumental music",
            "genre": [genre],
            "mood": [mood],
            "duration": duration
        }
        
        print(f"🚀 Sending to MusicAPI.ai: {payload}")
        
        async with aiohttp.ClientSession() as session:
            # Start generation
            async with session.post(
                "https://musicapi.ai/api/music",
                headers={"Authorization": f"Bearer {MUSICAPI_AI_KEY}"},
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"MusicAPI request failed: {response.status}")
                
                result = await response.json()
                print(f"✅ MusicAPI.ai response: {result}")
                
                if result.get("message") == "success":
                    task_id = result.get("task_id")
                    print(f"🎵 Got task_id: {task_id}, polling for result...")
                    
                    # Poll for result
                    max_attempts = 24  # 2 minutes max
                    for attempt in range(max_attempts):
                        await asyncio.sleep(5)
                        
                        async with session.get(
                            f"https://musicapi.ai/api/music/{task_id}",
                            headers={"Authorization": f"Bearer {MUSICAPI_AI_KEY}"}
                        ) as status_response:
                            if status_response.status == 200:
                                status_result = await status_response.json()
                                print(f"📊 Status {attempt + 1}: {status_result.get('status')}")
                                
                                if status_result.get("status") == "succeeded":
                                    music_url = status_result.get("data", {}).get("music_url")
                                    if music_url:
                                        print(f"🎉 Music generated: {music_url}")
                                        return music_url
                                    else:
                                        raise Exception("No music URL in response")
                                elif status_result.get("status") == "failed":
                                    raise Exception("Music generation failed")
                    
                    raise Exception("Music generation timeout")
                else:
                    raise Exception(f"MusicAPI error: {result}")
                    
    except Exception as e:
        print(f"💥 Music generation failed: {e}")
        raise Exception(f"Music generation failed: {str(e)}")

# Gemini Enhancement Functions
async def enhance_prompts_with_gemini(original_prompt: str, music_genre: str, music_mood: str, 
                                    seconds_per_image: float, music_bpm: str, image_style: str, 
                                    num_images: int) -> List[str]:
    """Use Gemini to enhance and create varied prompts"""
    try:
        if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
            return [f"{image_style} style: {original_prompt}"] * num_images
        
        # Detect gender consistency
        gender_keywords = {
            'female': ['woman', 'girl', 'female', 'lady', 'she', 'her'],
            'male': ['man', 'boy', 'male', 'guy', 'he', 'him', 'gentleman']
        }
        
        detected_gender = None
        prompt_lower = original_prompt.lower()
        for gender, keywords in gender_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_gender = gender
                break
        
        gender_instruction = ""
        if detected_gender:
            gender_instruction = f"CRITICAL: ALL prompts must feature ONLY {detected_gender} subjects. Maintain consistent gender throughout ALL {num_images} prompts."
        else:
            gender_instruction = "CRITICAL: Choose ONE gender and use consistently throughout ALL prompts."
        
        context = f"""
Original user prompt: "{original_prompt}"
Video style: {image_style}
Music: {music_genre} {music_mood} at {music_bpm} speed
Image duration: {seconds_per_image} seconds each
Number of images needed: {num_images}

{gender_instruction}

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

PROMPT STRUCTURE (125 tokens each):
- Subject description (25 tokens)
- Action/pose details (20 tokens) 
- Setting/environment (30 tokens)
- Lighting/atmosphere (25 tokens)
- Style/technical details (25 tokens)

Return ONLY a JSON array of prompts, no other text:
["prompt 1", "prompt 2", "prompt 3", ...]
"""
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            context,
            generation_config={
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "max_output_tokens": 3000,
            }
        )
        
        gemini_text = response.text
        print(f"🧠 Gemini response: {gemini_text[:200]}...")
        
        # Parse JSON response
        try:
            clean_text = gemini_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text.replace('```json', '').replace('```', '').strip()
            
            enhanced_prompts = json.loads(clean_text)
            
            if isinstance(enhanced_prompts, list) and len(enhanced_prompts) >= num_images:
                result = enhanced_prompts[:num_images]
                print(f"✅ Generated {len(result)} enhanced prompts")
                return result
            else:
                raise Exception("Invalid prompt format from Gemini")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            # Fallback to original prompt
            return [f"{image_style} style: {original_prompt}"] * num_images
            
    except Exception as e:
        print(f"💥 Gemini enhancement failed: {e}")
        # Fallback to original prompt
        return [f"{image_style} style: {original_prompt}"] * num_images

async def quality_check_with_gemini(prompts: List[str]) -> List[str]:
    """Use Gemini to quality check prompts for SFW content"""
    import json
    try:
        if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
            return prompts
        
        context = f"""
Check these {len(prompts)} image prompts for quality and safety:

{json.dumps(prompts, indent=2)}

Requirements:
1. All prompts must be Safe For Work (SFW)
2. No inappropriate content
3. Good anatomy descriptions (no misshapen bodies)
4. Professional quality descriptions

Return ONLY the approved prompts as a JSON array. Remove any problematic prompts.
If all prompts are good, return them all.

Format: ["prompt1", "prompt2", ...]
"""
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(context)
        
        gemini_text = response.text.strip()
        if gemini_text.startswith('```json'):
            gemini_text = gemini_text.replace('```json', '').replace('```', '').strip()
        
        filtered_prompts = json.loads(gemini_text)
        
        if isinstance(filtered_prompts, list):
            print(f"✅ Quality check passed: {len(filtered_prompts)} prompts approved")
            return filtered_prompts
        else:
            return prompts
            
    except Exception as e:
        print(f"💥 Quality check failed: {e}, using original prompts")
        return prompts

# Video Processing Functions
def create_static_clip(image_path: str, duration: float, size=(1152, 1536)) -> 'ImageClip':
    """Create a static video clip from an image"""
    if MOVIEPY_AVAILABLE:
        clip = ImageClip(image_path).set_duration(duration)
        if size:
            clip = clip.resize(size)
        return clip
    else:
        raise Exception("MoviePy not available for video processing")

async def continue_video_processing(job_id: str, request: VideoRequest, audio_path: str):
    """Continue with video processing using the provided audio path"""
    try:
        job = jobs[job_id]
        job_dir = UPLOAD_DIR / job_id
        
        print(f"🎬 Continuing video processing for job {job_id}")
        print(f"🎵 Using audio: {audio_path}")
        
        # Get images (uploaded or AI generated)
        if request.images:
            print(f"📁 Processing {len(request.images)} uploaded images")
            job.message = "Processing uploaded images..."
            job.progress = 30
            
            image_paths = []
            for i, img_b64 in enumerate(request.images):
                img_data = base64.b64decode(img_b64)
                img_path = job_dir / f"image_{i:03d}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                image_paths.append(str(img_path))
        else:
            # AI image generation
            print(f"🤖 Generating AI images")
            job.message = "Generating AI images..."
            job.progress = 30
            
            if not RUNWARE_AVAILABLE:
                raise Exception("AI generation not available")
            
            client = Runware(api_key=RUNWARE_API_KEY)
            await client.connect()
            
            # Generate prompts with Gemini enhancement
            prompts_to_use = []
            if request.enhance_prompts and GEMINI_AVAILABLE:
                print(f"🧠 Using Gemini 2.0-Flash for prompt enhancement...")
                job.message = "Enhancing prompts with AI..."
                job.progress = 35
                
                # Request 10% more images for quality filtering
                extra_images = max(1, min(2, int(request.num_images * 0.1)))
                total_needed = min(10, request.num_images + extra_images)
                
                enhanced_prompts = await enhance_prompts_with_gemini(
                    original_prompt=request.ai_prompt,
                    music_genre=request.music_genre,
                    music_mood=request.music_mood,
                    seconds_per_image=request.seconds_per_image,
                    music_bpm=request.music_bpm,
                    image_style=request.image_style,
                    num_images=total_needed
                )
                
                # Quality check
                job.message = "Quality checking prompts..."
                job.progress = 40
                filtered_prompts = await quality_check_with_gemini(enhanced_prompts)
                
                # Use the best prompts
                prompts_to_use = filtered_prompts[:request.num_images]
                print(f"✅ Using {len(prompts_to_use)} enhanced prompts")
            else:
                # Use original prompt with style
                style_prompt = f"{request.image_style} style: {request.ai_prompt}"
                prompts_to_use = [style_prompt] * request.num_images
            
            # Generate images in batches of 5
            image_paths = []
            batch_size = 5
            
            for batch_start in range(0, len(prompts_to_use), batch_size):
                batch_prompts = prompts_to_use[batch_start:batch_start + batch_size]
                batch_tasks = []
                
                for i, prompt in enumerate(batch_prompts):
                    global_index = batch_start + i
                    print(f"🎨 Generating image {global_index + 1}/{len(prompts_to_use)}: {prompt[:50]}...")
                    
                    payload = IImageInference(
                        model="runware:97@3",
                        positivePrompt=prompt,
                        width=1152,
                        height=1536,
                        steps=24,
                        scheduler="DPM++ 2M Karras",
                        numberResults=1
                    )
                    
                    batch_tasks.append(client.imageInference(requestImage=payload))
                
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(batch_results):
                    global_index = batch_start + i
                    if isinstance(result, Exception):
                        print(f"❌ Image {global_index + 1} failed: {result}")
                        continue
                    
                    if result and len(result) > 0:
                        # Save image
                        image_data = base64.b64decode(result[0].imageBase64)
                        img_path = job_dir / f"ai_image_{global_index:03d}.jpg"
                        with open(img_path, "wb") as f:
                            f.write(image_data)
                        image_paths.append(str(img_path))
                        print(f"✅ Saved image {global_index + 1}: {img_path}")
                    else:
                        print(f"❌ No image data for image {global_index + 1}")
                
                # Small delay between batches
                if batch_start + batch_size < len(prompts_to_use):
                    await asyncio.sleep(2)
                
                # Update progress
                progress = 40 + (30 * (batch_start + batch_size) / len(prompts_to_use))
                job.progress = min(70, int(progress))
                job.message = f"Generated {len(image_paths)}/{request.num_images} images..."
        
        # Deduct credit from user
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == job.user_id).first()
            if user and user.video_credits > 0:
                user.video_credits -= 1
                db.commit()
                print(f"💳 Credit deducted. User {user.email} now has {user.video_credits} credits")
        finally:
            db.close()
        
        # Create video
        job.message = "Creating video with audio..."
        job.progress = 75
        
        if not image_paths:
            raise Exception("No images available for video creation")
        
        print(f"📸 Creating video from {len(image_paths)} images")
        
        # Create video clips
        video_clips = []
        for img_path in image_paths:
            clip = create_static_clip(img_path, request.seconds_per_image)
            video_clips.append(clip)
        
        # Combine clips
        if MOVIEPY_AVAILABLE:
            final_video = ImageSequenceClip([clip.get_frame(0) for clip in video_clips], 
                                          fps=1/request.seconds_per_image)
            
            # Add audio
            audio_clip = AudioFileClip(audio_path)
            final_video = final_video.set_audio(audio_clip)
            
            # Export video
            output_path = OUTPUT_DIR / f"video_{job_id}.mp4"
            job.message = "Rendering final video..."
            job.progress = 85
            
            final_video.write_videofile(
                str(output_path),
                fps=24,
                audio_codec='aac',
                codec='libx264',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            final_video.close()
            audio_clip.close()
            for clip in video_clips:
                clip.close()
            
            job.status = "completed"
            job.video_url = f"/output/video_{job_id}.mp4"
            job.message = "Video created successfully!"
            job.progress = 100
            
            print(f"✅ Video completed: {output_path}")
        else:
            raise Exception("MoviePy not available for video creation")
        
    except Exception as e:
        print(f"💥 Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].message = "Video processing failed"

# API Routes
@app.get("/@vite/client")
async def vite_client():
    return Response(content="", media_type="application/javascript")

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "paypal_client_id": PAYPAL_CLIENT_ID
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user,
        "paypal_client_id": PAYPAL_CLIENT_ID
    })

@app.post("/api/register")
async def register(request: Request):
    print("Received registration request")
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
        print(f"Processing registration for: {email}")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        print(f"🔧 Registration attempt for: {email}")
        
        # Check if user exists
        db = SessionLocal()
        try:
            existing_user = db.query(User).filter(User.email == email).first()
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user - SUPER SIMPLE, NO EXTERNAL CALLS
            hashed_password = get_password_hash(password)
            user = User(
                email=email,
                name=data.get("full_name"),
                hashed_password=hashed_password,
                video_credits=0
            )
            print("🔍 About to add user to session")
            try:
                db.add(user)
                print("🔍 Added user to session, about to commit")
                db.commit()
                print("🔍 Committed, about to refresh")
                db.refresh(user)
                print("🔍 Refreshed user")
            except IntegrityError as ie:
                db.rollback()
                print(f"💥 Integrity error during user creation: {str(ie)}")
                raise HTTPException(status_code=400, detail="Email already registered")
            
            print(f"✅ User created successfully: {user.id}")
            
            # Create access token
            access_token = create_access_token({"user_id": user.id})
            
            response = JSONResponse({
                "message": "Registration successful!", 
                "credits": user.video_credits,
                "access_token": access_token
            })
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                max_age=86400,
                samesite="lax"
            )
            return response
            
        except Exception as db_error:
            print(f"💥 Database error: type={type(db_error).__name__}, message={str(db_error)}, repr={repr(db_error)}")
            import traceback
            traceback.print_exc()
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        finally:
            db.close()
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"💥 Registration error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user or not verify_password(password, user.hashed_password):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            access_token = create_access_token({"user_id": user.id})
            
            response = JSONResponse({
                "message": "Login successful",
                "access_token": access_token
            })
            response.set_cookie(
                key="access_token",
                value=access_token,
                httponly=True,
                max_age=86400,
                samesite="lax"
            )
            return response
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/logout")
async def logout():
    response = JSONResponse({"message": "Logged out successfully"})
    response.delete_cookie("access_token")
    return response

@app.post("/api/forgot-password")
async def forgot_password(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check if user exists
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                # Don't reveal that the user doesn't exist for security reasons
                return {"message": "If your email is registered, you will receive a password reset link"}
            
            # Create password reset token
            reset_token = create_password_reset_token(email)
            
            # In a real application, you would send an email with the reset link
            # For this demo, we'll just return the token
            reset_link = f"http://localhost:8000/?token={reset_token}"
            
            print(f"Password reset link for {email}: {reset_link}")
            
            # In a real application, you would use an email service like SendGrid, Mailgun, etc.
            # Example code for sending email (commented out):
            # await send_email(
            #     to_email=email,
            #     subject="Password Reset Request",
            #     content=f"Click the link below to reset your password:\n{reset_link}"
            # )
            
            return {"message": "If your email is registered, you will receive a password reset link"}
        finally:
            db.close()
    except Exception as e:
        print(f"Password reset request error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/reset-password")
async def reset_password(request: Request):
    try:
        data = await request.json()
        token = data.get("token")
        new_password = data.get("new_password")
        
        if not token or not new_password:
            raise HTTPException(status_code=400, detail="Token and new password are required")
        
        # Verify token and get email
        email = verify_password_reset_token(token)
        if not email:
            raise HTTPException(status_code=400, detail="Invalid or expired token")
        
        # Update user password
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Hash and update password
            hashed_password = get_password_hash(new_password)
            user.hashed_password = hashed_password
            db.commit()
            
            return {"message": "Password has been reset successfully"}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        print(f"Password reset error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/create_video")
async def create_video(request: Request, current_user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        num_images = data.get("num_images", 10)
        seconds_per_image = data.get("seconds_per_image", 3)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == current_user["user_id"]).first()
            if not user or user.video_credits <= 0:
                raise HTTPException(status_code=402, detail="Insufficient video credits")

            # Create job
            job_id = str(uuid.uuid4())
            job = VideoJob(
                id=job_id,
                user_id=user.id,
                status="pending",
                progress=0,
                message="Initializing...",
                created_at=datetime.now()
            )
            jobs[job_id] = job

            # Start processing asynchronously
            asyncio.create_task(process_video_job(job_id, prompt, num_images, seconds_per_image))

            return {"job_id": job_id, "message": "Video creation started"}
        finally:
            db.close()
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Video creation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    response.delete_cookie("access_token")
    return response

@app.post("/api/create-video")
async def create_video(
    request: Request,
    video_request: str = Form(...),
    audio_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """Create a video with either AI-generated or uploaded audio"""
    try:
        # Parse the JSON request data
        request_data = json.loads(video_request)
        video_req = VideoRequest(**request_data)
        
        print(f"🚀 Creating video for user {current_user['email']}")
        print(f"📊 Request: AI prompt='{video_req.ai_prompt}', Uploaded audio={video_req.uploaded_audio}")
        
        # Check if user has enough credits
        if current_user['video_credits'] <= 0:
            raise HTTPException(
                status_code=402, 
                detail="No video credits remaining. Please purchase more credits for just $1!"
            )
        
        # Validate audio file if uploaded
        if video_req.uploaded_audio:
            if not audio_file:
                raise HTTPException(status_code=400, detail="Audio file required when uploaded_audio is True")
            
            # Check file size (limit to 50MB)
            if audio_file.size > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Audio file too large. Maximum size is 50MB.")
            
            # Check file type
            allowed_types = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/x-m4a']
            if audio_file.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: MP3, WAV, M4A, OGG")
            
            print(f"📁 Uploaded audio: {audio_file.filename} ({audio_file.content_type}, {audio_file.size} bytes)")
        
        # Create job
        job_id = str(uuid.uuid4())[:8]
        job_dir = UPLOAD_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Initialize job
        jobs[job_id] = VideoJob(
            id=job_id,
            user_id=current_user['id'],
            status="processing",
            message="Initializing video creation...",
            progress=0
        )
        
        # Process audio (either upload or generate)
        try:
            if video_req.uploaded_audio and audio_file:
                # Process uploaded audio
                audio_path = await process_uploaded_audio(audio_file, job_dir)
                jobs[job_id].message = "Uploaded audio processed (trimmed to 60s)..."
                jobs[job_id].progress = 15
            else:
                # Generate AI music
                jobs[job_id].message = "Generating AI music..."
                jobs[job_id].progress = 10
                
                music_prompt = f"{video_req.music_bpm} {video_req.music_mood} {video_req.music_genre}"
                
                music_url = await generate_music_with_musicapi(
                    music_prompt, 
                    video_req.music_genre, 
                    video_req.music_mood, 
                    60
                )
                
                # Download generated music
                audio_path = job_dir / "generated_music.wav"
                async with aiohttp.ClientSession() as session:
                    async with session.get(music_url) as response:
                        if response.status == 200:
                            with open(audio_path, "wb") as f:
                                f.write(await response.read())
                        else:
                            raise Exception(f"Failed to download music: {response.status}")
                
                audio_path = str(audio_path)
                jobs[job_id].message = "AI music generated..."
                jobs[job_id].progress = 15
        
        except Exception as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = f"Audio processing failed: {str(e)}"
            jobs[job_id].message = "Failed to process audio"
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
        
        # Continue with video processing in background
        asyncio.create_task(continue_video_processing(job_id, video_req, audio_path))
        
        return {"job_id": job_id, "status": "processing", "message": "Video creation started"}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in video_request")
    except Exception as e:
        print(f"💥 Video creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video creation failed: {str(e)}")

@app.get("/output/{filename}")
async def serve_video(filename: str, current_user: dict = Depends(get_current_user)):
    """Serve generated video files"""
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/api/user/credits")
async def get_user_credits(current_user: dict = Depends(get_current_user)):
    """Get current user's video credits"""
    return {"video_credits": current_user['video_credits']}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "moviepy": MOVIEPY_AVAILABLE,
        "runware": RUNWARE_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "librosa": LIBROSA_AVAILABLE,
        "paypal_configured": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
             