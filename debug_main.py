# debug_main.py - Minimal version to test imports
import sys
print("🔍 Starting import test...")

try:
    print("1. Testing FastAPI...")
    from fastapi import FastAPI
    print("✅ FastAPI imported")
    
    print("2. Testing basic imports...")
    import os, uuid, asyncio
    from pathlib import Path
    print("✅ Basic imports OK")
    
    print("3. Testing Pydantic...")
    from pydantic import BaseModel
    print("✅ Pydantic OK")
    
    print("4. Testing SQLAlchemy...")
    import sqlalchemy
    from sqlalchemy import create_engine
    print("✅ SQLAlchemy OK")
    
    print("5. Testing Stripe...")
    import stripe
    print("✅ Stripe OK")
    
    print("6. Testing JWT...")
    import jwt
    print("✅ JWT OK")
    
    print("7. Testing bcrypt...")
    import bcrypt
    print("✅ bcrypt OK")
    
    print("8. Testing aiohttp...")
    import aiohttp
    print("✅ aiohttp OK")
    
    print("9. Testing MoviePy...")
    from moviepy import ImageSequenceClip
    print("✅ MoviePy OK")
    
    print("10. Testing Runware...")
    try:
        from runware import Runware
        print("✅ Runware main import OK")
        try:
            from runware.types import IImageInference
            print("✅ Runware types import OK")
        except ImportError as e:
            print(f"❌ Runware types failed: {e}")
            # Try alternative import
            try:
                from runware import IImageInference
                print("✅ Runware direct import OK")
            except ImportError as e2:
                print(f"❌ Runware direct import failed: {e2}")
    except ImportError as e:
        print(f"❌ Runware main import failed: {e}")
    
    print("11. Testing Google Generative AI...")
    try:
        import google.generativeai as genai
        print("✅ Gemini OK")
    except ImportError as e:
        print(f"❌ Gemini failed: {e}")
    
    print("12. Testing Librosa...")
    try:
        import librosa
        print("✅ Librosa OK")
    except ImportError as e:
        print(f"❌ Librosa failed: {e}")
    
    print("\n🎉 Creating minimal FastAPI app...")
    app = FastAPI(title="Debug Test")
    
    @app.get("/")
    def read_root():
        return {"message": "Debug test successful!"}
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"💥 Import failed at step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
