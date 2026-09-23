"""
AI Support System - Main Application
"""

import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AI Healthcare IT Support", version="1.0.0")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-support"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
