"""
backend.py
-----------

FastAPI backend for the AI Data Analyst Agent.

Run:
    uvicorn backend:app --reload

Swagger:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.ml_router import router as ml_router

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="AI Data Analyst API",
    version="1.0.0",
    description="Backend API for the AI Data Analyst Agent",
)

# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Health Check
# ==========================================================

@app.get("/")
def root():
    return {
        "message": "AI Data Analyst API Running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api": "running",
    }


# ==========================================================
# Register Routers
# ==========================================================

app.include_router(
    ml_router,
    tags=["Machine Learning"],
)

# ==========================================================
# Startup / Shutdown
# ==========================================================

@app.on_event("startup")
def startup():

    print("=" * 60)
    print("🚀 AI Data Analyst API Started")
    print("📘 Swagger : http://127.0.0.1:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
def shutdown():

    print("🛑 AI Data Analyst API Stopped")


# ==========================================================
# Exception Handlers
# ==========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
        },
    )