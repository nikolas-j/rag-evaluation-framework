"""Run the FastAPI development server.

Usage:
    python -m api.server
    
Or with uvicorn directly:
    uvicorn api.main:app --reload --port 8000
"""

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
