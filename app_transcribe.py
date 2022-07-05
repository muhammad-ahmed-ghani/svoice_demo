import uvicorn

if __name__ == "__main__":
    uvicorn.run("transcribe:asgi_app", host="0.0.0.0", port=8000, debug=True, reload=True)