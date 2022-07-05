from distutils.log import debug
import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:asgi_app", host="0.0.0.0", port=5000, debug=True, reload=True)