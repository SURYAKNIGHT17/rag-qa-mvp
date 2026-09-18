import uvicorn

if __name__ == "__main__":
    print("========================================================================")
    print("[RAG SYSTEM] API Server started successfully!")
    print("> Open in your browser: http://127.0.0.1:8000/docs")
    print("========================================================================")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

