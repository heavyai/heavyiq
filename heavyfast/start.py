if __name__ == "__main__":
    import uvicorn

    uvicorn.run("heavyiq.main:create_app", factory=True, host="0.0.0.0", port=8000, reload=True)
