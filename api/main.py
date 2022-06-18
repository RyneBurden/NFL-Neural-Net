from fastapi import FastAPI

app = FastAPI()

# auth0 with fastAPI:
#   https://auth0.com/blog/build-and-secure-fastapi-server-with-auth0/

@app.get("/")
def home():
    return "Hey"