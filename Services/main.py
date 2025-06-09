from fastapi import FastAPI
from pydantic import BaseModel
from .model import process_data

# Keep note of the above import


from .config import API_PREFIX

class DataRequest(BaseModel):
    data: str    # e.g. HTML or any payload

class DataResponse(BaseModel):
    result: str

app = FastAPI(title="Python Processing API")

@app.post(f"{API_PREFIX}/process", response_model=DataResponse)
def process_endpoint(req: DataRequest):
    output = process_data(req.data)
    return DataResponse(result=output)
