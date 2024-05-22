from typing import Union

from fastapi import FastAPI, Request
from stringmatcher import checksimilarity
from fastapi.middleware.cors import CORSMiddleware
from fastapi import  File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from stringmatcher import preprocess_data
import pandas as pd
import json
app = FastAPI()
obj=checksimilarity()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This should be limited to only the domains you want to allow
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")
templates = Jinja2Templates(directory=".")

@app.get("/")
async def read_item(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/")
# def read_root():
#     return {"_":"String matching server"}


@app.get("/strcmp/{n}/{str1}_{str2}")
def read_item(n,str1,str2):
    temp=pd.DataFrame([[str1,str2]])
    print(str1,str2,"***********")
    temp.columns=["key1","key2"]
    res=preprocess_data(temp)
    print(res,"___________________")
    sim=False
    if(res[0]>=float(n)):
        sim=True
        return {"similar":sim, "Confidence":str(res[0])}  
    return {"similar":False, "Confidence":str(res[0])}

        
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # contents = await file.read()
    print("File recieved")
    # Do something with the file contents, such as saving it to disk or processing it
    return {"filename": file.filename}
