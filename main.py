# Import dependencies
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import HttpUrl
from typing import List
import uvicorn

from models import QueryURL
from controller.add_url_controller import add_url_ctr
from controller.query_url_controller import query_urls_ctr

# Load .env variables
load_dotenv()

# Init llm
llm = llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)

# Persistent storage path
persist_directory = './chroma_db'

# Init embedding
embedding = OpenAIEmbeddings()

# Set the api version
version = "0.1.0"

# Init fastapi
app = FastAPI(title="Research AI API", version=version)


@app.post('/add_research_urls')
async def add_research_urls(urls: List[str]):
    return await add_url_ctr(urls=urls, persist_directory=persist_directory, embedding=embedding)


@app.post('/query')
async def query_urls(query: QueryURL):
    return await query_urls_ctr(query=query, persist_directory=persist_directory, llm=llm, embedding=embedding)


if __name__ == "__main__":
    uvicorn.run(app='main:app', port=8001, reload=True)

