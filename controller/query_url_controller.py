from pydantic import HttpUrl
from typing import List
from utils import load_index, get_answer
from fastapi.responses import JSONResponse

async def query_urls_ctr(llm, embedding, persist_directory, query):
    # Load index
    vectorstore = await load_index(persist_directory=persist_directory, embedding=embedding)

    # Get answers
    answer = await get_answer(query=query, llm=llm, vectorstore=vectorstore)
    
    return JSONResponse(
        content=answer
    )