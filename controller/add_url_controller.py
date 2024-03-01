from pydantic import HttpUrl
from typing import List
from fastapi.responses import JSONResponse

from utils import load_urls, split_document, index_documents


async def add_url_ctr(embedding, persist_directory: str, urls: List[str]):
    # Load documents from url
    data = await load_urls(urls=urls)

    # Split documents into chunks
    chunks = await split_document(documents=data)

    # index chunks
    await index_documents(chunks=chunks, embedding=embedding, persist_directory=persist_directory)

    return JSONResponse(
        content='Urls loaded successfully.'
    )