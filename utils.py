from pydantic import HttpUrl
from typing import List
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain


async def load_urls(urls: List[str]):
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()


async def split_document(documents, chunk_size: int = 1000, seperators: List[str] = ['\n\n', '\n', '.', ','], chunk_overlap: int = 0, is_recurssive: bool = True):
    if is_recurssive:
        splitter = RecursiveCharacterTextSplitter(
        separators=seperators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


async def index_documents(chunks, embedding, persist_directory):
    # Index data and save to db
    vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_directory
        )
    
    vectorstore.persist()


async def load_index(embedding, persist_directory):
    if os.path.exists(persist_directory):
        # Load the persistant stora  
        vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
            ) 
        return vectorstore
    return None


async def get_answer(query, vectorstore, llm, include_sources: bool = True):
    answer = None
    if vectorstore:
        # Init chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                
        # Get results
        result = chain({'question': query.query}, return_only_outputs=True)
        
        # Check if sources should be included
        if query.include_sources:
            answer = {'answer': result.get('answer'), 
                    'sources': result.get('sources').split('\n')}
        else:
            answer = {'answer': result.get('answer')}
    
    return answer