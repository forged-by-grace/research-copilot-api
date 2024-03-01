from pydantic import BaseModel, Field

class QueryURL(BaseModel):
    query: str = Field(description="Client query to be responded to by the LLM.")
    include_sources: bool = Field(default=True, description="Checks if the client wants the source returned or not.")