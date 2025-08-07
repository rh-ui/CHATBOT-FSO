from typing import Optional
from pydantic import BaseModel



class Query(BaseModel):
    question: str
    lang: str 
    k: int = 3
    score_threshold: float = 0.01
    use_llm: bool = True  
    context: Optional[dict] = None
