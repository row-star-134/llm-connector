from pydantic import BaseModel
class LLMOutputSchema(BaseModel):
    model_name: str 
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0 
    status: str = ""
    
    
    