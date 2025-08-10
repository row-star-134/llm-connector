from pydantic import BaseModel

class BaseAPI(BaseModel):
   
    model_id : str
    extra_params : dict = {}
    user_prompt: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_output_token: int = 1000
    top_p: float = 1.0
    top_k: int = 50
    enable_vision: bool = False
    vision_images: list = []
    
    def call_model(self, payload: dict):
        
        raise NotImplementedError("This method should be implemented by subclasses.")
   
    
    def prepare_prompt(self):
        
        raise NotImplementedError("This method should be implemented by subclasses.")
   
   
    