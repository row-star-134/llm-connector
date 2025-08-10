from pydantic import BaseModel
from typing import Literal
from llm.apis.ollama_model import OllamaModel

class LLMController(BaseModel):
    model_id: str
    llm_service : Literal["ollama"] = "ollama"
    user_prompt: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_output_token: int = 1000
    top_p: float = 1.0
    top_k: int = 50
    enable_vision: bool = False
    vision_images: list = []
    extra_params: dict = {}
    
    
    
    def forward_call(self):
        response_object = None
        
    
        if self.llm_service == "ollama":
            ollama_call = OllamaModel(
                model_id=self.model_id,
                user_prompt=self.user_prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature,
                max_output_token=self.max_output_token,
                top_p=self.top_p,
                top_k=self.top_k,
                extra_params=self.extra_params,
                enable_vision= self.enable_vision,
                vision_images= self.vision_images
            )
            response_object = ollama_call() 
                   
        
        return response_object
    
    
    def __call__(self):
        print("Calling LLMController with model_id:", self.model_id)
        return self.forward_call()