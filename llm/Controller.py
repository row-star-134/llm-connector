from pydantic import BaseModel
from typing import Literal
from llm.apis.ollama_model import OllamaModel
from llm.apis.gemini_models import GeminiModel
from llm.apis.output_schema import LLMOutputSchema

class LLMController(BaseModel):
    model_id: str
    llm_service : Literal["ollama",'gemini'] = "ollama"
    user_prompt: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_output_token: int = 1000
    top_p: float = 1.0
    top_k: int = 50
    enable_vision: bool = False
    vision_images: list = []
    extra_params: dict = {}
    
    
    
    def forward_call(self) -> LLMOutputSchema:
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
        
        elif self.llm_service == "gemini": 
            gemini_call = GeminiModel(
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
            response_object = gemini_call()
              
        if response_object:
            return response_object
    
        return LLMOutputSchema(
            model_name=self.model_id,
            response = "No response from model",
            input_tokens=0,
            output_tokens=0,
            status="error"
            
        )
    
    
    def __call__(self) -> LLMOutputSchema:

        return self.forward_call()