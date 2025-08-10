import requests
import os
import base64
from ollama import Client
from llm.apis.output_schema import LLMOutputSchema
from llm.apis.base import BaseAPI

class OllamaModel(BaseAPI):
    
    def call_model(self, payload: dict={}):
        # create the client
        client = Client(
            host=os.getenv("OLLAMA_HOST", "localhost"),
        )

        response =client.generate(model= self.model_id,
                        prompt= self.user_prompt,
                        system=self.system_prompt,
                        options = {
                            "num_ctx": 20000,
                            "temperature": self.temperature,
                            "num_predict": self.max_output_token,
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                        },
                        **payload
                        )

        return {
            "response": response.response,
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
        }



    def prepare_arguments(self):
        # Prepare the prompt based on user and system prompts
        images = []
        if self.enable_vision:
            for v_image in self.vision_images:
                with open(v_image, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    images.append(encoded_string)
            print(images)
            return {"images": images}

        return {}
        
    def __call__(self) -> LLMOutputSchema:
        try:
            # Prepare the payload
            extra_param = self.prepare_arguments()
            response = self.call_model(extra_param)
            
            # prepare the output schema
            return LLMOutputSchema(
                model_name=self.model_id,
                response=response.get("response", ""),
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                status="success" if response else "failed"
            )

        except Exception as e:
            return LLMOutputSchema(
                model_name=self.model_id,
                response=str(e),
                input_tokens=0,
                output_tokens=0,
                status="error"
            )
        

                
