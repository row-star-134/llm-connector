from typing import Any
from google import genai
from google.genai import types
from llm.apis.base import BaseAPI
from llm.apis.output_schema import LLMOutputSchema
import os
import dotenv

dotenv.load_dotenv()

class GeminiModel(BaseAPI):
    def call_model(self, payload: dict = {}):
        
        client = genai.Client()
        
        contents = [self.user_prompt] + payload['images'] if self.enable_vision else [self.user_prompt]
        response = client.models.generate_content(
            model=self.model_id,
            contents = contents,
            config= types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction= self.system_prompt,
                max_output_tokens=self.max_output_token,
                top_p=self.top_p,
                top_k=self.top_k,
                **self.extra_params 
            )
        )

        return {
            "response": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count, # type: ignore
            "output_tokens": response.usage_metadata.candidates_token_count, # type: ignore
            "status": "success"
        }
    
    def prepare_arguments(self):
        if self.enable_vision:
            images = []
            for image_i, v_image in enumerate(self.vision_images):
                
                with open(v_image, "rb") as image_file:
                    encoded_string = image_file.read()
                    images.append("image " + str(image_i+1)+ " :")
                    images.append(types.Part.from_bytes(
                        data= encoded_string,
                        mime_type="image/jpeg"
                    ))

            return {"images": images}
        return {}

    def __call__(self) -> LLMOutputSchema:
        try:
            # Prepare the payload
            extra_param = self.prepare_arguments()
            response = self.call_model(extra_param)
            return LLMOutputSchema(
                model_name=self.model_id,
                response=response.get("response", ""),
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                status="success"
            )
            # prepare the output schema
        except Exception as e:
            return LLMOutputSchema(
                model_name=self.model_id,
                response=str(e),
                input_tokens=0,
                output_tokens=0,
                status="error"      
            )