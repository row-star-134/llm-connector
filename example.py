import json
from llm.Controller import LLMController
print("Starting LLM Controller...   ")
controller = LLMController(
    model_id="gemma3:1b",
    llm_service="ollama",
    user_prompt="Extract the all important information from the image and return it in a structured format as json. Don't provide any extrac text with output",
    enable_vision=True,
    vision_images=['c:/Users/praja/Downloads/download.jpg']
    
)

response = controller()
print(response)
# print("calling ollama model")