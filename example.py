from llm.Controller import LLMController
print("Starting LLM Controller...   ")
controller = LLMController(
    model_id="gemma3:1b",
    user_prompt="What is the capital of France?",
    
)

response = controller()
print(response)
# print("calling ollama model")