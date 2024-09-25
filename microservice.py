from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from format_prompt import get_formatted_prompt
import json
import requests
import logging
model = "tinyllama"
ollama_host = "http://localhost:11434"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Create an instance of the FastAPI
app = FastAPI()

# structure of the input data
class TextData(BaseModel):
    text_: str
    category_list: List[str]
@app.get('/')
async def run_app():
    return {"detail": f"The API is running!!!"}

# Define an endpoint that accepts POST requests
@app.post("/process_text/")
async def process_text(data: TextData) -> Dict[str, str]:
    # Access the input data
    text_ = data.text_
    category_list = data.category_list
    # Validation logic to ensure valid input
 
    if not text_ or not isinstance(text_, str):
        raise HTTPException(status_code=400, detail="Invalid text_. It must be a non-empty string.")
    
    if not category_list or not isinstance(category_list, list) or len(category_list) == 0:
        raise HTTPException(status_code=400, detail="Invalid category_list. It must be a non-empty list.")


    prompt = get_formatted_prompt(text_, category_list)
    # logging.info(prompt)
    data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.5, "top_p": 0, "top_k": 0},
}
    try:
        response = requests.post(f"{ollama_host}/api/generate", json=data, stream=False)
        response.raise_for_status()  # Raises an error for HTTP error responses
        json_data = response.json()  # Directly parse the JSON response
        # Ensure the response format is correct
        if isinstance(json_data['response'], str):
            result = json.loads(json_data['response'])
        elif isinstance(json_data['response'], dict):
            result = json_data['response']
        else:
            return {"error": "Unexpected response format from model server"}
        logging.info("Result Retrieved.")
        return result
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}


# If running this code directly, use uvicorn to start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
