import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

app = FastAPI()

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    try:
        logger.info(f"Received prompt: {request.prompt}")

        input_ids = tokenizer.encode(request.prompt, return_tensors="pt", truncation=True, max_length=512)
        
        logger.info(f"Tokenized input length: {input_ids.shape[1]}")

        max_length = min(input_ids.shape[1] + request.max_tokens, model.config.max_position_embeddings)
        
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.n,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

        completions = [text[len(request.prompt):] for text in generated_texts]

        logger.info(f"Generated completions: {completions}")

        if request.stop:
            for i, completion in enumerate(completions):
                for stop_seq in request.stop:
                    stop_index = completion.find(stop_seq)
                    if stop_index != -1:
                        completions[i] = completion[:stop_index]

        choices = [{"text": compl, "index": i, "logprobs": None, "finish_reason": "length"} for i, compl in enumerate(completions)]

        return CompletionResponse(
            id="cmpl-" + model_name,
            object="text_completion",
            created=int(torch.rand(1).item() * 1000000000),
            model=model_name,
            choices=choices
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8899)
