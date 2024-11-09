import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import uuid
import json
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from rouge import Rouge
import numpy as np

app = FastAPI()

def setup_logging(log_file='phi3_api.log', log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024,
                                       backupCount=5)  # 10MB per file, max 5 files

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_format = logging.Formatter(log_format)
    file_format = logging.Formatter(log_format)
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging(log_file='llama_api.log')

# User management
users = {}
api_key_header = APIKeyHeader(name="X-API-Key")

class UserModel:
    def __init__(self, api_key):
        self.api_key = api_key

def get_current_user(api_key: str = Depends(api_key_header)):
    if api_key not in users:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return users[api_key]

# Persistence functions
def save_users():
    user_data = {api_key: user.api_key for api_key, user in users.items()}
    with open("users.json", "w") as f:
        json.dump(user_data, f)

def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            user_data = json.load(f)
        for api_key in user_data:
            users[api_key] = UserModel(api_key)
        logger.info(f"Loaded {len(users)} users")

# Model setup
global_pipeline = None

def load_llm():
    global global_pipeline
    model_id = "/data/models/LLaMa3/Meta-Llama-3-8B-Instruct-hf/"
    global_pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    logger.info("Model loaded successfully!")

load_llm()

chat_model_name = "microsoft/Phi-3-medium-128k-instruct"
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name, trust_remote_code=True, device_map="auto")

# Utility functions
def extract_bracket_content(text):
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip('"')
    return None

def load_knowledge_base(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

# FastAPI models
class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0,
    encoder_repetition_penalty: float = 1.0

    

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]

# FastAPI routes
@app.on_event("startup")
async def startup_event():
    load_users()

@app.post("/v1/register")
async def register_user():
    api_key = str(uuid.uuid4())
    users[api_key] = UserModel(api_key)
    save_users()
    return {"api_key": api_key}


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest, user: UserModel = Depends(get_current_user)):
    try:
        logger.info(f"Received prompt: {request.prompt}")
        input_ids = chat_tokenizer.encode(request.prompt, return_tensors="pt", truncation=True, max_length=1024)

        logger.info(f"Tokenized input length: {input_ids.shape[1]}")
        max_length = min(input_ids.shape[1] + request.max_new_tokens, chat_model.config.max_position_embeddings)

        logger.info(
            f"Generating with parameters: max_length={max_length}, temperature={request.temperature}, top_p={request.top_p}")
        with torch.no_grad():
            output = chat_model.generate(
                input_ids,
                max_length=max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.n,
                do_sample=True,
                pad_token_id=chat_tokenizer.eos_token_id,
                repetition_penalty = request.repetition_penalty,
                encoder_repetition_penalty = request.encoder_repetition_penalty
            )

        generated_texts = [chat_tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
        completions = [text[len(request.prompt):] for text in generated_texts]
        logger.info(f"Generated completions: {completions}")

        # if request.stop:
        #     logger.info(f"Applying stop sequences: {request.stop}")
        #     for i, completion in enumerate(completions):
        #         for stop_seq in request.stop:
        #             stop_index = completion.find(stop_seq)
        #             if stop_index != -1:
        #                 completions[i] = completion[:stop_index]
        #                 logger.info(f"Applied stop sequence to completion {i}")

        choices = [{"text": compl, "index": i, "logprobs": None, "finish_reason": "length"} for i, compl in
                   enumerate(completions)]

        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            object="text_completion",
            created=int(torch.rand(1).item() * 1000000000),
            model=chat_model_name,
            choices=choices
        )

        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/v1/completions", response_model=CompletionResponse)
# async def create_completion(request: CompletionRequest, user: UserModel = Depends(get_current_user)):
#     try:
#         logger.info(f"Received completion prompt: {request.prompt}")

#         messages = [
#             {"role": "user", "content": request.prompt},
#         ]
#         terminators = [
#             global_pipeline.tokenizer.eos_token_id,
#             global_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]
        
#         outputs = global_pipeline(
#             messages,
#             max_new_tokens=request.max_new_tokens,
#             # eos_token_id=terminators,
#             # do_sample=True,
#             temperature=request.temperature,
#             top_p=request.top_p,
#             num_return_sequences=request.n,
#         )
        
#         completions = []
#         for output in outputs:
#             result = output["generated_text"]
#             assistant_message = next((msg for msg in result if msg['role'] == 'assistant'), None)
#             if assistant_message:
#                 completions.append(assistant_message['content'])
#             else:
#                 completions.append(result[-1]['content'])

#         logger.info(f"Generated completions: {completions}")

#         if request.stop:
#             for i, completion in enumerate(completions):
#                 for stop_seq in request.stop:
#                     stop_index = completion.find(stop_seq)
#                     if stop_index != -1:
#                         completions[i] = completion[:stop_index]

#         choices = [{"text": compl, "index": i, "logprobs": None, "finish_reason": "length"} for i, compl in enumerate(completions)]

#         response = CompletionResponse(
#             id=f"cmpl-{uuid.uuid4()}",
#             object="text_completion",
#             created=int(torch.rand(1).item() * 1000000000),
#             model="llama-3-local",
#             choices=choices
#         )

#         logger.info(f"Returning completion response: {response}")
#         return response
#     except Exception as e:
#         logger.error(f"An error occurred in completion: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

class JudgeRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None

@app.post("/v1/judge", response_model=CompletionResponse)
async def create_judge(request: CompletionRequest, user: UserModel = Depends(get_current_user)):
    try:
        logger.info(f"Received judge prompt: {request.prompt}")

        kb_file = "./fol_rules_kb/all_rules.txt"
        properties_file = "./fol_rules_kb/gt_vehicles_info_017_a.txt"
        
        kb_content = load_knowledge_base(kb_file)
        properties_content = load_knowledge_base(properties_file)

        user_que = request.prompt.split('=')[-2].strip()
        gpt_fol = request.prompt.split('=')[-1].strip()
        
        
        system_prompt = f"""You are an AI assistant tasked with generating queries for a first-order logic knowledge base. Your task is to create a FOL query that can be used to answer the provided question.

    Knowledge Base Rules:
    {kb_content}

    Object Properties:
    {properties_content}

    Some more examples and explanations:
    TypeOf(x, Car)
    ColorOf(x, Red)

    Object properties take the type of vehicles and color. These can be retrieved from the question. But the color and type need
    to be capitalized, and any space in it should be removed. For example, a Police car should be PoliceCar.

    Question: {user_que}

    Based on the knowledge base rules and object properties, generate an FOL query that would be necessary to answer the question. The queries should be in the format used by the AIMA Python library's fol_fc_ask function.

    Rules for generating queries:
    1. Each query should be a string that might be a conjunction of multiple predicates.
    2. Use 'x' as the variable name for the main object in question.
    3. The predicates should be combined to form a conjunctive query.
    4. Include predicates for type, attributes, and relevant relationships.
    5. Please Do not use any predicate that is not present in the Knowledge Base Rules.
    6. Please use the predicates that are present in the rules. Please do not make any changes in predicates.
    7. Check the location of the object if the location is mentioned in the question.
    8. Questions that involve a single object should be responded to with a query which is a conjunction of some predicates.

    For example. For Location, use InitialLocation(x, position) or LastLocation(x, position) predicates. Don't use only Location(x). Then, create another list of actions between the two objects.

    Example 1:
    Question: "Is there a white car near the left?"
    Your response should be: ["TypeOf(x, Car)^ColorOf(x, White)^InitialLocation(x, NearLeft)"]

    10. If the question involves two objects, then first generate the FOL query for each object similar to a single object question.
    11. Use 'y' as the variable name for the second object in the question.
    12. For two objects, the response would be the conjunction of all the predicates for both objects, which contains the predicates for the first object, the predicates for the second object, and the predicates for the interaction between them (if available).
    
    Example 2:
    Question: "Does the white car near the left come close to a pedestrian at the front?"
    Your response should be: ["TypeOf(x, Car)^ColorOf(x, White)^InitialLocation(x, NearLeft)^Pedestrian(y)^InitialLocation(y, NearFront)^ComeClose(x, y)"]

    Example 3:
    Question: "Can you spot a pedestrian walking near the right of the police car?"
    Your response should be: ["Pedestrian(x)^Walk(x)^InitialLocation(x, NearRight)^TypeOf(y, PoliceCar)"]

    Example 4:
    Question: "Can you spot a pedestrian walking near the right of the police car at the center?"
    Your response should be: ["Pedestrian(x)^Walk(x)^InitialLocation(x, NearRight)^TypeOf(y, PoliceCar)^InitialLocation(x, NearFront)"]

    In the above examples (example 3, 4), notice that both the examples mentioned about police car. But in example 3 there was no mention of location of the police car, hence the FOL query does not include any location for the police car. But in the 4th example, as there is a mention of the police car's position, we include it in the FOL query.

    Location can be any of the following: NearLeft, FarLeft, NearRight, FarRight, NearFront, and FarFront. Please analyze the question
    Choose the proper one, and do not use any words other than the mentioned 6 for location.
    Inside the location  predicates (InitialLocation, LastLocation) please don't use any other location other than these six: NearLeft, FarLeft, NearRight, FarRight, NearFront, and FarFront
    If a synonym is used in the question, try to find the closest one from the six mentioned positions.
    
    For example, the center can be replaced by NearFront.
    For location, we only have two predicates. Example: InitialLocation(x, NearLeft) and LastLocation(x, NearLeft)

    Please provide a similar python list of a query for the given question. Ensure that the queries cover all aspects necessary to answer the question based on the knowledge base and object properties.
    Also, please consider some true predicates can lead other predicates to be true. For example: 

    ((Vehicles(x) & SpeedUp(x)) ==> Accelerate(x))
    ((Vehicles(x) & SpeedDown(x)) ==> Decelerate(x))
    (((Vehicles(x) & NotAccelerate(x)) & NotDecelerate(x)) ==> ConstantSpeed(x))

    Here, you do not need to check all the predicates (SpeedUp, SpeedDown, ConstantSpeed) to understand if a vehicle is moving at a constant speed.
    You can only check ConstantSpeed(x).
    
    Your response should be a list of a query string. For example:
    ["Query1(x, Value)^Query2(x)^Query3(x, OtherValue)"]

    Respond with only the Python list of the query string, no additional text. Please do not add any additional text.
        """
        # 9. If the location of an object is not mentioned in the question then do not include InitialLocation or LastLocation predicates in the query.
        # If you are not sure about the location/position of an object seeing the question dont include location predicates for that object.
        logger.info(f"System prompt: {system_prompt}")

        messages = [
            # {"role": "system", "content": "You are an AI assistant specializing in generating logical queries for AIMA Python based on first-order logic knowledge bases and object properties."},
            {"role": "user", "content": system_prompt},
        ]

        terminators = [
            global_pipeline.tokenizer.eos_token_id,
            global_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # inputs = global_pipeline.tokenizer(system_prompt)['input_ids'].shape[1] + request.max_tokens
        
        # logger.info(f"Processed input length: {inputs['input_ids'].shape[1]}")
        # max_length = min(inputs['input_ids'].shape[1] + request.max_tokens, base_model.config.max_position_embeddings)
        
        outputs = global_pipeline(
            messages,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        result = outputs[0]["generated_text"] # outputs[0]["generated_text"][-1]['content']

        assistant_message = next((msg for msg in result if msg['role'] == 'assistant'), None)
        if assistant_message:
            result = assistant_message['content']
        else:
            result = result[-1]['content']

        logger.info(f"Generated judge completions: {result}")
        extracted_result = extract_bracket_content(result).replace("'", "").replace('"', '')

        gpt_fol = gpt_fol.replace("∧", "^")
        extracted_result = extracted_result.replace("∧", "^")
        
        gpt_predicates = gpt_fol.split('^')
        judge_predicates = extracted_result.split('^')

        scores = np.zeros((len(gpt_predicates), len(judge_predicates)))

        rouge = Rouge()
        for i in range(len(gpt_predicates)):
            for j in range(len(judge_predicates)):
                # print(gpt_predicates[i], gt_predicates[j])
                gpt_r_i = gpt_predicates[i].strip()
                judge_r_i = judge_predicates[j].strip()
                try:
                    scores[i, j] = rouge.get_scores(gpt_r_i, judge_r_i)[0]['rouge-l']['f']
                except:
                    scores[i, j] = 0.0


        if scores.shape[0] >= scores.shape[1]:
            rouge_l_f1 = np.max(scores, axis=1).mean()
        else:
            rouge_l_f1 = np.max(scores, axis=0).mean()
        
        # scores = rouge.get_scores(gpt_fol, extracted_result)
        # rouge_l_f1 = scores[0]['rouge-l']['f']
        threshold = 0.7

        extracted_result = f"ROUGE-L F1: {rouge_l_f1:.2f}"
            
        final_result = [extracted_result] if extracted_result else []

        choices = [{"text": compl, "index": i, "logprobs": None, "finish_reason": "length"} for i, compl in enumerate(final_result)]
        
        response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            object="text_completion",
            created=int(torch.rand(1).item() * 1000000000),
            model="llama-3-local",
            choices=choices
        )

        logger.info(f"Returning judge response: {response}")
        return response
    except Exception as e:
        logger.error(f"An error occurred in judge: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8899)