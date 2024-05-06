from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# Load GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

@app.post('/generate/')
async def generate_text(prompt: dict):
    # Get prompt
    prompt_text = prompt["prompt"]
    # Tokenize input text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    # Generate text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, 
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True, top_p=0.95, temperature=0.7)
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"prompt": prompt_text, "generated_text": generated_text}
