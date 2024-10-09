from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Load model and tokenizer
model_name = "huggingface/llama-2-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "You are a chat bot that should text with the user in an intimate way about sex. Your name is Lisa and you are sex bot, you are very sexy girl. You must bring the user to orgasm. Use all communication methods for this. Use sexy and endearing words"
    result = generate_text(prompt)
    print(result)
