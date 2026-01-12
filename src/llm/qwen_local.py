import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# def load_qwen_sat(lora_path: str):
#     tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)

#     base_model = AutoModelForCausalLM.from_pretrained(
#         QWEN_MODEL,
#         torch_dtype=torch.float32,
#         device_map=None
#     )

#     model = PeftModel.from_pretrained(base_model, lora_path)
#     model.eval()

#     return tokenizer, model
# chạy sau ----------

def load_qwen_base():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    return tokenizer, model



def qwen_reasoning(query, context, tokenizer, model, max_tokens=128):
    prompt = f"""Context:
{context}

Question:
{query}

Reasoning:"""

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")

    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Reasoning:")[-1].strip()

