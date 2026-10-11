import gc
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-tc-big-he-en"
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def translate_prompt_if_hebrew(prompt: str) -> str:
    """Translate Hebrew prompts to English locally; leave English prompts untouched."""
    prompt = (prompt or "").strip()
    if not HEBREW_RE.search(prompt):
        print("Prompt contains no Hebrew characters; using it as English.")
        return prompt

    print(f"Hebrew detected; translating prompt locally with {TRANSLATION_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATION_MODEL,
        use_safetensors=True,
    ).to("cpu")
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=2,
            do_sample=False,
        )
    translated = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    del model, tokenizer, inputs, output
    gc.collect()
    print(f"Translated prompt: {translated}")
    return translated or prompt
