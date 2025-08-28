from threading import Thread

import gradio as gr

from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer


generation_args = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = './finetunes/03-unsloth-CoT/checkpoint-60'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
)
model = FastLanguageModel.for_inference(model)

def generate_text(prompt):
    # tokenize the inputs
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # create the streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # run generation in a separate thread
    kwargs = {
        **inputs,
        **generation_args,
        "streamer": streamer,
    }
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    # decode the outputs
    full_response = ""
    for new_text in streamer:
        full_response += new_text
        yield full_response


interface = gr.Interface(
    fn=generate_text,
    inputs=["text"],
    outputs=["text"],
)
interface.launch()