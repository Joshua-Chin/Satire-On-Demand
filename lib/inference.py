import gc
import warnings

import torch
from unsloth import FastLanguageModel


unsloth_generation_args = {
    "num_return_sequences": 5,
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
}

unsloth_default_prompts = {
    "unguided": [{"role": "user", "content": "Write a satirical headline in the style of Onion News."}],
    "trump":  [{"role": "user", "content": "Write a satirical headline in the style of Onion News about Donald Trump."}]
}

def sample_unsloth(
    model_name,
    prompts=unsloth_default_prompts,
    generation_args=unsloth_generation_args,
    seed=42,
):
    # Set random seed
    torch.random.manual_seed(seed)
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
    )
    model = FastLanguageModel.for_inference(model)
    # Process each prompt independently
    responses = {}
    for name, messages in prompts.items():
        # Parse the messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        # Generate the response
        with warnings.catch_warnings():
            # unsloth issues a warning when `num_return_sequences` is provided.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="An output with one or more elements was resized.*"
            )
            outputs = model.generate(**inputs, **generation_args)
        # Decode the response
        output_texts = []
        for output_tokens in outputs:
            new_tokens = output_tokens[inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            output_texts.append(output_text)
        responses[name] = output_texts

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return responses