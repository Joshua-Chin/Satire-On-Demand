import random


SEP = '#~#'

with open('./prompts/unguided-generation.txt') as f:
    PROMPTS_NO_TOPICS = [prompt.strip() for prompt in f]

def get_as_messages(data):
    headline = get_headline(data['text'])
    messages = get_prompt()
    messages.append(format_reply(headline))
    return {'messages': messages}

def get_text_as_messages(data):
    messages = get_prompt()
    messages.append(format_reply(data['text']))
    return {'messages': messages}
    
def get_headline(text):
    return text.split(SEP)[0].strip()

def get_prompt():
    return [
        {
            "role": "user", 
            "content": get_prompt_no_topic(),
        }
    ]
    
def get_prompt_no_topic():
    return random.choice(PROMPTS_NO_TOPICS)

def format_reply(headline):
    return {
        "role": "assistant",
        "content": headline,
    }