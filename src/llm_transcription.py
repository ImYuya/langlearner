import os
import requests
import base64
import mimetypes
import PIL.Image
import google.generativeai as genai
from openai import OpenAI

import config
from text_to_speech import text_to_speech

# Set Google API key
genai.configure(api_key=config.GOOGLE_API_KEY)
# make model
gemini_model = genai.GenerativeModel(config.GOOGLE_MODEL)
gemini_model_vision = genai.GenerativeModel(config.GOOGLE_MODEL_VISION)

# Set OpenAI API key
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gemini(prompt, image_path, chatbot=[]):
    """
    Function to handle gemini model and gemini vision model interactions.
    Parameters:
    prompt (str): The prompt text.
    image_path (str): The path to the image file.
    chatbot (list): A list to keep track of chatbot interactions.
    Returns:
    tuple: Updated chatbot interaction list, an empty string, and None.
    """

    messages = []
    # print(f"{chatbot=}")

    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'parts': [user]},
                {'role': 'model', 'parts': [bot]}
            ])
        messages.append({'role': 'user', 'parts': [prompt]})
    else:
        messages.append({'role': 'user', 'parts': [prompt]})

    try:
        # Process image if file is provided
        if image_path is not None:
            with PIL.Image.open(image_path) as img:
                message = [{'role': 'user', 'parts': [prompt, img]}]
                response = gemini_model_vision.generate_content(message)
                gemini_video_resp = response.text
                messages.append({'role': 'model', 'parts': [gemini_video_resp]})

                # Construct list of messages in the required format
                file_data = {
                    "name": os.path.basename(image_path),
                    "path": image_path,
                    "type": mimetypes.guess_type(image_path)[0],
                    "size": os.path.getsize(image_path)
                }
                user_msg = {"text": prompt, "files": [{"file": file_data}]}
                bot_msg = {"text": gemini_video_resp, "files": []}
                chatbot.append([user_msg, bot_msg])
        else:
            response = gemini_model.generate_content(messages)
            gemini_resp = response.text

            # Construct list of messages in the required format
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": gemini_resp, "files": []}
            chatbot.append([user_msg, bot_msg])
    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def ollama(prompt, image_path, chatbot=[]):
    messages = []
    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        # print(f"{chatbot=}")
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': bot}
            ])
        messages.append({'role': 'user', 'content': prompt})
    else:
        messages.append({'role': 'user', 'content': prompt})

    def generate_ollama_response(messages):
        def flatten(lst):
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        context = []
        jsonParam = {
            "model": config.LLM['model'],
            "stream": config.LLM['stream'],
            "context": context,
            "system": config.LLM['systemPrompt'],
            "messages": flatten(messages)
        }
        # print(f"{jsonParam=}")
        response = requests.post(
            config.LLM['url'],
            json=jsonParam,
            headers={'Content-Type': 'application/json'},
            stream=config.LLM['stream'],
            timeout=config.LLM['timeout']
        )  # Set the timeout value as per your requirement
        response.raise_for_status()  # raise exception if http calls returned an error
        
        # for non-streaming response
        body = response.json()
        response = body.get('message', '').get('content', '')
        return response

    try:
        # Process image if file is provided
        if image_path is not None:
            message = [{'role': 'user', 'content': prompt, 'images': [image_to_base64(image_path)]}]
            response = generate_ollama_response(message)
            messages.append({'role': 'assistant', 'content': response})

            # Construct list of messages in the required format
            file_data = {
                "name": os.path.basename(image_path),
                "path": image_path,
                "type": mimetypes.guess_type(image_path)[0],
                "size": os.path.getsize(image_path)
            }
            user_msg = {"text": prompt, "files": [{"file": file_data}]}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])
        else:
            response = generate_ollama_response(messages)
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])

    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def openai(prompt, image_path, chatbot=[]):
    messages = []

    # Process previous chatbot messages if present
    if len(chatbot) != 0:
        # print(f"{chatbot=}")
        for chat in chatbot:
            user, bot = chat[0]['text'], chat[1]['text']
            messages.extend([
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': bot}
            ])
        messages.append({'role': 'user', 'content': prompt})
    else:
        if config.LLM['systemPrompt']:
            messages.append({'role': 'system', 'content': config.LLM['systemPrompt']})
        messages.append({'role': 'user', 'content': prompt})
    
    def generate_openai_response(messages):
        def flatten(lst):
            return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        # print(f"{messages=}")
        if isinstance(messages[-1]['content'], list) and 'type' in messages[-1]['content'][-1] and messages[-1]['content'][-1]['type'] == 'image_url':
            model = config.OPENAI_MODEL_VISION
        else:
            model = config.OPENAI_MODEL
        # print(f"{model=}")
        # print(f"{messages=}")
        # print(f"{flatten(messages)=}")
        # print("=====================")
        response = openai_client.chat.completions.create(
            model=model,
            messages=flatten(messages),
            max_tokens=300,
            # stream=config.LLM['stream']
        )
        # print(f"{response=}")

        # for non-streaming response
        response = response.choices[0].message.content
        return response

    try:
        # Process image if file is provided
        if image_path is not None:
            base64_image = image_to_base64(image_path)
            message = [
                {
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
            response = generate_openai_response(message)
            messages.append({'role': 'assistant', 'content': response})

            user_msg = {"text": prompt, "files": [{"file": image_path}]}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])
        else:
            response = generate_openai_response(messages)
            user_msg = {"text": prompt, "files": []}
            bot_msg = {"text": response, "files": []}
            chatbot.append([user_msg, bot_msg])

    except Exception as e:
        # Handling exceptions and raising error to the modal
        print(f"An error occurred: {e}")

    return chatbot

def ask_llm(prompt, image_path=None, chatbot=[]):
    if config.LLM['model'] not in ["gemini", "openai"]:
        # print("Using ollama model")
        chatbot = ollama(prompt, image_path=image_path, chatbot=chatbot)
    elif config.LLM['model'] == "gemini":
        # print("Using gemini model")
        chatbot = gemini(prompt, image_path=image_path, chatbot=chatbot)
    elif config.LLM['model'] == "openai":
        # print("Using openai model")
        chatbot = openai(prompt, image_path=image_path, chatbot=chatbot)  # image_path should be URL
    return chatbot
    
if __name__ == "__main__":
    # chatbot is [[user_msg, bot_msg], [user_msg, bot_msg], ...]

    # paturn 1:  text (example: gemini-pro)
    print("=====================================")
    system_prompt_for_the_first = f"System: {config.LLM['systemPrompt']} \n User:"
    chatbot = ask_llm(prompt=f"{system_prompt_for_the_first} how to output csv from dataframe in python", chatbot=[])
    user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    print(f"{system_prompt_for_the_first.replace('User:', '')}")
    print(f"user: {user.replace(system_prompt_for_the_first, '')}")
    print(f"bot: {bot}")
    # text_to_speech(text=bot)
    print("=====================================")
    chatbot = ask_llm(prompt="how to read it in python", chatbot=chatbot)
    user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    print(f"user: {user}")
    print(f"bot: {bot}")
    # text_to_speech(text=bot)

    # paturn 2:  upload images (example: gemini-pro-vision + gemini-pro)
    # print("=====================================")
    # system_prompt_for_the_first = f"System: {config.LLM['systemPrompt']} \n User: "
    # chatbot = ask_llm(prompt=f"{system_prompt_for_the_first} What is this image? Discribe the image in detail.", image_path="./temp/temp.jpg")
    # user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    # print(f"user: {user}")
    # print(f"bot: {bot}")
    # # text_to_speech(text=bot)
    # print("=====================================")
    # chatbot = ask_llm(prompt="Please describe the image from a different perspective.", chatbot=chatbot)
    # user, bot = chatbot[-1][0]['text'], chatbot[-1][1]['text']
    # print(f"user: {user}")
    # print(f"bot: {bot}")
    # # text_to_speech(text=bot)
