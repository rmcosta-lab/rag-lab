from openai import OpenAI, DefaultHttpxClient
import httpx
from pprint import pprint
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import httpx

load_dotenv(dotenv_path=Path.cwd() / find_dotenv())

# Verifica se carregou
api_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)
if not api_key:
    raise ValueError("OPENAI_API_KEY não foi carregada. Verifique o arquivo .env e a pasta atual.")


base_url = 'https://api.openai.com/v1' # If using together endpoint, add it here https://api.together.xyz/
llm_model = "gpt-5.4-nano"

# Custom transport to bypass SSL verification. This is only needed if using our proxy. Otherwise you can ignore it.
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# Create a DefaultHttpxClient instance with the custom transport
http_client = DefaultHttpxClient(transport=transport)

client = OpenAI(
    api_key = api_key, # Set any as our proxy does not use it. Set the together api key if using the together endpoint.
    base_url=base_url, 
    http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)



def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str =llm_model,
                               open_api_key = api_key,
                              **kwargs):

    # Remove None parameters for OpenAI API - don't set to string 'none'
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_completion_tokens": max_tokens,
        **kwargs
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    response = client.chat.completions.create(**payload)

    try:
        output_dict = {'role': response.choices[-1].message.role, 'content': response.choices[-1].message.content}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")

    return output_dict

def generate_with_multiple_input(messages: List[Dict],
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str =llm_model,
                               open_api_key = api_key,
                              **kwargs):

    # Remove None parameters for OpenAI API - don't set to string 'none'
    if top_p is None:
        payload_top_p = None
    else:
        payload_top_p = top_p
    if temperature is None:
        payload_temperature = None
    else:
        payload_temperature = temperature

    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        **kwargs
    }
    # Only add temperature and top_p if they're not None
    if payload_temperature is not None:
        payload["temperature"] = payload_temperature
    if payload_top_p is not None:
        payload["top_p"] = payload_top_p

    response = client.chat.completions.create(**payload)

    try:
        output_dict = {'role': response.choices[-1].message.role, 'content': response.choices[-1].message.content}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")

    return output_dict


def call_llm_with_context(prompt: str, context: list,  role: str = 'user', **kwargs):
    """
    Calls a language model with the given prompt and context to generate a response.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).

    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """

    # Append the dictionary {'role': role, 'content': prompt} into the context list
    context.append({'role': role, 'content': prompt})

    # Call the llm with multiple input passing the context list and the **kwargs
    response = generate_with_multiple_input(context, **kwargs)

    # Append the LLM response in the context dict
    context.append(response)

    return response