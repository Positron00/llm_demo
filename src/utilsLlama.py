# The code below should be added to a util.py file
import os
import requests
import json
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from pygments import highlight, lexers, formatters
from wolframalpha import Client

def load_env():
    _ = load_dotenv(find_dotenv())

  # The right API to pass in a prompt (of type string) is the completions API https://docs.together.ai/reference/completions-1
  # The right API to pass in a messages (of type of list of message) is The chat completions API https://docs.together.ai/reference/chat-completions-1

def get_wolfram_alpha_api_key():
    load_env()
    wolfram_alpha_api_key = os.getenv("WOLFRAM_ALPHA_KEY")
    return wolfram_alpha_api_key

def get_tavily_api_key():
    load_env()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return tavily_api_key

def llama31(prompt_or_messages, model_size=8, temperature=0, raw=False, debug=False):
    model = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct-Turbo"
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }
    else:
        messages = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

    if debug:
        print(payload)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()  # Raises HTTPError for bad responses
        res = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    if 'error' in res:
        raise Exception(f"API Error: {res['error']}")

    if raw:
        return res

    if isinstance(prompt_or_messages, str):
        return res['choices'][0].get('text', '')
    else:
        return res['choices'][0].get('message', {}).get('content', '')

def llama32(messages, model_size=11):
  model = f"meta-llama/Llama-3.2-{model_size}B-Vision-Instruct-Turbo"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
  payload = {
    "model": model,
    "max_tokens": 4096,
    "temperature": 0.0,
    "stop": ["<|eot_id|>","<|eom_id|>"],
    "messages": messages
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  return res['choices'][0]['message']['content']

def llamaguard3(prompt, debug=False):
  model = "meta-llama/Meta-Llama-Guard-3-8B"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
  payload = {
    "model": model,
    "temperature": 0,
    "prompt": prompt,
    "max_tokens": 4096,
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  if debug:
    print(res)
  return res['choices'][0]['text']

def html_tokens(tokens):
  # simulate the color values used in https://tiktokenizer.vercel.app
  on_colors = ["#ADE0FC", "#FCE278", "#B2D1FE", "#AFF7C6", "#FDCE9B", "#97F1FB", "#DEE1E7", "#E3C9FF", "#BBC6FD", "#D1FB8C"]

  # Create an HTML string with colored spans
  html_string = ""
  for i, t in enumerate(tokens):
      if t == "\n":
            t = "\\n"
      elif t == "\n\n":
            t = "\\n\\n"
      on_col = on_colors[i % len(on_colors)]
      html_string += f'<span style="color: black; background-color: {on_col}; padding: 2px;">{t}</span>'

  return html_string

def disp_image(address):
    if address.startswith("http://") or address.startswith("https://"):
        response = requests.get(address)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(address)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def resize_image(img, max_dimension = 1120):
  original_width, original_height = img.size

  if original_width > original_height:
      scaling_factor = max_dimension / original_width
  else:
      scaling_factor = max_dimension / original_height

  new_width = int(original_width * scaling_factor)
  new_height = int(original_height * scaling_factor)

  # Resize the image while maintaining aspect ratio
  resized_img = img.resize((new_width, new_height))

  resized_img.save("images/resized_image.jpg")

  print("Original size:", original_width, "x", original_height)
  print("New size:", new_width, "x", new_height)

  return resized_img

def merge_images(image_1, image_2, image_3):
    img1 = Image.open(image_1)
    img2 = Image.open(image_2)
    img3 = Image.open(image_3)
    
    width1, height1 = img1.size
    width2, height2 = img2.size
    width3, height3 = img3.size
    
    print("Image 1 dimensions:", width1, height1)
    print("Image 2 dimensions:", width2, height2)
    print("Image 3 dimensions:", width3, height3)
    
    total_width = width1 + width2 + width3
    max_height = max(height1, height2, height3)
    
    merged_image = Image.new("RGB", (total_width, max_height))
    
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (width1, 0))
    merged_image.paste(img3, (width1 + width2, 0))
    
    merged_image.save("images/merged_image_horizontal.jpg")
    
    print("Merged image dimensions:", merged_image.size)
    return merged_image

# pretty print JSON with syntax highlighting
def cprint(response):
    formatted_json = json.dumps(response, indent=4)
    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(colorful_json)

import nest_asyncio
nest_asyncio.apply()

def wolfram_alpha(query: str) -> str:
    WOLFRAM_ALPHA_KEY = get_wolfram_alpha_api_key()
    client = Client(WOLFRAM_ALPHA_KEY)
    result = client.query(query)

    results = []
    for pod in result.pods:
        if pod["@title"] == "Result" or pod["@title"] == "Results":
          for sub in pod.subpods:
            results.append(sub.plaintext)

    return '\n'.join(results)

def get_boiling_point(liquid_name, celsius):
  # function body
  return []

import base64
from IPython.display import Image, display

def mm(graph):
  graphbytes = graph.encode("ascii")
  base64_bytes = base64.b64encode(graphbytes)
  base64_string = base64_bytes.decode("ascii")
  display(Image(url="https://mermaid.ink/img/" + base64_string))

def genai_app_arch():
  mm("""
  flowchart TD
    A[Users] --> B(Applications e.g. mobile, web)
    B --> |Hosted API|C(Platforms e.g. Custom, HuggingFace, Replicate)
    B -- optional --> E(Frameworks e.g. LangChain)
    C-->|User Input|D[Llama 3]
    D-->|Model Output|C
    E --> C
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def rag_arch():
  mm("""
  flowchart TD
    A[User Prompts] --> B(Frameworks e.g. LangChain)
    B <--> |Database, Docs, XLS|C[fa:fa-database External Data]
    B -->|API|D[Llama 3]
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama2_family():
  mm("""
  graph LR;
      llama-2 --> llama-2-7b
      llama-2 --> llama-2-13b
      llama-2 --> llama-2-70b
      llama-2-7b --> llama-2-7b-chat
      llama-2-13b --> llama-2-13b-chat
      llama-2-70b --> llama-2-70b-chat
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama3_family():
  mm("""
  graph LR;
      llama-3 --> llama-3-8b
      llama-3 --> llama-3-70b
      llama-3-8b --> llama-3-8b
      llama-3-8b --> llama-3-8b-instruct
      llama-3-70b --> llama-3-70b
      llama-3-70b --> llama-3-70b-instruct
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)
  
def llama3_1_family():
  mm("""
  graph LR;
      llama-3-1 --> llama-3-8b
      llama-3-1 --> llama-3-70b
      llama-3-1 --> llama-3-4050b
      llama-3-1-8b --> llama-3-1-8b
      llama-3-1-8b --> llama-3-1-8b-instruct
      llama-3-1-70b --> llama-3-1-70b
      llama-3-1-70b --> llama-3-1-70b-instruct
      llama-3-1-405b --> llama-3-1-405b-instruct
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

import ipywidgets as widgets
from IPython.display import display, Markdown

# Create a text widget
API_KEY = widgets.Password(
    value='',
    placeholder='',
    description='API_KEY:',
    disabled=False
)

def md(t):
  display(Markdown(t))

def bot_arch():
  mm("""
  graph LR;
  user --> prompt
  prompt --> i_safety
  i_safety --> context
  context --> Llama_3
  Llama_3 --> output
  output --> o_safety
  i_safety --> memory
  o_safety --> memory
  memory --> context
  o_safety --> user
  classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def fine_tuned_arch():
  mm("""
  graph LR;
      Custom_Dataset --> Pre-trained_Llama
      Pre-trained_Llama --> Fine-tuned_Llama
      Fine-tuned_Llama --> RLHF
      RLHF --> |Loss:Cross-Entropy|Fine-tuned_Llama
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def load_data_faiss_arch():
  mm("""
  graph LR;
      documents --> textsplitter
      textsplitter --> embeddings
      embeddings --> vectorstore
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def mem_context():
  mm("""
      graph LR
      context(text)
      user_prompt --> context
      instruction --> context
      examples --> context
      memory --> context
      context --> tokenizer
      tokenizer --> embeddings
      embeddings --> LLM
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)
