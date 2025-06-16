from vllm import LLM, SamplingParams
from huggingface_hub import login
import os
from dotenv import load_dotenv
import json
import string
import random

load_dotenv()

# Set environment variables for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["CC"] = "gcc"  # Set C compiler

login(token=os.getenv("HF_TOKEN"))

# Using a smaller model that's more likely to fit in memory
text_generation_model = "facebook/opt-1.3b"  # 1.3B parameters model

try:
    llm = LLM(
        model=text_generation_model,
        gpu_memory_utilization=0.7,
        max_model_len=512,
        dtype="bfloat16",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying with more conservative settings...")
    try:
        llm = LLM(
            model=text_generation_model,
            gpu_memory_utilization=0.5,
            max_model_len=256,
            dtype="bfloat16",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load model with conservative settings: {e}")
        raise
# A chat template can be optionally supplied.
# If not, the model will use its default chat template.

chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}
{{ message['content'] }}
{% elif message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant:"""

sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
