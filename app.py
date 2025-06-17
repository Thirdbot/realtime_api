import os
from huggingface_hub import login
from dotenv import load_dotenv
from api import LLMFLow
load_dotenv()

# Set environment variables for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["CC"] = "gcc"  # Set C compiler


login(token=os.getenv("HF_TOKEN"))


text_generation_model = "facebook/opt-1.3b"  # 0.5B parameters model

llm = LLMFLow(text_generation_model)
llm.create_llm()
llm.create_rag()

print(llm.invoke("What is ram"))