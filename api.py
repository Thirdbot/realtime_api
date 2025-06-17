from vllm import LLM, SamplingParams

# from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from huggingface_hub import login
import os
from dotenv import load_dotenv
import json
import string
import random

from retrival import RAG

# Import torch for memory management
import torch

load_dotenv()

# Set environment variables for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["CC"] = "gcc"  # Set C compiler


login(token=os.getenv("HF_TOKEN"))

#load models abunch of it and type
text_generation_model = "openbmb/MiniCPM4-0.5B"  # 0.5B parameters model

#create a multimodal jinja template and conditional phrasing template for the model to use

#create an api like chat,tools,realtime,camera,microphone,files ,etc..
#use database api for chat store and format and retrieve like image,audio,etc.. assume its can be rag also

#realtime api load a bunch of models in its own type and combine output as output type etc...

#distributed api load model as its type and output as its type


propmt = PromptTemplate(
    template="""
    You are an assistant for question-answering tasks.
    You are given a question and a context.
    You need to answer the question based on the context.
    If you don't know the answer, just say that you don't know.
    Do not try to make up an answer.
    use three sentences and keep the answer short and concise.
    Question: {question}
    Context: {context}
    Answer:
    """
)

class VLLMRunnable(Runnable):
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sampling_params = SamplingParams(
            temperature=0.5,  # Lower temperature for more focused answers
            top_p=0.9,
            max_tokens=512,  # Shorter responses to avoid repetition
            stop=["\n\n", "Question:", "Answer:"],  # Stop at natural boundaries
            frequency_penalty=1.0,  # Reduce repetition
            presence_penalty=0.6  # Encourage diverse content
        )
        
    def invoke(self, input: str, config=None) -> str:
        outputs = self.llm.generate(str(input), self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response

try:
    vllm_model = LLM(
        model=text_generation_model,
        gpu_memory_utilization=0.5,
        max_model_len=32768,
        dtype="float16",
        trust_remote_code=True,
        max_num_batched_tokens=2048,  # Reduced batch size
        max_num_seqs=16  # Reduced concurrent sequences
    )
    llm = VLLMRunnable(vllm_model)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying with more conservative settings...")
    try:
        vllm_model = LLM(
            model=text_generation_model,
            gpu_memory_utilization=0.4,
            max_model_len=32768,
            dtype="float16",
            trust_remote_code=True,
            max_num_batched_tokens=1024,
            max_num_seqs=8
        )
        llm = VLLMRunnable(vllm_model)
    except Exception as e:
        print(f"Failed to load model with conservative settings: {e}")
        raise

rag = RAG()
retriver = rag.create_vector_store()

rag_chain = propmt | llm | StrOutputParser()

# Test with a few different questions
questions = [
    "What are the main characteristics of mammals?",
    "How do birds adapt to their environment?",
    "What is the role of insects in ecosystems?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    docs = retriver.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    print(f"Answer: {generation}")
    print("-" * 80)




# chat_template = """{% for message in messages %}
# {% if message['role'] == 'system' %}
# {{ message['content'] }}
# {% elif message['role'] == 'user' %}
# User: {{ message['content'] }}
# {% elif message['role'] == 'assistant' %}
# Assistant: {{ message['content'] }}
# {% endif %}
# {% endfor %}
# Assistant:"""

# sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
