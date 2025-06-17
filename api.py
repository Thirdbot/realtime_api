from vllm import LLM, SamplingParams

# from langchain_community.llms import VLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

import json
import string
import random

from retrival import RAG

# Import torch for memory management
import torch


#load models abunch of it and type


#create a multimodal jinja template and conditional phrasing template for the model to use

#create an api like chat,tools,realtime,camera,microphone,files ,etc..
#use database api for chat store and format and retrieve like image,audio,etc.. assume its can be rag also

#realtime api load a bunch of models in its own type and combine output as output type etc...

#distributed api load model as its type and output as its type


propmt = PromptTemplate(
    template="""
    
    Question: {question}
    
    Context: {context}
    
    Please provide a clear and concise answer based on the context above. If the context doesn't contain relevant information, please say so.
    
    Answer:
    """
)

class VLLMRunnable(Runnable):
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sampling_params = SamplingParams(
            temperature=0.3,  # Lower temperature for more focused answers
            top_p=0.9,
            max_tokens=1024,  # Shorter responses to avoid repetition
            frequency_penalty=1.0,  # Reduce repetition
            presence_penalty=0.6  # Encourage diverse content
        )
        
    def invoke(self, input: str, config=None) -> str:
        outputs = self.llm.generate(str(input), self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response


class LLMFLow(VLLMRunnable):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vllm_model = None
        self.llm = None
        self.rag_chain = None
        self.retriver = None

    def create_llm(self):
        try:
            self.vllm_model = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.7,
                dtype="float16",
                trust_remote_code=True,
                max_num_batched_tokens=1024,  # Reduced batch size
                max_num_seqs=16  # Reduced concurrent sequences
            )
            self.llm = VLLMRunnable(self.vllm_model)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with more conservative settings...")
            try:
                self.vllm_model = LLM(
                    model=self.model_name,
                    gpu_memory_utilization=0.4,
                    dtype="float16",
                    trust_remote_code=True,
                    max_num_batched_tokens=1024,
                    max_num_seqs=8
                )
                self.llm = VLLMRunnable(self.vllm_model)
            except Exception as e:
                print(f"Failed to load model with conservative settings: {e}")
                raise
    def create_rag(self):
        rag = RAG()
        self.retriver = rag.create_vector_store()
        self.rag_chain = propmt  | self.llm | StrOutputParser()

    def invoke(self, question: str):
        #one day this will run as api
        docs = self.retriver.invoke(question)
        
        # Format the context from retrieved documents
        formatted_context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
        
        generation = self.rag_chain.invoke({
            "context": formatted_context,
            "question": question
        })
        return generation


