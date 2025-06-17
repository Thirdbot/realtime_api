from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_milvus import Milvus
from pymilvus import MilvusClient
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()


class RAG:
    def __init__(self):
        self.sentence_transformer_model = "sentence-transformers/all-MiniLM-L6-v2"
        #demo url about animal the realone will get from inputs like web_link or pdf or model suggestion or etc..
        self.urls = [
            "https://en.wikipedia.org/wiki/Animal",
            "https://en.wikipedia.org/wiki/Bird",
            "https://en.wikipedia.org/wiki/Fish",
            "https://en.wikipedia.org/wiki/Reptile"
        ]

        # Initialize Milvus client with proper environment variables
        mivus_client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
        
        docs = [WebBaseLoader(url).load() for url in self.urls]
        doc_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0,
        )

        self.doc_split = text_splitter.split_documents(doc_list)

    def create_vector_store(self):
        #Add to vector store DB
        vector_store = Milvus.from_documents(
            documents=self.doc_split,
            collection_name="rag_collection",
            embedding=HuggingFaceEmbeddings(model_name=self.sentence_transformer_model),
            connection_args={
                "uri": os.getenv("MILVUS_URI"),
                "token": os.getenv("MILVUS_TOKEN")
            },
        )
        retriver = vector_store.as_retriever()
        return retriver




