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
        #demo url about computer the realone will get from inputs like web_link or pdf or model suggestion or etc..
        self.urls = [
            "https://en.wikipedia.org/wiki/Computer",
            "https://en.wikipedia.org/wiki/Laptop",
            "https://en.wikipedia.org/wiki/Desktop",
            "https://en.wikipedia.org/wiki/Tablet"
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

        # Split documents while preserving metadata
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
        # Configure retriever to return formatted results
        retriver = vector_store.as_retriever(
            search_kwargs={
                "k": 3,  # Number of documents to retrieve
                "score_threshold": 0.5  # Minimum similarity score
            }
        )
        return retriver




