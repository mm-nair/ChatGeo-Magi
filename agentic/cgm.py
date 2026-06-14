"""
This script is responsible for orchestrating the full agentic chain. 
This includes loading documents into the RAG database, handling memory, 
setting up the ReAct agent, and providing inference functions.
"""

from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from opik.integrations.langchain import OpikTracer

from tqdm import tqdm
from IPython.display import Image, display

from apis.noaa_apis import (
    get_coordinates, plot, noaa_mag_api,  plot_many, contour_map,
)

from config import SystemConfig
from prompts import system_prompt
import os

class ChatGeoMagi:
    def __init__(self, config: SystemConfig):
        self.config = config

        self.llm = None
        self.embeddings = None
        self.vector_store = None

        self.memory = None
        self.retriever = None
        self.agent = None

    @staticmethod
    def setup_vector_store(config, embeddings):
        print("Loading vector store")
        if not os.path.exists(config.chroma_db_path):
            vector_store = Chroma(
                collection_name="cgmdocs",
                embedding_function=embeddings,
                persist_directory=config.chroma_db_path,
            )

            loaders = [
                DirectoryLoader(config.data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
                DirectoryLoader(config.data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
            ]

            loader_all = MergedDataLoader(loaders=loaders) 
            docs = loader_all.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True,
            )

            splits = splitter.split_documents(docs)

            for doc in tqdm(splits, desc="Adding documents to the vector store"):
                vector_store.add_documents(documents=[doc])

            print(f"Loaded {len(splits)} document chunks")
        else:
            vector_store = Chroma(
                collection_name="cgmdocs",
                embedding_function=embeddings,
                persist_directory=config.chroma_db_path,
            )
        
        vector_retriever = vector_store.as_retriever()
        return create_retriever_tool(
            vector_retriever,
            "retrieve_geomag_docs",
            "Search and return information about geomagnetism, from authoritative documents like the World Magnetic Model technical report, and Q&A emails.",
        )
    
    @staticmethod
    def get_shared_resources(config: SystemConfig):
        embeddings = HuggingFaceEmbeddings(model=config.embedding_model)
        llm = ChatOllama(model=config.model_name)
        retriever = ChatGeoMagi.setup_vector_store(config, embeddings)

        return llm, embeddings, retriever

    def initialize(self, llm, embeddings, retriever):
        self.llm = llm
        self.embeddings = embeddings
        self.retriever = retriever
        self.memory = MemorySaver()

        tools = [
            self.retriever,
            noaa_mag_api,
            plot,
            plot_many,
            contour_map,
        ]

        self.agent = create_react_agent(self.llm, tools, checkpointer=self.memory, prompt=SystemMessage(content=system_prompt))

    def invoke(self, question: str, thread_id: str = "1"):
        message = {"role": "user", "content": question}

        return self.agent.invoke(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}}
        )
    
    def stream_invoke(self, question: str, thread_id: str = "1"):
        message = {"role": "user", "content": question}

        return self.agent.stream(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values"
        )

    def traced_invoke(self, question: str, thread_id: str = "1"):
        message = {"role": "user", "content": question}
        tracer = OpikTracer(graph=self.agent.get_graph(xray=True))

        return self.agent.invoke(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}, "callbacks": [tracer]},
        )

    def visualize_agent(self, filepath="agent_graph.png"):
        self.agent.get_graph(xray=True).draw_mermaid_png(
            output_file_path=filepath,
        )
        
        print(self.agent.get_graph(xray=True).draw_ascii())
        print(f"Agent graph saved to {filepath}")