"""
LEGACY: Many portions of this script may be broken now,
and is solely to demonstrate how the two-LLM architecture was
constrcuted. No plots nor data were made from this architecture.

This script has the creation and orchestration of the two-LLM system.
"""

from typing import Tuple, Optional, Any
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    PromptTemplate,
    Settings,
    global_handler,
    set_global_handler,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.base.response.schema import Response as LlamaResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer

from config import TOOLS, SystemConfig
from printer import cprint
from dataclasses import dataclass

from typing import List, Dict
from opik import opik_context, Attachment, track

import chromadb
import os
import prompts
import matplotlib
import random
import opik
import string
import time

from dotenv import dotenv_values
env = dotenv_values(".env")

@dataclass
class TracedResponse:
    """
    A comprehensive response tracking class that captures all aspects of the query processing.
    
    Attributes:
    - original_prompt: The initial user prompt
    - llama_response: LlamaIndex's core response object
    - api_response: Raw API recommendation response
    - tool_call: Optional tool call information
    - tool_output: Output from tool execution
    - figure_output: Optional matplotlib figure
    - context_str: Retrieved context string
    - appendix: Additional context or metadata
    - error: Any error that occurred during processing
    - sources: List of sources used in the response
    - urls_used: List of URLs used in the response
    """
    original_prompt: str
    llama_response: Optional[LlamaResponse] = None
    api_response: Optional[str] = None
    tool_call: Optional[str] = None
    tool_output: Optional[Any] = None   
    figure_output: Optional[Any] = None
    context_strs: Optional[List[str]] = None
    appendix: Optional[str] = None
    error: Optional[Exception] = None
    sources: Optional[List[Dict[str, str]]] = None
    urls_used: Optional[List[str]] = None
    trace_id: Optional[str] = None

class ChatGeoMagiEngine:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.api_c_engine = None
        self.geomagi_c_engine = None
        self.index = None
        self.api_memory = None
        self.geomagi_memory = None
        
        self.opik_client = opik.Opik()
    
    @staticmethod
    def get_shared_resources(config: SystemConfig):
        prompts.generate_prompts_from_tools(TOOLS)

        set_global_handler("opik")
        opik_callback_handler = global_handler

        llm = Ollama(model=config.model_name, request_timeout=config.request_timeout)
        embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)

        Settings.llm = llm
        Settings.embed_model = embed_model

        reader = SimpleDirectoryReader(input_dir=config.data_dir, recursive=True)
        documents = reader.load_data()

        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?",
        )

        Settings.text_splitter = splitter

        existing_db = os.path.exists(config.chroma_db_dir) and os.listdir(config.chroma_db_dir)

        chroma_client = chromadb.PersistentClient(path=config.chroma_db_dir)
        chroma_collection = chroma_client.get_or_create_collection("llama_vec")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if existing_db:
            print("Loading existing vector store...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        else:
            print("Creating vector store (this may take a while)...")
            index = VectorStoreIndex.from_documents(
                documents,
                transformations=[splitter],
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )

        return llm, index
    
    def initialize_session(self, llm, index):
        self.api_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        self.geomagi_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

        api_template = PromptTemplate(prompts.api_prompt)
        geomagi_template = PromptTemplate(prompts.geomagi_prompt)

        self.api_c_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self.api_memory,
            llm=llm,
            context_prompt=api_template,
            verbose=False,
        )

        self.geomagi_c_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self.geomagi_memory,
            llm=llm,
            context_prompt=geomagi_template,
            verbose=False,
        )

    @track(name="ChatGeo-Magi", capture_input=True, capture_output=True)
    def invoke(self, prompt: str, silent=False) -> TracedResponse:
        """Main processing method that returns a TracedResponse with comprehensive tracking."""
        start_time = time.time()
        
        if not self.api_c_engine or not self.geomagi_c_engine:
            raise RuntimeError("Engines not initialized. Call initialize() first.")

        opik_context.update_current_trace(
            name="ChatGeo-Magi",
            input={"user_prompt": prompt, "config": {"model": self.config.model_name}},
            metadata={
                "session_id": id(self),
                "timestamp": time.time()
            },
        )

        traced_response = TracedResponse(original_prompt=prompt)
        traced_response.trace_id = opik_context.get_current_trace_data().id

        try:
            api_response, tool_name, tool_args = self._get_api_recommendation(prompt, silent)
            traced_response.api_response = api_response

            if not tool_name:
                result = self._handle_no_api(traced_response)
                self._finalize_trace(result, start_time, "no_api_path")
                return result

            tool_output, figure_output, execution_success, urls_used = self._execute_tool(tool_name, tool_args, silent)
            
            if not execution_success:
                result = self._handle_failed_api(traced_response)
                self._finalize_trace(result, start_time, "failed_api_path")
                return result

            result = self._generate_final_response(
                traced_response, 
                api_response, 
                tool_output, 
                figure_output, 
                tool_name, 
                urls_used,
            )
            
            self._finalize_trace(result, start_time, "successful_path")
            return result

        except Exception as e:
            traced_response.error = e
            cprint(f"Invoke Error: {e}", color="red")
            
            opik_context.update_current_trace(
                output={"error": str(e), "error_type": type(e).__name__},
                metadata={"execution_path": "error", "error_details": str(e)}
            )
            
            self._finalize_trace(traced_response, start_time, "error_path")
            return traced_response

    def _finalize_trace(self, response: TracedResponse, start_time: float, execution_path: str):
        """Finalize the trace with comprehensive output and metadata."""
        execution_time = time.time() - start_time
        
        output_data = {
            "response_text": str(response.llama_response) if response.llama_response else None,
            "tool_used": response.tool_call,
            "has_figure": response.figure_output is not None,
            "context_retrieved": len(response.context_strs) if response.context_strs else 0,
            "urls_used": response.urls_used,
            "execution_time_seconds": execution_time,
            "execution_path": execution_path
        }
        
        if response.error:
            output_data["error"] = str(response.error)
            output_data["error_type"] = type(response.error).__name__
        
        opik_context.update_current_trace(
            output=output_data,
            metadata={
                "execution_path": execution_path,
                "execution_time_seconds": execution_time,
                "tool_execution_success": response.tool_call is not None,
                "context_sources_count": len(response.context_strs) if response.context_strs else 0,
                "has_visual_output": response.figure_output is not None
            }
        )

    @track(name="API Recommendation Step", capture_input=True, capture_output=True)
    def _get_api_recommendation(self, prompt: str, silent=False) -> Tuple[str, Optional[str], Tuple]:
        """Enhanced API recommendation with comprehensive step-level tracking."""
        step_start_time = time.time()
        
        try:
            opik_context.update_current_span(
                input={"user_prompt": prompt},
                metadata={"step_type": "api_recommendation", "engine": "api_c_engine"}
            )
            
            api_response = str(self.api_c_engine.chat(prompt))
            cprint(api_response, color="green", silent=silent)
            
            if "N/A" in api_response:
                opik_context.update_current_span(
                    output={
                        "api_response": api_response,
                        "tool_selected": None,
                        "recommendation_type": "no_api_needed"
                    },
                    metadata={
                        "step_success": True,
                        "execution_time_seconds": time.time() - step_start_time
                    }
                )
                return api_response, None, ()
            
            parsed_response = self.parse_tool_response(api_response)
            
            if not isinstance(parsed_response, tuple) or len(parsed_response) < 2:
                raise ValueError("Invalid API response format")
            
            tool_name, tool_args = parsed_response[0], parsed_response[1:]
            
            opik_context.update_current_span(
                output={
                    "api_response": api_response,
                    "tool_selected": tool_name,
                    "tool_args": list(tool_args),
                    "recommendation_type": "tool_selected"
                },
                metadata={
                    "step_success": True,
                    "tool_name": tool_name,
                    "args_count": len(tool_args),
                    "execution_time_seconds": time.time() - step_start_time
                }
            )
            
            return api_response, tool_name, tool_args
        
        except (ValueError, SyntaxError) as e:
            cprint(e, color="red", silent=silent)
            
            opik_context.update_current_span(
                output={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "raw_response": api_response if 'api_response' in locals() else None
                },
                metadata={
                    "step_success": False,
                    "error_category": "parsing_error",
                    "execution_time_seconds": time.time() - step_start_time
                }
            )
            
            return str(e), None, ()

    @staticmethod
    def parse_tool_response(response: str) -> Tuple[str, ...]:
        """Parse the API response into a tuple of tool name and arguments."""
        stripped_response = response.strip().strip("()")
        
        parts = []
        current_part = ""
        in_quotes = False
        for char in stripped_response:
            if char in ('"', "'"):
                in_quotes = not in_quotes
                continue
            elif char == ',' and not in_quotes:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""
                continue
            elif char == ' ' and not in_quotes and not current_part:
                continue
            current_part += char
        if current_part:
            parts.append(current_part.strip())
        
        parsed_tuple = tuple(part.strip('"').strip("'").strip("()") for part in parts)
        return parsed_tuple
    
    @track(name="Tool Execution Step", capture_input=True, capture_output=True)
    def _execute_tool(self, tool_name: str, tool_args: Tuple, silent=False) -> Tuple[Any, Any, bool, Optional[List[str]]]:
        """Execute the specified tool with comprehensive execution tracking."""
        step_start_time = time.time()
        
        opik_context.update_current_span(
            input={
                "tool_name": tool_name,
                "tool_args": list(tool_args),
                "available_tools": list(TOOLS.keys())
            },
            metadata={
                "step_type": "tool_execution",
                "tool_name": tool_name,
                "args_count": len(tool_args)
            }
        )
        
        tool_output = None
        figure_output = None
        urls_used = None
        execution_success = False
        
        for tool in TOOLS.values():
            if tool_name == tool["name"]:
                try:
                    result = tool["fn"](*tool_args)

                    if not isinstance(result, tuple) or len(result) != 2:
                        error_msg = f"Tool `{tool_name}` returned {len(result) if isinstance(result, tuple) else 'a non-tuple'}; expected exactly 2 values"
                        print(result)
                        raise RuntimeError(error_msg)
                    
                    tool_output, urls = result

                    output_type = "figure" if isinstance(tool_output, matplotlib.figure.Figure) else "data"
                    
                    if isinstance(tool_output, matplotlib.figure.Figure):
                        cprint("Created a matplotlib figure", color="green", silent=silent)
                        figure_output = tool_output
                    else:
                        cprint(tool_output, color="green", silent=silent)

                    if urls:
                        urls_used = urls
                    
                    execution_success = True
                    
                    opik_context.update_current_span(
                        output={
                            "execution_success": True,
                            "output_type": output_type,
                            "urls_used": urls_used,
                            "output_preview": str(tool_output)[:500] if not isinstance(tool_output, matplotlib.figure.Figure) else "matplotlib_figure"
                        },
                        metadata={
                            "step_success": True,
                            "tool_found": True,
                            "output_type": output_type,
                            "urls_count": len(urls_used) if urls_used else 0,
                            "execution_time_seconds": time.time() - step_start_time
                        }
                    )
                    
                    break
                    
                except Exception as e:
                    cprint(e, color="red", silent=silent)
                    
                    opik_context.update_current_span(
                        output={
                            "execution_success": False,
                            "error": str(e),
                            "error_type": type(e).__name__
                        },
                        metadata={
                            "step_success": False,
                            "tool_found": True,
                            "error_category": "tool_execution_error",
                            "execution_time_seconds": time.time() - step_start_time
                        }
                    )
                    
                    break
        
        if not execution_success and tool_name not in [tool["name"] for tool in TOOLS.values()]:
            opik_context.update_current_span(
                output={
                    "execution_success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": [tool["name"] for tool in TOOLS.values()]
                },
                metadata={
                    "step_success": False,
                    "tool_found": False,
                    "error_category": "tool_not_found",
                    "execution_time_seconds": time.time() - step_start_time
                }
            )
        
        return tool_output, figure_output, execution_success, urls_used

    @track(name="Fallback Response - No API", capture_input=True, capture_output=True)
    def _handle_no_api(self, traced_response: TracedResponse) -> TracedResponse:
        """Handle case where no suitable API was found with step tracking."""
        error_prompt = traced_response.original_prompt
        
        opik_context.update_current_span(
            input={"fallback_reason": "no_api_found", "original_prompt": error_prompt},
            metadata={"step_type": "fallback_response", "fallback_reason": "no_api"}
        )
        
        llama_response, appendix = self._query_cgm(error_prompt)

        traced_response.context_strs = [node.dict()["node"]["text"] for node in llama_response.source_nodes]
        traced_response.llama_response = llama_response
        traced_response.appendix = appendix
        traced_response.tool_call = None
        
        opik_context.update_current_span(
            output={
                "response_generated": True,
                "context_sources": len(traced_response.context_strs),
                "response_preview": str(llama_response)[:200]
            },
            metadata={
                "step_success": True,
                "response_type": "fallback_no_api",
                "context_sources_count": len(traced_response.context_strs)
            }
        )
        
        return traced_response
    
    @track(name="Fallback Response - Failed API", capture_input=True, capture_output=True)
    def _handle_failed_api(self, traced_response: TracedResponse) -> TracedResponse:
        """Handle case where API execution failed with step tracking."""
        error_prompt = traced_response.original_prompt + " --- API failed/is down! Mention this to the user!"
        
        opik_context.update_current_span(
            input={"fallback_reason": "api_failed", "modified_prompt": error_prompt},
            metadata={"step_type": "fallback_response", "fallback_reason": "api_failed"}
        )
        
        llama_response, appendix = self._query_cgm(error_prompt)
        
        traced_response.context_strs = [node.dict()["node"]["text"] for node in llama_response.source_nodes]
        traced_response.llama_response = llama_response
        traced_response.appendix = appendix
        traced_response.tool_call = None
        
        opik_context.update_current_span(
            output={
                "response_generated": True,
                "context_sources": len(traced_response.context_strs),
                "response_preview": str(llama_response)[:200],
                "api_failure_acknowledged": True
            },
            metadata={
                "step_success": True,
                "response_type": "fallback_failed_api",
                "context_sources_count": len(traced_response.context_strs)
            }
        )
        
        return traced_response

    @track(name="Final Response Generation", capture_input=True, capture_output=True)
    def _generate_final_response(
        self, 
        traced_response: TracedResponse, 
        api_response: str, 
        tool_output: Any, 
        figure_output: Any, 
        tool_name: str, 
        urls_used: Optional[List[str]] = None
    ) -> TracedResponse:
        """Generate the final response with comprehensive tracking."""

        traced_response.tool_call = tool_name
        
        if figure_output:
            prompt_with_context = (
                traced_response.original_prompt + 
                " --- Generated a figure from NOAA APIs, inform the user to look at the plot above, it will auto render"
            )
            traced_response.figure_output = figure_output
            
            random_hash = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            file_path = f"/tmp/{random_hash}_plt.png"
            figure_output.savefig(file_path)

            opik_context.update_current_trace(
                attachments=[
                    Attachment(
                        data=file_path,
                        content_type="image/png",
                    )
                ]
            )
            
            response_type = "figure_generated"
            
        else:
            updated_api_response = f"{api_response} = {tool_output}"
            prompt_with_context = traced_response.original_prompt + f" --- {updated_api_response}"
            traced_response.tool_output = tool_output
            response_type = "data_generated"

        opik_context.update_current_span(
            input={
                "tool_name": tool_name,
                "tool_output_type": "figure" if figure_output else "data",
                "context_prompt": prompt_with_context,
                "urls_used": urls_used
            },
            metadata={
                "step_type": "final_response_generation",
                "response_type": response_type,
                "tool_name": tool_name,
                "has_figure": figure_output is not None,
                "urls_count": len(urls_used) if urls_used else 0
            }
        )

        llama_response, appendix = self._query_cgm(
            prompt_with_context, 
        )

        traced_response.urls_used = urls_used
        traced_response.context_strs = [node.dict()["node"]["text"] for node in llama_response.source_nodes]
        traced_response.llama_response = llama_response
        traced_response.appendix = appendix

        opik_context.update_current_span(
            output={
                "response_generated": True,
                "context_sources": len(traced_response.context_strs),
                "response_preview": str(llama_response)[:200],
                "urls_used": urls_used,
                "tool_used": tool_name
            },
            metadata={
                "step_success": True,
                "context_sources_count": len(traced_response.context_strs),
                "final_response_type": response_type
            }
        )

        return traced_response

    @track(name="ChatGeo-Magi Query Step", capture_input=True, capture_output=True)
    def _query_cgm(
        self, 
        prompt: str, 
    ) -> Tuple[LlamaResponse, Optional[str], str]:
        """Query the ChatGeo-Magi engine with comprehensive step tracking.""" 
        
        step_start_time = time.time()
        
        opik_context.update_current_span(
            input={"query_prompt": prompt},
            metadata={
                "step_type": "llm_query",
                "engine": "geomagi_c_engine",
                "memory_token_limit": 3900
            }
        )

        response = self.geomagi_c_engine.chat(prompt)
        
        context_count = len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
        
        opik_context.update_current_span(
            output={
                "response_text": str(response),
                "context_sources_used": context_count,
                "response_length": len(str(response))
            },
            metadata={
                "step_success": True,
                "context_sources_count": context_count,
                "response_length": len(str(response)),
                "execution_time_seconds": time.time() - step_start_time,
            }
        )

        return response, prompt
            
    def create_evaluation_dataset(self, interactions: List[Dict], dataset_name: str = "ChatGeo-Magi-Evaluation"):
        """Create a dataset for evaluation purposes"""
        try:
            dataset = self.opik_client.get_or_create_dataset(name=dataset_name)
            
            formatted_items = []
            for interaction in interactions:
                formatted_items.append({
                    "input": interaction.get("prompt", ""),
                    "expected_output": interaction.get("expected_response", ""),
                    "expected_tool": interaction.get("expected_tool", ""),
                    "metadata": interaction.get("metadata", {})
                })
            
            dataset.insert(formatted_items)
            cprint(f"Created evaluation dataset '{dataset_name}' with {len(formatted_items)} items", color="green")
            return dataset
        except Exception as e:
            cprint(f"Error creating evaluation dataset: {e}", color="red")
            return None