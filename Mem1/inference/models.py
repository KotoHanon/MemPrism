import os
import requests
import logging
import torch
from typing import Dict, List, Union, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import litellm
from amem.memory_system import AgenticMemorySystem
import abc
from litellm import completion
import os
import asyncio
from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.working_slot import WorkingSlot
from memory.memory_system.utils import (
    setup_logger,
    _safe_dump_str,
    new_id,
    now_iso,
)
from memory.memory_system.models import EpisodicRecord, SemanticRecord
from mem0 import Memory
from datetime import datetime
import os
from tqdm import tqdm
from uuid import uuid4
import random

log_filename = datetime.now().strftime("train_%Y%m%d_%H%M%S.log")
logger = setup_logger("retrieve_memory", log_path=os.path.join("inference/log/retrieve_memory/", log_filename) ,level=logging.INFO)


class BaseClient(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt, model="gpt-4o", temperature=0.01, force_json=False):
        pass

    def reset(self):
        pass
    
    @property
    def has_memory(self):
        return False


class VLLMOpenAIClient(BaseClient):
    def __init__(self):
        self.url = "http://localhost:8014"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    def generate_response(self, prompt, model="gpt-4o", temperature=0.01, force_json=False):
        try:
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "stop": ["</search>", "</answer>"]
                }
            )

            choice = response.json()['choices'][0]

            content = choice["message"]["content"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content

        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"

    def make_completion(self, initial_prompt, content, model="gpt-4o", temperature=0.01, force_json=False, is_last_turn=False):
        prompt_message = [{"role": "user", "content": initial_prompt}]
        prompt_message.append({"role": "assistant", "content": content})
        prompt_message = self.tokenizer.apply_chat_template(prompt_message, tokenize=False)

        # remove the <|im_end> at the end of the prompt
        prompt_message = prompt_message[:-len("<|im_end|>\n")]

        stop = []
        if is_last_turn:
            stop = ["</answer>"]
        else:
            stop = ["</search>", "</answer>"]

        try:
            response = requests.post(
                self.url + "/v1/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "prompt": prompt_message,
                    "stop": stop,
                    "top_p": 0.95,
                    "top_k": -1,
                    "max_tokens": 1024,
                }
            )

            choice = response.json()['choices'][0]

            content = choice["text"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}" 

class LiteLLMClient(BaseClient):
    def __init__(self):
        litellm.drop_params = True
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
        #assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"

    def generate_response(self, prompt, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        config = {
            "temperature": temperature,
            "top_p": 0.95,
            "provider": {
                "sort": "throughput"
            },
        }

        if model.startswith("openai/"):
            # drop provider
            config.pop("provider")

        '''if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model'''

        try: 
            # Format messages properly with content type
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if force_json:
                response = litellm.completion(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    **config
                )
            else:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **config
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"
    
    def make_completion(self, prompt, cur_obs, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        config = {
            "temperature": temperature,
            "top_p": 1,
            "provider": {
                "sort": "throughput"
            },
        }

        if model.startswith("openai/"):
            # drop provider
            config.pop("provider")

        '''if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model'''

        try: 
            # Format messages properly with content type
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": cur_obs}]}
            ]
            if force_json:
                response = litellm.completion(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    **config
                )
            else:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **config
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"


class AMemClient(BaseClient):
    def __init__(self, use_local_model: bool = False):
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
        #assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"
        litellm.drop_params = True
        # Initialize the memory system 🚀
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model="gpt-4o-mini"         # LLM model name
        )
        self.memories = []
        self.use_local_model = use_local_model
        if self.use_local_model:
            self.url = "http://localhost:8014"
            self.tokenizer = AutoTokenizer.from_pretrained("/hpc_stor03/sjtu_home/zijian.wang/MEM1/.cache/Qwen3-4B")
        else:
            assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
            #assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"
    
    def chat_with_memories(self, message: str, model: str, temperature: float = 0.01, force_json: bool = False, user_id: str = "default_user") -> str:
        # Retrieve relevant memories
        relevant_memories = self.memory_system.search_agentic(message, k=3)
        print("1st relevant_memories:", relevant_memories[0]['content'] if len(relevant_memories) > 0 else "None")
        memories_str = "\n".join(f"- {entry['content']}" for entry in relevant_memories)
        self.memories.append(memories_str)
        # Generate Assistant response
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

        config = {
            "temperature": temperature, 
            "top_p": 0.95,
            "provider": {
                "sort": "throughput"
            }
        }
        if force_json:
            config["response_format"] = {"type": "json_object"}
        
        if model.startswith("openai/"):
            config.pop("provider")
        
        '''if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model'''

        if self.use_local_model:
            # Use local vLLM server for inference
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    "stop": ["</search>", "</answer>"]
                }
            )

            choice = response.json()['choices'][0]

            content = choice["message"]["content"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content

        else:
            response = litellm.completion(model=model, messages=messages, **config)
            assistant_response = response.choices[0].message.content.strip()

            return assistant_response

    def generate_response(self, prompt, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        return self.chat_with_memories(prompt, model=model, temperature=temperature, force_json=force_json)

    @property
    def has_memory(self):
        return True
    
    def reset(self):
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model="gpt-4o-mini"         # LLM model name
        )
        self.memories = []


class Mem0Client(BaseClient):
    def __init__(self, use_local_model: bool = False, use_graph: bool = False):
        litellm.drop_params = True
        faiss_root = "/tmp/faiss_memories"
        kuzu_root = "/tmp/kuzu_graphs"
        os.makedirs(faiss_root, exist_ok=True)
        os.makedirs(kuzu_root, exist_ok=True)
        store_time = f"store_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        # Initialize the memory system 🚀
        self.config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.01,
                        "max_tokens": 2000,
                    }
                },
                "vector_store": {
                    "provider": "faiss",
                    "config": {
                        "collection_name": "test",
                        "path": os.path.join(faiss_root, store_time),
                        "distance_strategy": "euclidean"
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                },
                "memory": {
                    "auto_embed": True,
                    "summarization": True,
                },
            }
        if use_graph:
            # Add graph store configuration
            graph_store = {
                "provider": "kuzu",
                "config": {
                    "db": os.path.join(kuzu_root, store_time),
                },
            }
            self.config.update({"graph_store": graph_store})

        self.memory_system = Memory.from_config(self.config)
        self.memories = []
        self.use_local_model = use_local_model
        if self.use_local_model:
            self.url = "http://localhost:8014"
            self.tokenizer = AutoTokenizer.from_pretrained("/hpc_stor03/sjtu_home/zijian.wang/MEM1/.cache/Qwen3-4B")
        else:
            assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
            #assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"        

    def chat_with_memories(self, query_text: str, message: str, thread_name: str, model: str, temperature: float = 0.01, force_json: bool = False, user_id: str = "default_user") -> str:
        # Retrieve relevant memories
        try:
            returns = self.memory_system.search(message, user_id=f"agent_{thread_name}", limit=3)
            print("Raw returns from memory search:", returns)
            if isinstance(returns, list):
                relevant_memories = returns
            else:
                relevant_memories = returns['results']
        except Exception as e:
            print("[Error] Memory search failed:", repr(e))
            relevant_memories = []
        
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
        self.memories.append(memories_str)
        # Generate Assistant response
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

        config = {
            "temperature": temperature, 
            "top_p": 0.95,
            "provider": {
                "sort": "throughput"
            }
        }
        if force_json:
            config["response_format"] = {"type": "json_object"}
        
        if model.startswith("openai/"):
            config.pop("provider")
        
        '''if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model'''
        
        if self.use_local_model:
            # Use local vLLM server for inference
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    "stop": ["</search>", "</answer>"]
                }
            )

            choice = response.json()['choices'][0]

            content = choice["message"]["content"].strip()

            if choice["stop_reason"] == "</search>":
                content += "</search>"
            elif choice["stop_reason"] == "</answer>":
                content += "</answer>"

            return content

        else:
            response = litellm.completion(model=model, messages=messages, **config)
            assistant_response = response.choices[0].message.content.strip()

            return assistant_response

    def generate_response(self, query_text: str, prompt: str, thread_name: str, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        return self.chat_with_memories(query_text=query_text, message=prompt, thread_name=thread_name, model=model, temperature=temperature, force_json=force_json)

    @property
    def has_memory(self):
        return True

    def reset(self):
        self.memories = []


class AyumuClient(BaseClient):
    def __init__(self, model: str = "gpt-4o-mini", use_local_model: bool = False):
        litellm.drop_params = True
        # Initialize the memory system 🚀
        self.slot_process = SlotProcess()
        self.semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_name="gpt-4o-mini")
        self.episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_name="gpt-4o-mini")

        self.retrieved_working_slots = []
        self.semantic_memories = []
        self.episodic_memories = []

        self.use_local_model = use_local_model
        if self.use_local_model:
            self.url = "http://localhost:8014"
            self.tokenizer = AutoTokenizer.from_pretrained("/hpc_stor03/sjtu_home/zijian.wang/MEM1/.cache/Qwen3-4B")
            self.slot_process = SlotProcess(llm_name=model, llm_backend="vllm")
            self.semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_name=model, llm_backend="vllm")
            self.episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_name=model, llm_backend="vllm")
        else:
            assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY is not set"
            #assert "OPENROUTER_API_KEY" in os.environ, "OPENROUTER_API_KEY is not set"
            self.slot_process = SlotProcess(llm_name=model, llm_backend="openai")
            self.semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_name=model, llm_backend="openai")
            self.episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_name=model, llm_backend="openai")

    def chat_with_memories(self, query_text: str, slots: List[WorkingSlot], message: str, model: str, temperature: float = 0.01, force_json: bool = False, user_id: str = "default_user", threshold: float = 0.4) -> str:
        # Retrieve relevant memories
        slot_query_limit = min(5, self.slot_process.get_container_size())
        sem_query_limit = min(3, self.semantic_memory_system.size // 3)
        epi_query_limit = min(3, self.episodic_memory_system.size // 3)
        if len(query_text) > 0:
            #relevant_slots = self.slot_process.query(query_text=message, slots=slots, limit=slot_query_limit, key_words=query_text.split())
            relevant_slots = self.slot_process.query(query_text=message, slots=slots, limit=slot_query_limit, key_words=query_text.split(), use_svd=True, embed_func=self.semantic_memory_system.vector_store._embed)
            relevant_semantic_memories = self.semantic_memory_system.query(query_text=query_text, limit=sem_query_limit, threshold=threshold)
            relevant_episodic_memories = self.episodic_memory_system.query(query_text=query_text, limit=epi_query_limit)
            
        else:
            relevant_slots = []
            relevant_semantic_memories = []
            relevant_episodic_memories = []

        if len(relevant_slots) > 0:
            logger.info(f"Query Text: {message}, key_words: {query_text.split()}, 1st retrieved slot: {_safe_dump_str(relevant_slots[0][1])}")
        if len(relevant_semantic_memories) > 0:
            logger.info(f"Query Text: {query_text}, 1st retrieved semantic memory: {relevant_semantic_memories[0][1].summary}")
        if len(relevant_episodic_memories) > 0:
            logger.info(f"Query Text: {query_text}, 1st retrieved episodic memory: {relevant_episodic_memories[0][1].detail}")

        slots_str = "\n".join(f"- {_safe_dump_str(entry[1])}" for entry in relevant_slots)
        semantic_memories_str = "\n".join(f"- {entry[1].summary}" for entry in relevant_semantic_memories)
        episodic_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].detail)}" for entry in relevant_episodic_memories)
        self.retrieved_working_slots.append(slots_str)
        self.semantic_memories.append(semantic_memories_str)
        self.episodic_memories.append(episodic_memories_str)
        # Generate Assistant response
        system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Short Memories:\n{slots_str} \nUser Semantic Memories:\n{semantic_memories_str}. \nUser Episodic Memories:\n{episodic_memories_str}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]

        config = {
            "temperature": temperature, 
            "top_p": 0.95,
            "provider": {
                "sort": "throughput"
            }
        }
        if force_json:
            config["response_format"] = {"type": "json_object"}
        
        if model.startswith("openai/"):
            config.pop("provider")
        
        '''if not model.startswith("openrouter/") and not model.startswith("openai/"):
            model = "openrouter/" + model'''
        
        if self.use_local_model:
            # Use local vLLM server for inference
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    "stop": ["</search>", "</answer>"]
                }
            )

            try:
                data = response.json()
            except Exception as e:
                print("[vLLM ERROR] status:", response.status_code)
                print("[vLLM ERROR] raw text:", response.text[:1000])
                raise

            if "choices" not in data:
                print("[vLLM ERROR] no 'choices' in response, got:")
                print(data)
                raise RuntimeError(f"No 'choices' in response: {data}")

            choice = data["choices"][0]
            content = choice["message"]["content"].strip()

            if choice.get("stop_reason") == "</search>":
                content += "</search>"
            elif choice.get("stop_reason") == "</answer>":
                content += "</answer>"

            return content

        else:
            response = litellm.completion(model=model, messages=messages, **config)
            assistant_response = response.choices[0].message.content.strip()

            return assistant_response

    def generate_response(self, query_text, slots, prompt, model="openai/gpt-4o-mini", temperature=0.01, force_json=False):
        return self.chat_with_memories(query_text=query_text, slots=slots, message=prompt, model=model, temperature=temperature, force_json=force_json)

    async def transfer_context_to_slots(self, context: str, max_slots: int = 5) -> Optional[List[WorkingSlot]]:
        print("[Info] Transferring context to working slots...")

        try:
            working_slots = await self.slot_process.transfer_qa_agent_context_to_working_slots(context=context, max_slots=max_slots)
            print("after await, len of working_slots =", len(working_slots) if working_slots is not None else 0)
        except Exception as e:
            import traceback
            print("[ERROR] transfer_qa_agent_context_to_working_slots failed:", repr(e))
            traceback.print_exc()
            return

        if not working_slots:
            print("[Info] No working slots returned.")
            return
        
        for slot in working_slots:
            self.slot_process.add_slot(slot)

        return working_slots

    async def transfer_slots_to_memories(self, is_abstract: bool = False):
        if self.slot_process.get_container_size() == 0:
            return

        routed_slot_container = await self.slot_process.filter_and_route_slots()
        try:
            inputs = await self.slot_process.generate_long_term_memory(routed_slots=routed_slot_container)
        except Exception as e:
            import traceback
            print("[ERROR] generate_long_term_memory failed:", repr(e))
            traceback.print_exc()

        if not inputs or len(inputs) == 0:
            return

        semantic_records = []
        episodic_records = []

        for i in inputs:
            if i['memory_type'] == 'semantic':
                semantic_records.append(self.semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(self.semantic_memory_system.instantiate_epi_record(**i['input']))
        
        if is_abstract and len(episodic_records) > 0:
            await self.abstract_episodic_records_to_semantic_record(episodic_records)

        self.semantic_memory_system.add(semantic_records)
        self.episodic_memory_system.add(episodic_records)

    async def multi_thread_transfer_dicts_to_memories(self, is_abstract):
        semantic_records = []
        episodic_records = []

        for i in self.slot_process.memory_dict:
            if i['memory_type'] == 'semantic':
                semantic_records.append(self.semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(self.semantic_memory_system.instantiate_epi_record(**i['input']))
        
        if is_abstract and len(episodic_records) > 0:
            await self.abstract_episodic_records_to_semantic_record(episodic_records)

        if len(semantic_records) > 0:
            try:
                self.semantic_memory_system.upsert_normal_records(semantic_records)
            except Exception as e:
                import traceback
                print("[ERROR] upsert_normal_records for semantic_records failed:", repr(e))
                traceback.print_exc()
        if len(episodic_records) > 0:
            try:
                self.episodic_memory_system.upsert_normal_records(episodic_records)
            except Exception as e:
                import traceback
                print("[ERROR] upsert_normal_records for episodic_records failed:", repr(e))
                traceback.print_exc()

    async def load_inputs_to_memory_records(self, inputs: List[Dict[str, Any]]):
        if not inputs or len(inputs) == 0:
            return

        semantic_records = []
        episodic_records = []

        for i in tqdm(inputs):
            if i['memory_type'] == 'semantic':
                semantic_records.append(semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(semantic_memory_system.instantiate_epi_record(**i['input']))
        
        if is_abstract and len(episodic_records) > 0:
            await self.abstract_episodic_records_to_semantic_record(episodic_records)

        semantic_memory_system.add(semantic_records)
        episodic_memory_system.add(episodic_records)

        print(f"[Info] Number of semantic memories: {semantic_memory_system.size}")
        print(f"[Info] Number of episodic memories: {episodic_memory_system.size}")

    async def abstract_episodic_records_to_semantic_record(self, epi_records: List[EpisodicRecord], consistency_threshold: float = 0.8):
        try:
            abstract_result, cidmap2semrec = await self.episodic_memory_system.abstract_episodic_records(epi_records, consistency_threshold)
            print(f"[Info] Number of abstracted semantic records: {len(abstract_result)}")
            self.semantic_memory_system.upsert_abstract_semantic_records(abstract_result, cidmap2semrec)
        except Exception as e:
            import traceback
            print("[ERROR] abstract_episodic_records_to_semantic_record failed:", repr(e))
            traceback.print_exc()

    @property
    def has_memory(self):
        return True
    
    def reset(self):
        self.slot_process = SlotProcess()
        
        self.retrieved_working_slots = []
        self.semantic_memories = []
        self.episodic_memories = []