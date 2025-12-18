import os
import json
import time
import asyncio
import logging
from typing import Any, List, Dict, Optional

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from openai import OpenAI
from overrides import override
from qwen_agent.llm import get_chat_model
from datetime import datetime
import time

from memory.memory_system.utils import (
    _push_event,
    _drain_snapshot,
    _safe_dump_str,
    _multi_thread_run,
    setup_logger,
)
from inference.amem.memory_system import AgenticMemorySystem

log_filename = datetime.now().strftime("eval_%Y%m%d_%H%M%S.log")

class QwenAgentThinkHandlerWithAMem(OpenAICompletionsHandler):

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        """
        Note: Need to start vllm server first with command:
        vllm serve xxx \
            --served-model-name xxx \
            --port 8000 \
            --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
            --max-model-len 65536
        """
        
        self.llm = get_chat_model({
        'model': model_name,  # name of the model served by vllm server
        'model_type': 'oai',
        'model_server':'http://localhost:8014/v1', # can be replaced with server host
        'api_key': "none",
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'extra_body': {
                'chat_template_kwargs': {
                    'enable_thinking': True
                }
            },
            "thought_in_content": True,
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'max_input_tokens': 58000,
            'timeout': 1000,
            'max_tokens': 4096
        }
    })

        self.MEM_TAG = "PRIVATE_MEMORY:"
        self.memory_system = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
            llm_backend="openai",           # LLM backend (openai/ollama)
            llm_model=self.model_name       # LLM model name
        )
        self._cur_test_id = None

        self.logger_context = setup_logger("context", log_path=os.path.join("log/qwen3-4b-think/amem/context/", log_filename) ,level=logging.INFO)
        self.logger_memory = setup_logger("memory", log_path=os.path.join("log/qwen3-4b-think/amem/memory/", log_filename) ,level=logging.INFO)

    def _build_client_kwargs(self):
        kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if headers_env := os.getenv("OPENAI_DEFAULT_HEADERS"):
            kwargs["default_headers"] = json.loads(headers_env)
        return kwargs

    def _reset_if_new_case(self, test_id: str):
        if test_id != self._cur_test_id:
            # push the lateset turn context to slots
            self._cur_test_id = test_id
            self.memory_system = AgenticMemorySystem(
                model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
                llm_backend="openai",           # LLM backend (openai/ollama)
                llm_model=self.model_name       # LLM model name
            )

    def _strip_old_memory_block_keep_fc_items(self, message: list) -> list:
        out = []
        for m in message:
            if isinstance(m, dict):
                if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith(self.MEM_TAG):
                    continue
            out.append(m)
        return outs

    def _inject_memory(self, query_text: str, message: list[dict]):

        # Retrieve relevant memories
        relevant_memories = self.memory_system.search_agentic(query_text, k=3)
        memories_str = "\n".join(f"- {entry['content']}" for entry in relevant_memories)

        self.logger_memory.info(f"[Info] Retrieved Memories for Query '{query_text}': {memories_str}")

        mem_msg = {
            "role": "user",
            "content": (
                f"{self.MEM_TAG}\n"
                "Context note (optional): the following are retrieved memories. "
                "They may be irrelevant. Prioritize the current user request.\n\n"
                "Here are some relevant memories:\n"
                f"{memories_str}\n"
            ),
        }

        last_user_idx = None
        for i in range(len(message) - 1, -1, -1):
            if isinstance(message[i], dict) and message[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            # No user message found; just prepend
            return [mem_msg] + message

        return message[:last_user_idx] + [mem_msg] + message[last_user_idx:]

    #### FC methods ####
    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        # 1. remove old memory blocks but keep function call items
        message = self._strip_old_memory_block_keep_fc_items(message)
        # 2. extract latest user query text
        query_text = self._extract_latest_user_query_text_keep_fc_items(message)
        # 3. inject memory
        message = self._inject_memory(query_text=query_text, message=message)
        
        inference_data["message"] = message

        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        start_time = time.time()
        if len(tools) > 0:
            responses = None
            for resp in self.llm.quick_chat_oai(message, tools):
                responses = resp 
                
        else:
            responses = None
            for resp in self.llm.quick_chat_oai(message):
                responses = resp
        end_time = time.time()
        
        return responses, end_time-start_time

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        self._reset_if_new_case(test_entry["id"])
        inference_data["message"] = []
        return inference_data

    @override
    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        for m in first_turn_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    @override
    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        for m in user_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    @override
    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {"role": "tool", "tool_call_id": tool_call_id, "content": execution_result}
            inference_data["message"].append(tool_message)
            _push_event(self._event_buffer, "TOOL_RESULT", execution_result[:2000])
        return inference_data

    
    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        try:
            model_responses = [
                {func_call['function']['name']: func_call['function']['arguments']}
                for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
            tool_call_ids = [
                func_call['function']['name'] for func_call in api_response["choices"][0]["message"]["tool_calls"]
            ]
        except:
            model_responses = api_response["choices"][0]["message"]["content"]
            tool_call_ids = []
        
        response_data = {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response["choices"][0]["message"],
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.get("usage", {}).get("prompt_tokens", 0),
            "output_token": api_response.get("usage", {}).get("completion_tokens", 0),
        }
        return response_data
        

    @override
    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        
        if isinstance(model_response_data["model_responses_message_for_chat_history"], list):
            inference_data["message"]+=model_response_data["model_responses_message_for_chat_history"]
        else:
            inference_data["message"].append(
                model_response_data["model_responses_message_for_chat_history"]
            )
        memory = str(model_response_data["model_responses"])
        _push_event(self._event_buffer, "ASSISTANT", memory)

        if len(model_response_data.get("tool_call_ids", [])) == 0:
            # No tool calls means that the end of turn
            self._materialize_turn_slots()

        return inference_data

    def _flatten_user_content(self, content: Any) -> str:
        """Flatten user message content into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()

        # Some libs use list[{"type": "...", "text": "..."}]
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    # common keys: {"type":"text","text":...} / {"type":"input_text","text":...}
                    if "text" in item and isinstance(item["text"], str):
                        parts.append(item["text"])
                    # some variants might use {"type":"text","content":...}
                    elif "content" in item and isinstance(item["content"], str):
                        parts.append(item["content"])
            return "\n".join(p.strip() for p in parts if p and p.strip()).strip()

        # fallback (dict or others)
        if isinstance(content, dict):
            # rare: {"text": "..."} or {"content": "..."}
            for k in ("text", "content"):
                v = content.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""


    def _extract_latest_user_query_text_keep_fc_items(self, message: list) -> str:
        for m in reversed(message):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        return ""


    def _materialize_turn_slots(self):
        # transfer the latest turn snapshot to working slots
        snapshot_events = _drain_snapshot(event_buffer=self._event_buffer)
        if not snapshot_events:
            return

        self.memory_system.add_note(snapshot_events)