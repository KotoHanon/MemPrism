# bfcl_eval/model_handler/api_inference/openai_response_with_memory.py
import json
import os
import time
import asyncio

from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from openai.types.responses import Response
from typing import List, Any, Optional

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.working_slot import WorkingSlot
from memory.memory_system.utils import (
    _push_event,
    _drain_snapshot,
    _safe_dump_str,
)


class OpenAIResponsesHandlerWithMemory(BaseHandler):
    MEM_TAG = "PRIVATE_MEMORY:"

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.model_name = model_name
        self.client = OpenAI(**self._build_client_kwargs())

        self.slot_process = SlotProcess(llm_name=model_name, llm_backend="openai")
        self._event_buffer: List[str] = []
        self.slots = []
        self.semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_name=model_name, llm_backend="openai")
        self.episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_name=model_name, llm_backend="openai")
        self.procedural_memory_system = FAISSMemorySystem(memory_type="procedural", llm_name=model_name, llm_backend="openai")
        self._cur_test_id = None


    def _build_client_kwargs(self):
        kwargs = {}
        if api_key := os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = api_key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if headers_env := os.getenv("OPENAI_DEFAULT_HEADERS"):
            kwargs["default_headers"] = json.loads(headers_env)
        return kwargs

    @staticmethod
    def _substitute_prompt_role(prompts: list[dict]) -> list[dict]:
        for prompt in prompts:
            if prompt.get("role") == "system":
                prompt["role"] = "developer"
        return prompts

    def _reset_if_new_case(self, test_id: str):
        if test_id != self._cur_test_id:
            # push the lateset turn context to slots
            snapshot_events = _drain_snapshot(event_buffer=self._event_buffer)
            if len(snapshot_events) > 0:
                new_slots = asyncio.run(self.slot_process.transfer_fc_agent_context_to_working_slots(context=snapshot_events, max_slots=10))
                for slot in new_slots:
                    self.slot_process.add_slot(slot)
                self.slots.extend(new_slots)
            # reset short-term meory
            self.slot_process = SlotProcess(llm_name=self.model_name, llm_backend="openai")
            self._cur_test_id = test_id
            print(f"[Info] Size of slots before reset: {len(self.slots)}")
            if len(self.slots) >= 10:
                # Transfer existing slots to long-term memory
                asyncio.run(self.transfer_slots_to_memories(self.slots, is_abstract=False))
                self.slots = [] # Reset

    def _strip_old_memory_block(self, message: list[dict]) -> list[dict]:
        out = []
        for m in message:
            if m.get("role") == "developer" and isinstance(m.get("content"), str) and m["content"].startswith(self.MEM_TAG):
                continue
            out.append(m)
        return out

    def _inject_memory(self, query_text: str, message: list[dict], threshold: float = 0.4):
        '''slot_query_limit = min(5, self.slot_process.get_container_size())
        sem_query_limit = min(3, self.semantic_memory_system.size // 3)
        epi_query_limit = min(3, self.episodic_memory_system.size // 3)
        proc_query_limit = min(3, self.procedural_memory_system.size // 3)'''

        slot_query_limit = 5
        sem_query_limit = min(3, self.semantic_memory_system.size // 3)
        epi_query_limit = min(3, self.episodic_memory_system.size // 3)
        proc_query_limit = min(3, self.procedural_memory_system.size // 3)

        if len(query_text) > 0:
            relevant_slots = self.slot_process.query(query_text=_safe_dump_str(message), slots=self.slots, limit=slot_query_limit, key_words=query_text.split(), use_svd=False, embed_func=self.semantic_memory_system.vector_store._embed)
            relevant_semantic_memories = self.semantic_memory_system.query(query_text=query_text, limit=sem_query_limit, threshold=threshold)
            relevant_episodic_memories = self.episodic_memory_system.query(query_text=query_text, limit=epi_query_limit)
            relevant_procedural_memories = self.procedural_memory_system.query(query_text=query_text, limit=proc_query_limit, threshold=threshold)
            
        else:
            relevant_slots = []
            relevant_semantic_memories = []
            relevant_episodic_memories = []
            relevant_procedural_memories = []
        
        if len(relevant_slots) > 0:
            print(f"[Info] Size of Retrieved Slots: {len(relevant_slots)}")
            print(f"[Info] 1st Retrieved Slot: {_safe_dump_str(relevant_slots[0])}")

        slots_str = "\n".join(f"- {_safe_dump_str(entry[1])}" for entry in relevant_slots)
        semantic_memories_str = "\n".join(f"- {entry[1].summary}" for entry in relevant_semantic_memories)
        episodic_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].detail)}" for entry in relevant_episodic_memories)
        procedural_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].steps)}" for entry in relevant_procedural_memories)

        mem_msg = {
            "role": "developer",
            "content": (
                f"{self.MEM_TAG}\n"
                "Below is the agent's private memory from earlier turns in THIS test case.\n"
                "Use it only if helpful; do not mention it explicitly.\n\n"
                "Here are some relevant working slots:\n"
                f"{slots_str}\n"
                "Here are some relevant semantic memories:\n"
                f"{semantic_memories_str}\n"
                "Here are some relevant episodic memories:\n"
                f"{episodic_memories_str}\n"
                "Here are some relevant procedural memories:\n"
                f"{procedural_memories_str}\n"
            ),
        }
        return [mem_msg] + message

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        message = self._strip_old_memory_block(message)
        query_text = self._extract_latest_user_query_text(message)
        message = self._inject_memory(query_text=query_text, message=message)
        inference_data["message"] = message

        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        kwargs = {
            "input": message,
            "model": self.model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }
        if ("o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name):
            del kwargs["temperature"]
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        if len(tools) > 0:
            kwargs["tools"] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        self._reset_if_new_case(test_entry["id"])

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(test_entry["question"][round_idx])

        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data["tools"] = tools
        return inference_data

    def _parse_query_response_FC(self, api_response: Response) -> dict:
        model_responses = []
        tool_call_ids = []
        for func_call in api_response.output:
            if func_call.type == "function_call":
                model_responses.append({func_call.name: func_call.arguments})
                tool_call_ids.append(func_call.call_id)
        if not model_responses:
            model_responses = api_response.output_text

        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response.output,
            "tool_call_ids": tool_call_ids,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        for m in first_turn_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        for m in user_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].extend(model_response_data["model_responses_message_for_chat_history"])
        memory = str(model_response_data["model_responses"])[:2000]
        _push_event(self._event_buffer, "ASSISTANT", memory)
        return inference_data

    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {"type": "function_call_output", "call_id": tool_call_id, "output": execution_result}
            inference_data["message"].append(tool_message)
            _push_event(self._event_buffer, "TOOL_RESULT", execution_result[:2000])
        return inference_data

    def _query_prompting(self, inference_data: dict):
        msg = inference_data["message"]
        msg = self._strip_old_memory_block(msg)
        query_text = self._extract_latest_user_query_text(msg)
        print(f"[Debug] Query Text for Memory Injection: {query_text}")
        msg = self._inject_memory(query_text=query_text, message=msg)
        inference_data["message"] = msg

        inference_data["inference_input_log"] = {"message": repr(msg)}

        kwargs = {
            "input": msg,
            "model": self.model_name,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"summary": "auto"},
            "temperature": self.temperature,
        }
        if ("o3" in self.model_name or "o4-mini" in self.model_name or "gpt-5" in self.model_name):
            del kwargs["temperature"]
        else:
            del kwargs["reasoning"]
            del kwargs["include"]

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        self._reset_if_new_case(test_entry["id"])

        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = self._substitute_prompt_role(test_entry["question"][round_idx])

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Response) -> dict:
        reasoning_content = ""
        for item in api_response.output:
            if item.type == "reasoning":
                for summary in item.summary:
                    reasoning_content += summary.text + "\n"

        return {
            "model_responses": api_response.output_text,
            "model_responses_message_for_chat_history": api_response.output,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.input_tokens,
            "output_token": api_response.usage.output_tokens,
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        for m in first_turn_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        for m in user_message:
            if m.get("role") == "user":
                memory = str(m.get("content", ""))
                _push_event(self._event_buffer, "USER", memory)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].extend(model_response_data["model_responses_message_for_chat_history"])
        memory = str(model_response_data["model_responses"])[:2000]
        _push_event(self._event_buffer, "ASSISTANT", memory)
        return inference_data

    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        formatted_results_message = format_execution_results_prompting(inference_data, execution_results, model_response_data)
        inference_data["message"].append({"role": "user", "content": formatted_results_message})
        _push_event(self._event_buffer, "TOOL_RESULT", formatted_results_message[:2000])
        return inference_data

    from typing import Any

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


    def _extract_latest_user_query_text(self, message: list[dict]) -> str:
        """Get the latest user turn text as query_text for memory retrieval."""
        for m in reversed(message):
            if m.get("role") != "user":
                continue
            return self._flatten_user_content(m.get("content", ""))
        return ""

    def _materialize_step_slots(self, max_slots: int = 3):
        snapshot_events = _drain_snapshot(event_buffer=self._event_buffer)
        if not snapshot_events:
            return

        new_slots = asyncio.run(
            self.slot_process.transfer_fc_agent_context_to_working_slots(
                context=snapshot_events,
                max_slots=max_slots,
            )
        )
        for slot in new_slots:
            self.slot_process.add_slot(slot)
        self.slots.extend(new_slots)

    async def transfer_slots_to_memories(self, slots: List[WorkingSlot], is_abstract: bool = False):
        if len(slots) == 0:
            return

        routed_slot_container = await self.slot_process.filter_and_route_slots(slots=slots)
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
        procedural_records = []

        for i in inputs:
            if i['memory_type'] == 'semantic':
                semantic_records.append(self.semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(self.episodic_memory_system.instantiate_epi_record(**i['input']))
            elif i['memory_type'] == 'procedural':
                procedural_records.append(self.procedural_memory_system.instantiate_proc_record(**i['input']))
        
        if is_abstract and len(episodic_records) > 0:
            await self.abstract_episodic_records_to_semantic_record(episodic_records)

        self.semantic_memory_system.add(semantic_records)
        self.episodic_memory_system.add(episodic_records)
        self.procedural_memory_system.add(procedural_records)
