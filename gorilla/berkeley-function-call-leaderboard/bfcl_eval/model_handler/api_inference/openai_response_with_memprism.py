# bfcl_eval/model_handler/api_inference/openai_response_with_memory.py
import json
import os
import time
import asyncio
import logging

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
from datetime import datetime
from collections import OrderedDict

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.models import EpisodicRecord, SemanticRecord, ProceduralRecord
from memory.memory_system.working_slot import WorkingSlot
from memory.memory_system.utils import (
    _push_event,
    _drain_snapshot,
    _safe_dump_str,
    _multi_thread_run,
    setup_logger,
)

log_filename = datetime.now().strftime("eval_%Y%m%d_%H%M%S.log")


class OpenAIResponsesHandlerWithMemory(BaseHandler):

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_RESPONSES
        self.model_name = model_name
        self.client = OpenAI(**self._build_client_kwargs())
        self.MEM_TAG = "PRIVATE_MEMORY:"

        self.slot_process = SlotProcess(llm_name=model_name, llm_backend="openai")
        self._event_buffer: List[str] = []
        self.slots = []
        self.semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_name=model_name, llm_backend="openai")
        self.episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_name=model_name, llm_backend="openai")
        self.procedural_memory_system = FAISSMemorySystem(memory_type="procedural", llm_name=model_name, llm_backend="openai")
        self._cur_test_id = None
        self._turn_injected_once = False # Only inject memory once per turn, at the beginning.

        self._last_tool_sig: str | None = None
        self._same_tool_streak: int = 0
        self._trigger_k: int = 3
        self._inject_memory_next: bool = False       

        self.logger_context = setup_logger("context", log_path=os.path.join("log/gpt-4o-mini/memprism/context/", log_filename) ,level=logging.INFO)
        self.logger_memory = setup_logger("memory", log_path=os.path.join("log/gpt-4o-mini/memprism/memory/", log_filename) ,level=logging.INFO)


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
            self._cur_test_id = test_id
            self._last_tool_sig = None
            self._same_tool_streak = 0
            self._inject_memory_next = False

            print(f"[Info] Size of slots before reset: {len(self.slots)}")
            if len(self.slots) == 0:
                return
            # Transfer existing slots to long-term memory for every test entry
            '''asyncio.run(self.transfer_slots_to_memories(self.slots, is_abstract=True))'''
            # Multi-threaded version
            self.multi_thread_transfer_slots_to_memories()
            self.slot_process = SlotProcess(llm_name=self.model_name, llm_backend="openai") # Reset
            self.slots = [] # Reset

    def _strip_old_memory_block_keep_fc_items(self, message: list) -> list:
        out = []
        for m in message:
            if isinstance(m, dict):
                if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].startswith(self.MEM_TAG):
                    continue
            out.append(m)
        return out

    def slot_query(self, query_text: str, message: List[dict]):
        slot_query_limit = 3
        if len(query_text) > 0:
            relevant_slots = self.slot_process.query(query_text=query_text, slots=self.slots, limit=slot_query_limit, use_svd=False, embed_func=self.semantic_memory_system.vector_store._embed)
        else:
            relevant_slots = []
        return relevant_slots

    def long_term_memory_query(self, query_text: str, message: List[dict], threshold: float = 0.5):
        sem_query_limit = min(3, self.semantic_memory_system.size // 3)
        epi_query_limit = min(3, self.episodic_memory_system.size // 3)
        proc_query_limit = min(3, self.procedural_memory_system.size // 3)

        if len(query_text) > 0:
            relevant_semantic_memories = self.semantic_memory_system.query(query_text=query_text, limit=sem_query_limit, threshold=threshold)
            relevant_episodic_memories = self.episodic_memory_system.query(query_text=query_text, limit=epi_query_limit)
            relevant_procedural_memories = self.procedural_memory_system.query(query_text=query_text, limit=proc_query_limit)
        else:
            relevant_semantic_memories = []
            relevant_episodic_memories = []
            relevant_procedural_memories = []
        
        return relevant_semantic_memories, relevant_episodic_memories, relevant_procedural_memories

    def _inject_memory(self, query_text: str, message: List[dict], threshold: float = 0.4):
        relevant_slots = self.slot_query(query_text=query_text, message=message)        
        relevant_semantic_memories, relevant_episodic_memories, relevant_procedural_memories = self.long_term_memory_query(query_text=query_text, message=message, threshold=threshold)
        
        if len(relevant_slots) > 0:
            print(f"[Info] Size of Retrieved Slots: {len(relevant_slots)}")
            print(f"[Info] 1st Retrieved Slot: {_safe_dump_str(relevant_slots[0])}")

        slots_str = "\n".join(f"- {_safe_dump_str(entry[1].summary)}" for entry in relevant_slots)
        semantic_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].to_dict())}" for entry in relevant_semantic_memories)
        episodic_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].detail)}" for entry in relevant_episodic_memories)
        procedural_memories_str = "\n".join(f"- {_safe_dump_str(entry[1].to_dict())}" for entry in relevant_procedural_memories)

        self.logger_memory.info(f"Retrieved Slots:\n{slots_str} \nRetrieved Semantic Memories:\n{semantic_memories_str}\nRetrieved Episodic Memories:\n{episodic_memories_str}\nRetrieved Procedural Memories:\n{procedural_memories_str}")

        mem_msg = {
            "role": "user",
            "content": (
                f"{self.MEM_TAG}\n"
                "Context note (optional): the following are retrieved memories. "
                "They may be irrelevant. Prioritize the current user request.\n\n"
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
        '''mem_msg = {
            "role": "user",
            "content": (
                f"{self.MEM_TAG}\n"
                "Context note (optional): the following are retrieved memories. "
                "They may be irrelevant. Prioritize the current user request.\n\n"
                "Here are some relevant working slots:\n"
                f"{slots_str}\n"
            ),
        }'''

        #inject BEFORE last user message (FC-safe; user remains most recent)
        last_user_idx = None
        for i in range(len(message) - 1, -1, -1):
            if isinstance(message[i], dict) and message[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            # No user message found; just prepend
            return [mem_msg] + message

        return message[:last_user_idx] + [mem_msg] + message[last_user_idx:]

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]

        # Extract latest user query text
        query_text = self._extract_latest_user_query_text_keep_fc_items(message)
        # Inject memory
        if self._inject_memory_next or (not self._turn_injected_once):
            message = self._inject_memory(query_text=query_text, message=message, threshold=0.4)
            self._turn_injected_once = True
            self._inject_memory_next = False
        
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
        memory = str(model_response_data["model_responses"])
        _push_event(self._event_buffer, "ASSISTANT", memory)

        # Get tool signature for tracking
        sig = self._tool_signature(model_response_data.get("model_responses"))

        if sig is not None:
            if sig == self._last_tool_sig:
                self._same_tool_streak += 1
            else:
                self._last_tool_sig = sig
                self._same_tool_streak = 1

            if self._same_tool_streak >= self._trigger_k:
                self._inject_memory_next = True
                self._same_tool_streak = 0
        else:
            self._last_tool_sig = None
            self._same_tool_streak = 0

        if len(model_response_data.get("tool_call_ids", [])) == 0:
            # No tool calls means that the end of turn
            self._materialize_turn_slots(max_slots=5)
            # Remove old memory block to avoid accumulation
            message = inference_data["message"]
            message = self._strip_old_memory_block_keep_fc_items(message)
            inference_data["message"] = message

        return inference_data

    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = {"type": "function_call_output", "call_id": tool_call_id, "output": execution_result}
            inference_data["message"].append(tool_message)
            _push_event(self._event_buffer, "TOOL_RESULT", execution_result[:2000])
        return inference_data

    def _query_prompting(self, inference_data: dict):
        msg = inference_data["message"]
        msg = self._strip_old_memory_block_keep_fc_items(msg)
        query_text = self._extract_latest_user_query_text_keep_fc_items(msg)
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

    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result, has_tool_call_tag)

    def _extract_latest_user_query_text_keep_fc_items(self, message: list) -> str:
        for m in reversed(message):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
        return ""


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


    def _materialize_turn_slots(self, max_slots: int = 8):
        # transfer the latest turn snapshot to working slots
        snapshot_events = _drain_snapshot(event_buffer=self._event_buffer)
        if not snapshot_events:
            return

        new_slots = asyncio.run(
            self.slot_process.transfer_fc_agent_context_to_working_slots(
                context=snapshot_events,
                max_slots=max_slots,
            )
        )
        print(f"[Info] Materialized {len(new_slots)} new slots from turn snapshot.")
        for slot in new_slots:
            self.logger_context.info(f"New Slot\n: {_safe_dump_str(slot)}")
            self.slot_process.add_slot(slot)
        self.slots.extend(new_slots)

        self._turn_injected_once = False # Reset for next turn

    def _tool_signature(self, model_responses) -> str | None:
        if not isinstance(model_responses, list):
            return None

        parts = []
        for d in model_responses:
            if not isinstance(d, dict) or len(d) != 1:
                continue
            name = next(iter(d.keys()))
            args_raw = d[name]
            try:
                args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                args_norm = json.dumps(args_obj, sort_keys=True, ensure_ascii=False)
            except Exception:
                args_norm = str(args_raw)
            parts.append(f"{name}:{args_norm}")
        return "|".join(parts) if parts else None


    async def transfer_slots_to_memories(self, slots: List[WorkingSlot], is_abstract: bool = False):
        if len(slots) == 0:
            return

        routed_slot_container = await self.slot_process.filter_and_route_slots(slots=slots, task="fc")
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

        print(f"[Info] Num of Semantic Records to add: {len(semantic_records)}")
        print(f"[Info] Num of Episodic Records to add: {len(episodic_records)}")
        print(f"[Info] Num of Procedural Records to add: {len(procedural_records)}")

        self.semantic_memory_system.add(semantic_records)
        self.episodic_memory_system.add(episodic_records)
        self.procedural_memory_system.add(procedural_records)

    async def multi_thread_transfer_dicts_to_memories(self, is_abstract: bool = False):
        semantic_records = []
        episodic_records = []
        procedural_records = []

        for i in self.slot_process.memory_dict:
            if i['memory_type'] == 'semantic':
                semantic_records.append(self.semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(self.episodic_memory_system.instantiate_epi_record(**i['input']))
            elif i['memory_type'] == 'procedural':
                procedural_records.append(self.procedural_memory_system.instantiate_proc_record(**i['input']))

        print(f"[Info] Num of Semantic Records to add: {len(semantic_records)}")
        print(f"[Info] Num of Episodic Records to add: {len(episodic_records)}")
        print(f"[Info] Num of Procedural Records to add: {len(procedural_records)}")
        
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
        if len(procedural_records) > 0:
            try:
                self.procedural_memory_system.upsert_normal_records(procedural_records)
            except Exception as e:
                import traceback
                print("[ERROR] upsert_normal_records for procedural_records failed:", repr(e))
                traceback.print_exc()

    def multi_thread_transfer_slots_to_memories(self, max_workers: int = 20):
        # Multi-threaded version
        num_slots = len(self.slots)
        print(f"[Info] Filtering and routing {num_slots} slots")
        _multi_thread_run(self.slot_process.multi_thread_filter_and_route_slot, row_data=self.slots, max_workers=max_workers)
        routed_slots = self.slot_process.routed_slot_container
        num_routed_slots = len(routed_slots)
        print(f"[Info] Transferring memories from {num_routed_slots} slots to memory systems")
        _multi_thread_run(self.slot_process.multi_thread_transfer_slot_to_memory, row_data=routed_slots, max_workers=max_workers)
        # transfer memories to records
        asyncio.run(self.multi_thread_transfer_dicts_to_memories(is_abstract=False))

    async def abstract_episodic_records_to_semantic_record(self, epi_records: List[EpisodicRecord], consistency_threshold: float = 0.8):
        try:
            abstract_result, cidmap2semrec = await self.episodic_memory_system.abstract_episodic_records(epi_records, consistency_threshold)
            print(f"[Info] Number of abstracted semantic records: {len(abstract_result)}")
            self.semantic_memory_system.upsert_abstract_semantic_records(abstract_result, cidmap2semrec)
        except Exception as e:
            import traceback
            print("[ERROR] abstract_episodic_records_to_semantic_record failed:", repr(e))
            traceback.print_exc()