import json
import asyncio
import numpy as np
import re

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Any
from collections import deque
from memory.memory_system.utils import (
    dump_slot_json, 
    _extract_json_between, 
    _hard_validate_slot_keys,
    _build_context_snapshot,
    _safe_dump,
    _truncate_text,
    compute_overlap_score,
    new_id,
    now_iso,
    _extract_session_id_from_context,
)
from memory.memory_system.user_prompt import (
    WORKING_SLOT_COMPRESS_USER_PROMPT,
    TRANSFER_SLOT_TO_TEXT_PROMPT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_CHAT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_FC,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_QA,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_CHAT,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_FC,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_QA,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_CHAT,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_FC,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_QA,
    TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT,
    TRANSFER_QA_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT,
    TRANSFER_FC_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT,
    TRANSFER_CHAT_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT,
)
from textwrap import dedent
from memory.memory_system import WorkingSlot, OpenAIClient, LLMClient
from memory.memory_system.models import (
    EpisodicRecord,
    SemanticRecord,
    ProceduralRecord,
)
from memory.memory_system.schema import Schema
from tqdm import tqdm

class SlotProcess:
    def __init__(self, llm_name: str = "gpt-4o-mini", llm_backend: Literal["openai", "vllm"] = "openai", task: Literal["experiment", "qa", "fc", "chat"] = "qa"):
        self.slot_container: Dict[str, WorkingSlot] = {}
        self.filtered_slot_container: List[WorkingSlot] = []
        self.routed_slot_container: List[Dict] = []
        self.llm_model = OpenAIClient(model=llm_name, backend=llm_backend)
        self.memory_dict = []
        self.task = task
        self.total_working_slots = []

    def add_slot(self, slot: WorkingSlot) -> None:
        self.slot_container[slot.to_dict().get('id')] = slot
    
    def clear_container(self) -> None:
        self.slot_container = {}

    def get_container_size(self) -> int:
        return len(self.slot_container)

    def query(self, query_text: str, slots: Optional[List[WorkingSlot]] = None, limit: int = 5, key_words: Optional[List[str]] = None, use_svd: bool = False, embed_func = None, alpha: float = 0.9) -> List[Tuple[float, WorkingSlot]]:
        if slots is None:
            slots = list(self.slot_container.values())

        k = min(limit, len(slots))
        scored_slots: List[Tuple[float, WorkingSlot]] = []

        if len(slots) <= 3:
            use_svd = False

        if use_svd == False:
            # Normal Retrieval
            for slot in slots:
                score = compute_overlap_score(query_text, slot.summary, key_words)
                scored_slots.append((score, slot))
            scored_slots.sort(key=lambda x: x[0], reverse=True)
        
        else:
            # Reduced-SVD-based Retrieval
            assert embed_func is not None, "Embedding function must be provided when use_svd is True."
            query_emb = embed_func([query_text]) # [1, dim]
            slot_embs = embed_func([slot.summary for slot in slots]) # [n, dim]
            U, S, Vt = np.linalg.svd(slot_embs, full_matrices=False) # [n, dim] -> U[n, r], S: [r,], Vt: [r, dim]
            Sigma = np.diag(S)

            Z = (query_emb @ Vt.T).ravel()  # -> (r,)
            tmp = [(i, z_i) for i, z_i in enumerate(Z)]
            tmp.sort(key=lambda x : abs(x[1]), reverse=True)

            remain_index = list(range(len(slots)))
            for t in range(k):
                dim_idx = tmp[t][0]
                slot_score_triplet = []

                for idx in remain_index:
                    score = alpha * compute_overlap_score(query_text, slots[idx].summary, key_words) + (1 - alpha) * float((np.abs(U[idx, dim_idx]) / np.sum(np.abs(U[:, dim_idx]))))
                    slot_score_triplet.append((score, slots[idx], idx))
                slot_score_triplet.sort(key=lambda x: x[0], reverse=True)
                scored_slots.append((slot_score_triplet[0][0], slot_score_triplet[0][1]))

                # delete the chosen slot
                remain_index.remove(slot_score_triplet[0][2])

        return scored_slots[:k]
        
    async def filter_and_route_slots(self, slots: List[WorkingSlot] = None) -> List[Dict[str, WorkingSlot]]:
        self.filtered_slot_container = []
        self.routed_slot_container = []

        if slots is None:
            slots = list(self.slot_container.values())

        for slot in tqdm(slots):
            check_result = await slot.slot_filter(self.llm_model, task=self.task)
            print(check_result)
            if check_result == True:
                self.filtered_slot_container.append(slot)
        
        try:
            for filtered_slot in self.filtered_slot_container:
                route_result = await filtered_slot.slot_router(self.llm_model, task=self.task)
                pair = {
                    "memory_type": route_result,
                    "slot": filtered_slot
                }
                self.routed_slot_container.append(pair)
        except Exception as e:
            print(f"Routing error: {e}")
        
        return self.routed_slot_container

    def multi_thread_filter_and_route_slot(self, slot: WorkingSlot):
        #check_result = asyncio.run(slot.slot_filter(self.llm_model, task=self.task))
        check_result = True
        if check_result == True:
            try:
                route_result = asyncio.run(slot.slot_router(self.llm_model))
                pair = {
                        "memory_type": route_result,
                        "slot": slot
                    }
                self.routed_slot_container.append(pair)
            except Exception as e:
                print(f"Routing error: {e}")
        else:
            return

    
    async def compress_slots(self, sids: List[str] = None) -> WorkingSlot:
        slot_json_blobs = []
        if sids is None:
            for idx, slot in enumerate(self.slot_container.values()):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(slot)}")
        else:
            for idx, slot_id in enumerate(sids):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(self.slot_container[slot_id])}")
        slots_block = "\n\n".join(slot_json_blobs)

        system_prompt = (
            "You are an expert research assistant and memory compressor. "
            "Given multiple WorkingSlot JSON dumps, produce a single, compact summary "
            "that preserves non-redundant, reusable knowledge while discarding noise."
            "Be precise, consistent, and avoid hallucinations. Output only the requested JSON inside the tags."
        )
        user_prompt = WORKING_SLOT_COMPRESS_USER_PROMPT.format(slots_block=slots_block)

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "compressed-slot", "compressed-slot")
        print(f"Compressed slot payload: {payload}")
        try:
            _hard_validate_slot_keys(payload, allowed_keys={"stage", "topic", "summary", "attachments", "tags"})
        except Exception as e:
            raise ValueError(f"Compressed slot validation error: {e}")
        
        stage = str(payload.get("stage", ""))
        topic = str(payload.get("topic", ""))
        summary = str(payload.get("summary", ""))
        attachments = payload.get("attachments", {})
        tags = payload.get("tags", [])

        compressed_slot = WorkingSlot(
            stage=stage,
            topic=topic,
            summary=summary,
            attachments=attachments,
            tags=tags
        )

        return compressed_slot
    
    async def transfer_slot_to_text(self, slot: WorkingSlot) -> str:
        system_prompt = (
            "You are an expert assistant that converts WorkingSlot JSON data into a clear, concise text summary. "
            "Focus on key insights, important metrics, and actionable items. Output only the requested text inside the tags."
        )

        user_prompt = TRANSFER_SLOT_TO_TEXT_PROMPT.format(dump_slot_json=dump_slot_json(slot))

        text = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return text

    def _retry_llm_to_slots(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        schema_name: str,
        allowed_keys: set,
        max_slots: int,
        max_retries: int = 5,
        max_tokens: int = 4096,
        post_process_slot: Optional[Callable[[dict, str], dict]] = None,
        context: Optional[str] = None,
        is_async: bool = False,
    ) -> List[WorkingSlot]:
        """
        Generic retry wrapper for LLM -> WorkingSlot conversion.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt for LLM.
            json_schema: JSON schema for structured output.
            schema_name: Name of the schema.
            allowed_keys: Allowed keys in slot dict for validation.
            max_slots: Maximum number of slots to return.
            max_retries: Number of retry attempts.
            max_tokens: Max tokens for LLM response.
            post_process_slot: Optional callable(slot_dict, context) -> slot_dict for per-slot post-processing.
            context: Original context string (passed to post_process_slot if provided).
            is_async: If True, use asyncio.run; otherwise assume already in async context.
        
        Returns:
            List of valid WorkingSlot objects.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                if is_async:
                    response = asyncio.run(self.llm_model.complete(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        strict=False,
                        max_tokens=max_tokens
                    ))
                else:
                    # For async methods, we need to await directly
                    # This branch is used when called from sync context
                    response = asyncio.run(self.llm_model.complete(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        strict=False,
                        max_tokens=max_tokens
                    ))
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] LLM call error: {e}")
                continue

            try:
                data = json.loads(response)
                if not data:
                    raise ValueError("Empty JSON response")
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] JSON parsing error: {e}")
                continue

            slots_data = data.get("slots", [])
            if not isinstance(slots_data, list):
                last_error = ValueError("`slots` must be a list.")
                print(f"[Retry {attempt}/{max_retries}] Invalid schema: `slots` must be a list.")
                continue

            working_slots: List[WorkingSlot] = []

            for slot_dict in slots_data[:max_slots]:
                if not isinstance(slot_dict, dict):
                    continue

                try:
                    _hard_validate_slot_keys(slot_dict, allowed_keys=allowed_keys)

                    # Apply post-processing if provided
                    if post_process_slot is not None and context is not None:
                        slot_dict = post_process_slot(slot_dict, context)

                    stage = str(slot_dict.get("stage", "")).strip()
                    topic = str(slot_dict.get("topic", "")).strip()
                    summary = str(slot_dict.get("summary", "")).strip()
                    attachments = slot_dict.get("attachments") or {}
                    tags = slot_dict.get("tags") or []

                    slot = WorkingSlot(
                        stage=stage,
                        topic=topic,
                        summary=summary,
                        attachments=attachments,
                        tags=list(tags),
                    )
                    working_slots.append(slot)

                except Exception as e:
                    last_error = e
                    print(f"[Retry {attempt}/{max_retries}] Error creating WorkingSlot: {e}")
                    continue

            if len(working_slots) != 0:
                return working_slots

            print(f"[Retry {attempt}/{max_retries}] No valid WorkingSlot created; retrying...")

        print(f"Failed to create any WorkingSlot after {max_retries} retries. Last error: {last_error}")
        return []

    @staticmethod
    def _post_process_chat_slot(slot_dict: dict, context: str) -> dict:
        """Post-process chat slot to inject extracted session_id."""
        try:
            extracted_session_id = _extract_session_id_from_context(context)
        except Exception as e:
            print(f"Session ID extraction error: {e}")
            raise

        attachments = slot_dict.get("attachments") or {}
        if not isinstance(attachments, dict):
            attachments = {}
        
        session_ids = attachments.get("session_ids")
        if not isinstance(session_ids, dict):
            session_ids = {"items": []}
            attachments["session_ids"] = session_ids
        
        session_ids["items"] = [extracted_session_id]
        slot_dict["attachments"] = attachments
        
        return slot_dict

    def _retry_llm_to_record(
        self,
        system_prompt: str,
        user_prompt: str,
        record_tag: str,
        slot: WorkingSlot,
        post_process_payload: Callable[[dict, WorkingSlot], Dict[str, Any]],
        max_retries: int = 5,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generic retry wrapper for LLM -> Record (semantic/episodic/procedural) conversion.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt for LLM.
            record_tag: The XML-like tag to extract JSON from (e.g., "semantic-record").
            slot: The source WorkingSlot (used as fallback for missing fields).
            post_process_payload: Callable(payload_dict, slot) -> final_record_dict.
            max_retries: Number of retry attempts.
            max_tokens: Max tokens for LLM response.
        
        Returns:
            The final record dict.
        
        Raises:
            ValueError: If all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                response = asyncio.run(self.llm_model.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens
                ))
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] LLM call error: {e}")
                continue

            try:
                # Clean up response
                response = response.strip()
                response = response.replace("<think>", "").replace("</think>", "")

                payload = _extract_json_between(response, record_tag, record_tag)
                
                if not payload or not isinstance(payload, dict):
                    raise ValueError(f"Empty or invalid payload extracted from <{record_tag}>")

                # Apply post-processing to build final record
                record = post_process_payload(payload, slot)
                return record

            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] Record extraction/processing error: {e}")
                continue

        raise ValueError(f"Failed to create record after {max_retries} retries. Last error: {last_error}")

    async def transfer_qa_agent_context_to_working_slots(self, context: str, max_slots: int = 20) -> List[WorkingSlot]:
        system_prompt = (
            "You are an expert workflow archivist. "
            "Transform the provided QA Agent context into WorkingSlot JSON objects. "
            "Each slot must capture the stage, topic, summary (≤120 words), attachments, and tags. "
            "Summaries must follow a Situation→Action→Result narrative whenever possible. "
            "You MUST output at least one slot."
        )

        user_prompt = TRANSFER_QA_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT.format(
            max_slots=max_slots,
            snapshot=context,
        )

        schema = Schema(max_slots=max_slots)
        qa_task_slot_schema = schema.QA_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=qa_task_slot_schema,
            schema_name="QA_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=context,
            is_async=False,
        )

        return working_slots

    async def transfer_fc_agent_context_to_working_slots(self, context: str, max_slots: int = 50) -> List[WorkingSlot]:
        system_prompt = (
            "You are an expert tool-using agent archivist. "
            "Transform BFCL-style multi-turn tool-calling trajectories into reusable memory slots. "
            "You must distinguish Semantic evidence, Episodic experience, and Procedural experience. "
            "You MUST output at least one slot. "
            "Output strictly as JSON."
        )

        user_prompt = TRANSFER_FC_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT.format(
            max_slots=max_slots,
            snapshot=context,
        )

        schema = Schema(max_slots=max_slots)
        fc_task_slot_schema = schema.FC_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=fc_task_slot_schema,
            schema_name="FC_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=context,
            is_async=False,
        )

        return working_slots

    async def transfer_experiment_agent_context_to_working_slots(self, context, state: str, max_slots: int = 50) -> List[WorkingSlot]:
        
        if state not in {"pre_analysis", "code_plan", "code_implement", "code_judge", "experiment_execute", "experiment_analysis"}:
            return []

        snapshot = _build_context_snapshot(context, state)

        system_prompt = (
            "You are an expert workflow archivist. "
            "Transform the provided Experiment Agent context into WorkingSlot JSON objects. "
            "Each slot must capture the stage, topic, summary (≤120 words), attachments, and tags. "
            "Summaries must follow a Situation→Action→Result narrative whenever possible. "
            "You MUST output at least one slot."
        )

        user_prompt = TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT.format(
            max_slots=max_slots,
            snapshot=snapshot,
        )

        schema = Schema(max_slots=max_slots)
        experiment_task_slot_schema = schema.EXPERIMENT_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=experiment_task_slot_schema,
            schema_name="EXPERIMENT_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=snapshot,
            is_async=False,
        )

        return working_slots

    async def transfer_experiment_agent_context_to_working_slots(self, context, state: str, max_slots: int = 50) -> List[WorkingSlot]:
        
        if state not in {"pre_analysis", "code_plan", "code_implement", "code_judge", "experiment_execute", "experiment_analysis"}:
            return []

        snapshot = _build_context_snapshot(context, state)

        system_prompt = (
            "You are an expert workflow archivist. "
            "Transform the provided Experiment Agent context into WorkingSlot JSON objects. "
            "Each slot must capture the stage, topic, summary (≤120 words), attachments, and tags. "
            "Summaries must follow a Situation→Action→Result narrative whenever possible. "
            "You MUST output at least one slot."
        )

        user_prompt = TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT.format(
            max_slots=max_slots,
            snapshot=snapshot,
        )

        schema = Schema(max_slots=max_slots)
        experiment_task_slot_schema = schema.EXPERIMENT_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=experiment_task_slot_schema,
            schema_name="EXPERIMENT_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=snapshot,
            is_async=False,
        )

        return working_slots

    def transfer_chat_agent_context_to_working_slots(self, context: str, max_slots: int = 50) -> List[WorkingSlot]:
        system_prompt = (
            "You are a personal memory archivist. "
            "Extract stable user facts (preferences, attributes, relationships, possessions) from the WorkingSlot "
            "into a semantic memory entry suitable for answering questions about the user's history. "
            "Focus on timeless facts, not event narratives. Output only the requested JSON."
        )

        user_prompt = TRANSFER_CHAT_AGENT_CONTEXT_TO_WORKING_SLOT_PROMPT.format(
            max_slots=max_slots,
            snapshot=context,
        )

        user_prompt += " "

        schema = Schema(max_slots=max_slots)
        chat_task_slot_schema = schema.CHAT_TASK_SLOT_SCHEMA

        # retry policy: up to 5 attempts if we fail to create any valid WorkingSlot
        max_retries = 5
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            response = asyncio.run(self.llm_model.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=chat_task_slot_schema,
                schema_name="CHAT_TASK_SLOT_SCHEMA",
                strict=False,
                max_tokens=4096
            ))

            try:
                data = json.loads(response)
                if not data:
                    raise ValueError("Empty JSON response")
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] JSON parsing error: {e}")
                continue

            slots_data = data.get("slots", [])
            if not isinstance(slots_data, list):
                last_error = ValueError("`slots` must be a list.")
                print(f"[Retry {attempt}/{max_retries}] Invalid schema: `slots` must be a list.")
                continue

            working_slots: List[WorkingSlot] = []
            allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

            try:
                extracted_session_id = _extract_session_id_from_context(context)
            except Exception as e:
                # If we cannot extract session_id, retry won't help; fail fast for this call.
                print(f"Session ID extraction error: {e}")
                return []

            for slot_dict in slots_data[:max_slots]:
                if not isinstance(slot_dict, dict):
                    continue

                try:
                    _hard_validate_slot_keys(slot_dict, allowed_keys=allowed_keys)

                    stage = str(slot_dict.get("stage", "")).strip()
                    topic = str(slot_dict.get("topic", "")).strip()
                    summary = str(slot_dict.get("summary", "")).strip()
                    attachments = slot_dict.get("attachments") or {}
                    tags = slot_dict.get("tags") or []

                    # Ensure attachments has the expected session_ids structure, then hard-set it.
                    if not isinstance(attachments, dict):
                        attachments = {}
                    session_ids = attachments.get("session_ids")
                    if not isinstance(session_ids, dict):
                        session_ids = {"items": []}
                        attachments["session_ids"] = session_ids
                    items = session_ids.get("items")
                    if not isinstance(items, list):
                        items = []
                        session_ids["items"] = items
                    session_ids["items"] = [extracted_session_id]

                    slot = WorkingSlot(
                        stage=stage,
                        topic=topic,
                        summary=summary,
                        attachments=attachments,
                        tags=list(tags),
                    )
                    working_slots.append(slot)

                except Exception as e:
                    last_error = e
                    print(f"[Retry {attempt}/{max_retries}] Error creating WorkingSlot from one slot: {e}")
                    continue

            if len(working_slots) != 0:
                self.total_working_slots.extend(working_slots)
                return working_slots

            # If we got here, no valid slots were created; retry the LLM call.
            print(f"[Retry {attempt}/{max_retries}] No valid WorkingSlot created; retrying...")

        # Exhausted retries
        print(f"Failed to create any WorkingSlot after {max_retries} retries. Last error: {last_error}")
        return []

    async def generate_long_term_memory(self, routed_slots: List[Dict[str, WorkingSlot]]) -> List[Dict[str, Any]]:
        allowed_types = {"semantic", "episodic", "procedural"}
        inputs: List[Dict[str, Any]] = []

        for pair in tqdm(routed_slots):
            memory_type = pair.get("memory_type")
            slot = pair.get("slot")

            if memory_type not in allowed_types or not isinstance(slot, WorkingSlot):
                continue

            try:
                if memory_type == "semantic":
                    input_dict = await self.transfer_slot_to_semantic_record(slot)
                elif memory_type == "episodic":
                    input_dict = await self.transfer_slot_to_episodic_record(slot)
                else:
                    input_dict = await self.transfer_slot_to_procedural_record(slot)
            except Exception as exc:
                print(
                    f"[MEMORY] Failed to convert slot {getattr(slot, 'id', 'unknown')} "
                    f"({memory_type}): {exc}"
                )
                continue

            if inputs is not None:
                inputs.append({"memory_type": memory_type, "input": input_dict})

        return inputs

    def multi_thread_transfer_slot_to_memory(self, pair: Dict[str, WorkingSlot]) -> List[Dict[str, Any]]:
        allowed_types = {"semantic", "episodic", "procedural"}
        memory_type = pair.get("memory_type")
        slot = pair.get("slot")

        if memory_type not in allowed_types or not isinstance(slot, WorkingSlot):
            return

        try:
            if memory_type == "semantic":
                input_dict = asyncio.run(self.transfer_slot_to_semantic_record(slot))
            elif memory_type == "episodic":
                input_dict = asyncio.run(self.transfer_slot_to_episodic_record(slot))
            elif memory_type == "procedural":
                input_dict = asyncio.run(self.transfer_slot_to_procedural_record(slot,))
        except Exception as exc:
            print(
                f"[MEMORY] Failed to convert slot {getattr(slot, 'id', 'unknown')} "
                f"({memory_type}): {exc}"
            )
            return

        self.memory_dict.append({"memory_type": memory_type, "input": input_dict})


    async def transfer_slot_to_semantic_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        if self.task == "experiment":
            system_prompt = (
                "You are a senior research archivist. Convert the WorkingSlot into a reusable "
                "semantic memory entry that captures enduring, generalizable insights."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT.format(dump_slot_json=dump_slot_json(slot))
        
        elif self.task == "qa":
            system_prompt = (
                "You are an expert knowledge base curator for multi-hop QA systems. "
                "Extract stable, factual evidence from the WorkingSlot (e.g., entity attributes, relations, Wikipedia facts) "
                "into a semantic memory entry suitable for future question answering. "
                "Focus on facts, not strategies. Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_QA.format(dump_slot_json=dump_slot_json(slot))
         
        elif self.task == "fc":
            system_prompt = (
                "You are a function-calling schema expert. "
                "Extract stable tool constraints, argument mappings, and validated invariants from the WorkingSlot "
                "into a semantic memory entry. Focus on declarative knowledge (what is true), not procedural steps (how to do). "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_FC.format(dump_slot_json=dump_slot_json(slot))
        
        else:
            system_prompt = (
                "You are a personal memory archivist. "
                "Extract stable user facts (preferences, attributes, relationships, possessions) from the WorkingSlot "
                "into a semantic memory entry suitable for answering questions about the user's history. "
                "Focus on timeless facts, not event narratives. Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_CHAT.format(dump_slot_json=dump_slot_json(slot))

        user_prompt += " /no_think"

        def post_process_semantic(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            summary = payload.get("summary") or slot.summary
            detail = payload.get("detail") or slot.summary
            tags = payload.get("tags") or slot.tags
            sem_id = new_id(prefix="sem")
            created_at = now_iso()
            updated_at = now_iso()

            return {
                "id": sem_id,
                "summary": summary.strip() if isinstance(summary, str) else str(summary),
                "detail": detail.strip() if isinstance(detail, str) else str(detail),
                "tags": list(tags) if tags else [],
                "created_at": created_at,
                "updated_at": updated_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="semantic-record",
            slot=slot,
            post_process_payload=post_process_semantic,
            max_retries=5,
            max_tokens=2048,
        )

    async def transfer_slot_to_episodic_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        if self.task == "experiment":
            system_prompt = (
                "You are a scientific lab journal assistant. Convert the WorkingSlot into an episodic "
                "memory capturing Situation → Action → Result, including measurable outcomes."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)
        
        elif self.task == "qa":
            system_prompt = (
                "You are a QA workflow historian. "
                "Convert the WorkingSlot into an episodic memory capturing the Situation → Action → Result narrative "
                "of how a multi-hop question was approached. Include retrieval strategies, evidence paths, and failure patterns. "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_QA.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)
        
        elif self.task == "fc":
            system_prompt = (
                "You are a function-calling trajectory logger. "
                "Convert the WorkingSlot into an episodic memory capturing the Situation → Action → Result narrative "
                "of a tool-use episode. Include error handling, retry logic, and tool output processing patterns. "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_FC.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)
        
        else:
            system_prompt = (
                "You are a personal timeline curator. "
                "Convert the WorkingSlot into an episodic memory capturing a chronological user experience "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_CHAT.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)

        user_prompt += " /no_think"

        def post_process_episodic(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            stage = payload.get("stage") or slot.stage
            summary = payload.get("summary") or slot.summary
            detail = payload.get("detail") or {}
            tags = payload.get("tags") or slot.tags
            epi_id = new_id(prefix="epi")
            created_at = now_iso()
            
            if not isinstance(detail, dict):
                detail = {"notes": detail}

            return {
                "id": epi_id,
                "stage": stage.strip() if isinstance(stage, str) else str(stage),
                "summary": summary.strip() if isinstance(summary, str) else str(summary),
                "detail": detail,
                "tags": list(tags) if tags else [],
                "created_at": created_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="episodic-record",
            slot=slot,
            post_process_payload=post_process_episodic,
            max_retries=5,
            max_tokens=2048,
        )

    async def transfer_slot_to_procedural_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        if self.task == "experiment":
            system_prompt = (
                "You are an expert operations documenter. Convert the WorkingSlot into a procedural "
                "memory entry describing reproducible steps/commands."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT.format(dump_slot_json=dump_slot_json(slot))
        
        elif self.task == "qa":
            system_prompt = (
                "You are a QA playbook author. "
                "Convert the WorkingSlot into a procedural memory entry capturing reusable step-by-step strategies "
                "for solving multi-hop questions. Focus on actionable, generalizable workflows. "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_QA.format(dump_slot_json=dump_slot_json(slot))
        
        elif self.task == "fc":
            system_prompt = (
                "You are a tool-use SOP (Standard Operating Procedure) writer. "
                "Convert the WorkingSlot into a procedural memory entry capturing reusable checklists and playbooks "
                "for function calling. Include validation steps, error recovery procedures, and best practices. "
                "Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_FC.format(dump_slot_json=dump_slot_json(slot))
        
        else:
            system_prompt = (
                "You are a personal assistant workflow designer. "
                "Convert the WorkingSlot into a procedural memory entry capturing reusable interaction patterns "
                "or decision-making workflows related to user preferences and habits. "
                "Focus on generalizable, user-centric procedures. Output only the requested JSON."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_CHAT.format(dump_slot_json=dump_slot_json(slot))

        user_prompt += " /no_think"

        def post_process_procedural(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            name = payload.get("name") or slot.topic or "skill"
            description = payload.get("description") or slot.summary
            steps = payload.get("steps") or []
            code = payload.get("code")
            tags = payload.get("tags") or slot.tags
            proc_id = new_id(prefix="proc")
            created_at = now_iso()
            updated_at = now_iso()

            if isinstance(steps, str):
                steps = [steps]

            clean_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]

            return {
                "id": proc_id,
                "name": name.strip() if isinstance(name, str) else str(name),
                "description": description.strip() if isinstance(description, str) else str(description),
                "steps": clean_steps,
                "code": code.strip() if isinstance(code, str) else None,
                "tags": list(tags) if tags else [],
                "created_at": created_at,
                "updated_at": updated_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="procedural-record",
            slot=slot,
            post_process_payload=post_process_procedural,
            max_retries=5,
            max_tokens=2048,
        )