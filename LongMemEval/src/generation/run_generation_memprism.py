import sys
import json
from tqdm import tqdm
import openai
from openai import OpenAI
import backoff
import random
import numpy as np
from datetime import datetime, timedelta
import argparse
from transformers import AutoTokenizer
import tiktoken
import asyncio
from typing import List, Dict, Any, Optional

from memory.memory_system.utils import (
    _safe_dump_str,
    _multi_thread_run,
)
from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.models import EpisodicRecord
from memory.memory_system.working_slot import WorkingSlot
from textwrap import dedent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--resume_file', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--out_file_suffix', type=str, default="")
        
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--openai_base_url', type=str, default=None)
    parser.add_argument('--openai_key', type=str, required=True)
    parser.add_argument('--openai_organization', type=str, default=None)

    parser.add_argument('--retriever_type', type=str, required=True)
    parser.add_argument('--topk_context', type=int, required=True)
    parser.add_argument('--history_format', type=str, required=True, choices=['json', 'nl'])
    parser.add_argument('--useronly', type=str, required=True, choices=['true', 'false'])
    parser.add_argument('--cot', type=str, required=True, choices=['true', 'false'])
    parser.add_argument('--con', type=str, required=False, choices=['true', 'false'], default='false')

    # user fact expansion
    parser.add_argument('--merge_key_expansion_into_value', type=str, choices=['merge', 'replace', 'none'], default='none')     # merge key expansion into value

    parser.add_argument('--gen_length', type=int, default=None)
    
    return parser.parse_args()


def check_args(args):
    print(args)


def prepare_prompt(entry, retriever_type, topk_context: int, useronly: bool, history_format: str, cot: bool, tokenizer, tokenizer_backend, max_retrieval_length, merge_key_expansion_into_value, slot_process_vllm, slot_process_openai, semantic_memory_system, episodic_memory_system, con=False, con_client=None, con_model=None, max_workers=50):    
    if retriever_type == 'no-retrieval':
        answer_prompt_template = '{}'
        if cot:
            answer_prompt_template += 'Answer step by step.'
            
    else:
        if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
            if cot:
                answer_prompt_template = 'I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        elif merge_key_expansion_into_value == 'merge':
            if cot:
                answer_prompt_template = 'I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several history chats between you and a user, as well as the relevant user facts extracted from the chat history. Please answer the question based on the relevant chat history and the user facts\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        elif merge_key_expansion_into_value == 'replace':
            if cot:
                answer_prompt_template = 'I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):'
            else:
                answer_prompt_template = 'I will give you several facts extracted from history chats between you and a user. Please answer the question based on the relevant facts.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'
        else:
            raise NotImplementedError
        
    question_date_string = entry['question_date']
    question_string = entry['question']

    corpusid2date, corpusid2entry = {}, {}
    for session_date, session_id, session_entry in zip(entry['haystack_dates'], entry['haystack_session_ids'], entry['haystack_sessions']):
        corpusid2date[session_id] = session_date
        corpusid2entry[session_id] = session_entry
        for i_turn, turn_entry in enumerate(session_entry):
            corpusid2date[session_id + '_' + str(i_turn+1)] = session_date
            corpusid2entry[session_id + '_' + str(i_turn+1)] = turn_entry

    corpusid2retvalue = {}
    try:
        for ret_result_entry in entry['retrieval_results']['ranked_items']:
            corpusid2retvalue[ret_result_entry['corpus_id']] = ret_result_entry['text']
    except:
        pass
    
    retrieved_chunks = []
    # get chunks in the original order
    if retriever_type == "orig-session":   # no retrieval, session
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
            else:
                retrieved_chunks.append((session_date, session_entry))
    elif retriever_type == "orig-turn":  # no retrieval, turn
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks += [(session_date, x) for x in session_entry if x['role'] == 'user']
            else:
                retrieved_chunks += [(session_date, x) for x in session_entry]

    # only retain oracle chunks 
    elif retriever_type == "oracle-session":   # no retrieval, session
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
            else:
                retrieved_chunks.append((session_date, session_entry))
    elif retriever_type == "oracle-turn":  # no retrieval, turn
        for session_date, session_entry in zip(entry['haystack_dates'], entry['haystack_sessions']):
            if useronly:
                retrieved_chunks += [(session_date, x) for x in session_entry if x['role'] == 'user']
            else:
                retrieved_chunks += [(session_date, x) for x in session_entry]

    elif retriever_type == "memprism-session":
        idmap2entry = {}
        context_list = []
        for session_date, session_entry, session_id in zip(entry['haystack_dates'], entry['haystack_sessions'], entry['haystack_session_ids']):
            idmap2entry[session_id] = (session_date, session_entry)

            if useronly:
                retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
                session_str = _safe_dump_str([x for x in session_entry if x['role'] == 'user'])
            else:
                retrieved_chunks.append((session_date, session_entry))
                session_str = _safe_dump_str(session_entry)
            context = dedent(f"""
            Here is a chat session between you and a user:
            Session ID: {session_id}
            Session Content: {session_str}
            """)
            context_list.append(context)
    
        # transfer chat context to working slots, filter and route slots, and transfer slots to memory
        _multi_thread_run(slot_process_vllm.transfer_chat_agent_context_to_working_slots, context_list, max_workers=10)
        working_slots = slot_process_vllm.total_working_slots.copy()           
        print(f"[Info] Transferring session {session_id} to memories, number of working slots: {len(working_slots)}")
        _multi_thread_run(slot_process_openai.multi_thread_filter_and_route_slot, working_slots, max_workers=max_workers)

        _multi_thread_run(slot_process_openai.multi_thread_transfer_slot_to_memory, slot_process_openai.routed_slot_container, max_workers=max_workers)
        asyncio.run(multi_thread_transfer_dicts_to_memories(slot_process_openai, semantic_memory_system, episodic_memory_system))
        
        print(f"[Info] Finished transferring, size of semantic memory: {semantic_memory_system.size}, size of episodic memory: {episodic_memory_system.size}")
        
        semantic_records, episodic_records, session_ids = get_related_information_by_query(query=question_string, slot_process=slot_process_vllm, working_slots=working_slots, semantic_memory_system=semantic_memory_system, episodic_memory_system=episodic_memory_system)

        # clean up retrieved chunks
        retrieved_chunks = []

        for session_id in session_ids:
            if session_id in idmap2entry.keys():
                session_date, session_entry = idmap2entry[session_id]
                if useronly:
                    retrieved_chunks.append((session_date, [x for x in session_entry if x['role'] == 'user']))
                else:
                    retrieved_chunks.append((session_date, session_entry))
        
        print(f"[Info] Retrieved {len(retrieved_chunks)} sessions from MemPrism for question: {question_string}")

            
    # get retrieved chunks
    elif retriever_type == "flat-turn":
        for ret_result_entry in entry['retrieval_results']['ranked_items']:
            converted_corpus_id = '_'.join(ret_result_entry['corpus_id'].replace('noans_', 'answer_').split('_')[:-1])
            converted_turn_id = int(ret_result_entry['corpus_id'].replace('noans_', 'answer_').split('_')[-1]) - 1   # we had offset one during retrieval
            # automatically expand turn into round
            try:
                cur_round_data = [corpusid2entry[converted_corpus_id][converted_turn_id]]
                converted_next_turn_id = converted_turn_id + 1
                if converted_next_turn_id < len(corpusid2entry[converted_corpus_id]):
                    cur_round_data.append(corpusid2entry[converted_corpus_id][converted_next_turn_id])
                
            except:
                continue
            
            # handle optional merging key into the value
            if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
                retrieved_chunks.append((corpusid2date[converted_corpus_id], cur_round_data))
            elif merge_key_expansion_into_value == 'replace':
                retrieved_chunks.append((corpusid2date[converted_corpus_id], corpusid2retvalue[ret_result_entry['corpus_id']]))
            elif merge_key_expansion_into_value == 'merge':
                retrieved_chunks.append((corpusid2date[converted_corpus_id], corpusid2retvalue[ret_result_entry['corpus_id']], cur_round_data))
            else:
                raise NotImplementedError

        if useronly and not merge_key_expansion_into_value == 'replace':
            retrieved_chunks = [x for x in retrieved_chunks if x[-1]['role'] == 'user']     

    elif retriever_type == "flat-session":
        for ret_result_entry in entry['retrieval_results']['ranked_items']:
            # handle optional merging key into the value
            if merge_key_expansion_into_value is None or merge_key_expansion_into_value == 'none':
                if useronly:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')],
                                            [x for x in corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')] if x['role'] == 'user']))
                else:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')]))
            elif merge_key_expansion_into_value == 'replace':
                retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], corpusid2retvalue[ret_result_entry['corpus_id']]))
            elif merge_key_expansion_into_value == 'merge':
                if useronly:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], corpusid2retvalue[ret_result_entry['corpus_id']], [x for x in corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')] if x['role'] == 'user']))
                else:
                    retrieved_chunks.append((corpusid2date[ret_result_entry['corpus_id'].replace('noans_', 'answer_')], corpusid2retvalue[ret_result_entry['corpus_id']], corpusid2entry[ret_result_entry['corpus_id'].replace('noans_', 'answer_')]))
            else:
                raise NotImplementedError

    elif retriever_type == "no-retrieval":
        retrieved_chunks = []
        
    else:
        raise NotImplementedError

    if retriever_type in ["orig-turn", "orig-session"]:
        retrieved_chunks = retrieved_chunks[-topk_context:]  # keep latest
    else:
        retrieved_chunks = retrieved_chunks[:topk_context]

    # clean up
    retrieved_chunks_cleaned = []
    for retrieved_item in retrieved_chunks:
        try:
            date, session_entry = retrieved_item
            for turn_entry in session_entry:
                if type(turn_entry) == dict and 'has_answer' in turn_entry:
                    turn_entry.pop('has_answer')
            retrieved_chunks_cleaned.append((date, session_entry))
        except:
            date, expansion_entry, session_entry = retrieved_item
            for turn_entry in session_entry:
                if type(turn_entry) == dict and 'has_answer' in turn_entry:
                    turn_entry.pop('has_answer')
            retrieved_chunks_cleaned.append((date, expansion_entry, session_entry))
    retrieved_chunks = retrieved_chunks_cleaned

    # optional: if CoN is specified, add an information extraction process before feeding into the model
    if con:
        con_prompt = "I will give you a chat history between you and a user, as well as a question from the user. Write reading notes to extract all the relevant user information relevant to answering the answer. If no relevant information is found, just output \"empty\". \n\n\nChat History:\nSession Date: {}\nSession Content:\n{}\n\nQuestion Date: {}\nQuestion: {}\nExtracted note (information relevant to answering the question):"
        retrieved_chunks_with_notes = []
        for i, cur_item in enumerate(retrieved_chunks):
            if merge_key_expansion_into_value == 'merge':
                (chunk_date, chunk_expansion_entry, chunk_entry) = cur_item
                                
            else:
                (chunk_date, chunk_entry) = cur_item
                
            kwargs = {
                'model': con_model,
                'messages':[
                    {"role": "user", "content": con_prompt.format(chunk_date, json.dumps(chunk_entry), question_date_string, question_string)}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': 500,
            }
            completion = chat_completions_with_backoff(con_client, **kwargs) 
            cur_note = completion.choices[0].message.content.strip()
            chunk_entry_con = {'session_summary': cur_note}

            if merge_key_expansion_into_value == 'merge':
                retrieved_chunks_with_notes.append((chunk_date, chunk_expansion_entry, chunk_entry_con))
            else:
                retrieved_chunks_with_notes.append((chunk_date, chunk_entry_con))

        retrieved_chunks = retrieved_chunks_with_notes
                
    # sort sessions by their dates

    retrieved_chunks.sort(key=lambda x: x[0])
    
    history_string = ""
    for i, cur_item in enumerate(retrieved_chunks):
        if merge_key_expansion_into_value == 'merge':
            (chunk_date, chunk_expansion_entry, chunk_entry) = cur_item
        else:
            (chunk_date, chunk_entry) = cur_item

        if history_format == 'json':
            if merge_key_expansion_into_value == 'merge':
                sess_string = '\n' + json.dumps({'session_summary_facts': chunk_expansion_entry, 'original_session': chunk_entry})
            else:
                sess_string = '\n' + json.dumps(chunk_entry)
        elif history_format == 'nl':
            sess_string = ""
            if merge_key_expansion_into_value == 'merge':
                sess_string += "\n\nSession summary and facts:" + chunk_expansion_entry
            if type(chunk_entry) == list:
                for turn_entry in chunk_entry:
                    sess_string += "\n\n{}: {}".format(turn_entry['role'], turn_entry['content'].strip())
            else:
                sess_string += "{}: {}".format(chunk_entry['role'], chunk_entry['content'].strip())    
        else:
            raise NotImplementedError

        if retriever_type in ["orig-session", "flat-session", "oracle-session", "memprism-session"]:
            history_string += '\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n'.format(i+1, chunk_date, sess_string)
        elif retriever_type in ["orig-turn", "flat-turn", "oracle-turn"]:  
            # history_string += '\n### Round {}:\nDate: {}\nRound Content:\n{}\n'.format(i+1, chunk_date, sess_string)
            history_string += '\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n'.format(i+1, chunk_date, sess_string)  # we include both sides right now
        elif retriever_type == "no-retrieval":
            history_string += ""
        else:
            raise NotImplementedError

        semantic_memories_str = ""
        episodic_memories_str = ""

        if len(semantic_records) > 0:
            semantic_memories_str = "\n".join(f"- {record.summary}" for record in semantic_records[:9])
        if len(episodic_records) > 0:
            episodic_memories_str = "\n".join(f"- {_safe_dump_str(record.detail)}" for record in episodic_records[:9])        

        history_string += f'\n### Related Semantic Memory: {semantic_memories_str}\n### Related Episodic Memory: {episodic_memories_str}\n'

    assert retriever_type == "no-retrieval" or history_string != ""
    if retriever_type == "no-retrieval":
        prompt = answer_prompt_template.format(question_string)
    else:
        # truncate history string
        if tokenizer_backend == 'openai':
            tokens = tokenizer.encode(history_string, allowed_special={'<|endoftext|>'})
            if len(tokens) > max_retrieval_length:
                print('Truncating from {} to {}'.format(len(tokens), max_retrieval_length), flush=True)
                truncated_tokens = tokens[:max_retrieval_length]
                history_string = tokenizer.decode(truncated_tokens)
        elif tokenizer_backend == 'huggingface':
            encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=False, return_tensors="pt")
            if len(encoded_input['input_ids'][0]) > max_retrieval_length:
                print('Truncating from {} to {}'.format(len(encoded_input['input_ids'][0]), max_retrieval_length))
                encoded_input = tokenizer(history_string, max_length=max_retrieval_length, truncation=True, return_tensors="pt")
                history_string = tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)
        else:
            raise NotImplementedError
        prompt = answer_prompt_template.format(history_string, question_date_string, question_string)

    return prompt

async def multi_thread_transfer_dicts_to_memories(slot_process: SlotProcess, semantic_memory_system: FAISSMemorySystem, episodic_memory_system: FAISSMemorySystem, is_abstract: bool = False):
    semantic_records = []
    episodic_records = []

    for i in slot_process.memory_dict:
        if i['memory_type'] == 'semantic':
            semantic_records.append(semantic_memory_system.instantiate_sem_record(**i['input']))
        elif i['memory_type'] == 'episodic':
            episodic_records.append(episodic_memory_system.instantiate_epi_record(**i['input']))
    
    if is_abstract and len(episodic_records) > 0:
        await abstract_episodic_records_to_semantic_record(episodic_records, semantic_memory_system, episodic_memory_system)

    if len(semantic_records) > 0:
        try:
            semantic_memory_system.upsert_normal_records(semantic_records)
        except Exception as e:
            import traceback
            print("[ERROR] upsert_normal_records for semantic_records failed:", repr(e))
            traceback.print_exc()
    if len(episodic_records) > 0:
        try:
            episodic_memory_system.upsert_normal_records(episodic_records)
        except Exception as e:
            import traceback
            print("[ERROR] upsert_normal_records for episodic_records failed:", repr(e))
            traceback.print_exc()

async def abstract_episodic_records_to_semantic_record(epi_records: List[EpisodicRecord], semantic_memory_system: FAISSMemorySystem, episodic_memory_system: FAISSMemorySystem, consistency_threshold: float = 0.8):
    try:
        abstract_result, cidmap2semrec = await episodic_memory_system.abstract_episodic_records(epi_records, consistency_threshold)
        print(f"[Info] Number of abstracted semantic records: {len(abstract_result)}")
        semantic_memory_system.upsert_abstract_semantic_records(abstract_result, cidmap2semrec)
    except Exception as e:
        import traceback
        print("[ERROR] abstract_episodic_records_to_semantic_record failed:", repr(e))
        traceback.print_exc()

def reset_memprism_system(args):
    if "gpt" in args.model_name.lower():
        llm_backend = "openai"
    else:
        llm_backend = "vllm"

    slot_process_vllm = SlotProcess(llm_name="qwen3-4b", llm_backend="vllm", task="chat")
    slot_process_openai = SlotProcess(llm_name="gpt-4o-mini", llm_backend="openai", task="chat")
    semantic_memory_system = FAISSMemorySystem(memory_type="semantic", llm_model=args.model_alias, llm_backend=llm_backend)
    episodic_memory_system = FAISSMemorySystem(memory_type="episodic", llm_model=args.model_alias, llm_backend=llm_backend)

    '''slot_process = SlotProcess(task="chat")
    semantic_memory_system = FAISSMemorySystem(llm_backend="openai")
    episodic_memory_system = FAISSMemorySystem(llm_backend="openai")'''

    return slot_process_vllm, slot_process_openai, semantic_memory_system, episodic_memory_system

def get_related_information_by_query(query: str, slot_process: SlotProcess, working_slots: List[WorkingSlot], semantic_memory_system: FAISSMemorySystem, episodic_memory_system: FAISSMemorySystem, limit: int = 40):
    semantic_query_results = semantic_memory_system.query(query, limit=limit, threshold=0.5)
    episodic_query_results = episodic_memory_system.query(query, limit=limit)

    keyword_prompt = f"Given a query: \n{query}\nExtract the 5 keywords that can represent the main idea of the query in a space-separated format. For example, if the query is 'What are the health benefits of regular exercise?', the keywords could be 'health benefits exercise'. You also can use phrases as keywords. Please only output the keywords without any additional explanation."
    kwargs = {
        'model': "gpt-4o-mini",
        'messages':[
            {"role": "user", "content": keyword_prompt}
        ],
        'n': 1,
        'temperature': 0,
        'max_tokens': 100,
        }
    client = OpenAI(
        api_key=args.openai_key,
        base_url=args.openai_base_url,
    )
    key_words = chat_completions_with_backoff(client, **kwargs).choices[0].message.content.strip()
    slots_query_results = slot_process.query(query, working_slots, key_words=key_words.split(), limit=limit, use_svd=True, embed_func=semantic_memory_system.vector_store._embed)

    semantic_records = [record for score, record in semantic_query_results]
    episodic_records = [record for score, record in episodic_query_results]
    slots = [slot for score, slot in slots_query_results]
    session_ids = []
    total_session_ids = []
    
    for slot in slots:
        session_id = slot.attachments.get("session_ids").get("items", [])[0]
        if session_id is not None:
            session_ids.append(session_id)

    for slot in working_slots:
        session_id = slot.attachments.get("session_ids").get("items", [])[0]
        if session_id is not None:
            total_session_ids.append(session_id)
    
    # duplicate removal
    session_ids = list(set(session_ids))
    total_session_ids = list(set(total_session_ids))

    print(f"[Info] Total session ids in working slots: {len(total_session_ids)}, retrieved session ids: {len(session_ids)}")
    
    return semantic_records, episodic_records, session_ids


@backoff.on_exception(backoff.constant, (openai.RateLimitError), 
                      interval=5)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def main(args):
    # setup
    if args.openai_organization:
        openai.organization = args.openai_organization
    client = OpenAI(
        api_key=args.openai_key,
        base_url=args.openai_base_url,
    )

    slot_process_vllm, slot_process_openai, semantic_memory_system, episodic_memory_system = reset_memprism_system(args)

    try:
        in_data = json.load(open(args.in_file))
    except:
        in_data = [json.loads(line) for line in open(args.in_file).readlines()]

    if args.resume_file != "none":
        try:
            result_data = json.load(open(args.resume_file))
            already_done_question_ids = [entry['question_id'] for entry in result_data]
        except:
            result_data = [json.loads(line) for line in open(args.resume_file).readlines()]
            already_done_question_ids = [entry['question_id'] for entry in result_data]


    in_file_tmp = args.in_file.split('/')[-1]
    if args.merge_key_expansion_into_value is not None and args.merge_key_expansion_into_value != 'none':
        out_file = args.out_dir + '/' + in_file_tmp + '_testlog_top{}context_{}format_useronly{}_factexpansion{}_{}'.format(args.topk_context, args.history_format, args.useronly, args.merge_key_expansion_into_value, datetime.now().strftime("%Y%m%d-%H%M"))
    else:
        out_file = args.out_dir + '/' + in_file_tmp + '_testlog_top{}context_{}format_useronly{}_{}'.format(args.topk_context, args.history_format, args.useronly, datetime.now().strftime("%Y%m%d-%H%M"))
    if args.out_file_suffix.strip() != "":
        out_file += args.out_file_suffix
    if args.resume_file == "none":
        out_f = open(out_file, 'w')
    else:
        out_f = open(args.resume_file, 'a')

    # inference
    model2maxlength = {
        'gpt-4o': 128000,
        'gpt-4o-2024-08-06': 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        'meta-llama/Meta-Llama-3.1-8B-Instruct': 128000,
        'meta-llama/Meta-Llama-3.1-70B-Instruct': 128000,
        'microsoft/Phi-3-medium-128k-instruct': 120000,
        'microsoft/Phi-3.5-mini-instruct': 120000,
        'microsoft/phi-4': 16000,
        'mistral-7b-instruct-v0.2': 32000,
        'mistral-7b-instruct-v0.3': 32000,
        'In2Training/FILM-7B': 32000,
        'qwen3-4b': 32000,
    }
    model_max_length = model2maxlength[args.model_name]
    if 'gpt-4' in args.model_name.lower()  or 'gpt-3.5' in args.model_name.lower():
        tokenizer = tiktoken.get_encoding('o200k_base')
        tokenizer_backend = 'openai'
    else:
        tokenizer = AutoTokenizer.from_pretrained("/hpc_stor03/sjtu_home/zijian.wang/MemPrism/.cache/Qwen3-4B")
        tokenizer_backend = 'huggingface'

    total_prompt_tokens, total_completion_tokens = 0, 0
    for entry in tqdm(in_data):

        if entry['question_id'] in already_done_question_ids:
            continue

        # Ttruncate the retrieval part of the prompt such that the context length never exceeds
        gen_length = args.gen_length
        if gen_length is None:
            gen_length = 500 if not args.cot else 800
        max_retrieval_length = model_max_length - gen_length - 1000

        if args.con == 'true':
            prompt = prepare_prompt(entry, args.retriever_type, args.topk_context, args.useronly=='true',
                                    args.history_format, args.cot=='true', 
                                    tokenizer=tokenizer, tokenizer_backend=tokenizer_backend, max_retrieval_length=max_retrieval_length,
                                    merge_key_expansion_into_value=args.merge_key_expansion_into_value, slot_process_vllm=slot_process_vllm, slot_process_openai=slot_process_openai,
                                    semantic_memory_system=semantic_memory_system, episodic_memory_system=episodic_memory_system,
                                    con=True, con_client=client, con_model=args.model_name)
        else:
            prompt = prepare_prompt(entry, args.retriever_type, args.topk_context, args.useronly=='true',
                                    args.history_format, args.cot=='true', 
                                    tokenizer=tokenizer, tokenizer_backend=tokenizer_backend, max_retrieval_length=max_retrieval_length,
                                    merge_key_expansion_into_value=args.merge_key_expansion_into_value, slot_process_vllm=slot_process_vllm, slot_process_openai=slot_process_openai,
                                    semantic_memory_system=semantic_memory_system, episodic_memory_system=episodic_memory_system,)

        # reset MemPrism system after each example
        slot_process_vllm, slot_process_openai, semantic_memory_system, episodic_memory_system = reset_memprism_system(args)

        try:
            print(json.dumps({'question_id': entry['question_id'], 'question': entry['question'], 'answer': entry['answer']}, indent=4), flush=True)
            
            kwargs = {
                'model': args.model_name,
                'messages':[
                    {"role": "user", "content": prompt}
                ],
                'n': 1,
                'temperature': 0,
                'max_tokens': gen_length,
            }
            completion = chat_completions_with_backoff(client,**kwargs) 
            answer = completion.choices[0].message.content.strip()

            total_prompt_tokens += completion.usage.prompt_tokens
            total_completion_tokens += completion.usage.completion_tokens
            print(json.dumps({'hypothesis': answer}), flush=True)
            print(json.dumps({'question_id': entry['question_id'], 'hypothesis': answer}), file=out_f, flush=True)
        except Exception as e:
            print('One exception captured', repr(e))
            continue

    print('Total prompt tokens:', total_prompt_tokens)
    print('Total completion tokens:', total_completion_tokens)
    out_f.close()
    

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    main(args)
