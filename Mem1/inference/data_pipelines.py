from typing_extensions import Literal
from typing import List, Dict, Any
import requests
import re
import random
from openai import OpenAI
import os
import asyncio
import threading
import itertools
from memory.memory_system.working_slot import WorkingSlot
TOP_K = 3
SEARCH_URL = "http://127.0.0.1:8013/retrieve"
MAX_ITERATION = 6

########################################################
#  utils for search
########################################################
def batch_search(query):
    def search_tool(queries):
        payload = {
            "queries": queries,
            "topk": TOP_K,
            "return_scores": True
        }
        return requests.post(SEARCH_URL, json=payload).json()

    def passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):   
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
    
    results = search_tool([query])['result']
    return [passages2string(result) for result in results][0]


########################################################
#  utils for determining the action
########################################################
def act(response: str):
    if "<search>" in response and "</search>" in response:
        # regex to find the search query
        search_query = re.findall(r'<search>(.*?)</search>', response, re.DOTALL)
        # extract the search query string
        search_query = search_query[0].strip()
        search_results = batch_search(search_query)
        return {"type": "search", "content": search_results, "query": search_query}
    elif "<answer>" in response and "</answer>" in response:
        # regex to find the answer
        answer = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
        # extract the answer string
        answer = answer[0].strip()
        return {"type": "answer", "content": answer}
    else:
        return None

def extract_internal_state(response: str, tag: str):
                    
    # regex to find the think part
    if f"<{tag}>" in response and f"</{tag}>" in response:
        pattern = f"<{tag}>(.*?)</{tag}>"
        istate = re.findall(pattern, response, re.DOTALL)
        # extract the think string
        istate = istate[0].strip()
        return f"<{tag}>{istate}</{tag}>"
    else:
        return None


def model_estimated_match(answer, golden_answer, question, _):
    prompt = f"""
    Your goal is to determine if a model's answer answers the question based on the golden answer.
    The question is: {question}
    The model's answer is: {answer}
    The golden answer is: {golden_answer}
    Output your answer as 0 or 1, where 0 means the model's answer does not align with the golden answer and 1 means the model's answer aligns with the golden answer. Output only the number, no other text.
    """

    ## uncomment this on to use gpt-4o-mini to estimate the match    
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.0,
    #     max_tokens=1
    # )
    
    # return int(response.choices[0].message.content.strip())
    return 1

########################################################
#  pipelines
########################################################
from abc import ABC, abstractmethod

class Pipeline(ABC):
    def __init__(self, llm_client):
        self.llm_client = llm_client

    @abstractmethod
    def run_llm_loop(self, prompt):
        pass


class Mem1Pipeline(Pipeline):
    def __init__(self, llm_client, inference_type: Literal["normal", "amem" "mem1"]):
        super().__init__(llm_client)
        self.inference_type = inference_type
        
    def run_llm_loop(self, prompt, model="openai/gpt-4o-mini"):
        use_mem1 = self.inference_type == "mem1"
        is_compress_memory = self.inference_type in ["amem", "mem1"]

        cur_response = ""
        if use_mem1:
            # if mem1 model, we separate the prompt and cur_obs
            # such tht cur_obs only stores the responses
            cur_obs = ""
        else:
            # for other models, cur_obs stores the entire conversation trajectory
            cur_obs = prompt
        iteration_cnt = 0
        # Initialize results tracking dictionary
        results_dict = {"q": prompt}

        while iteration_cnt < MAX_ITERATION:
            # make summary and update the observation
            if use_mem1:
                cur_response = self.llm_client.make_completion(prompt, cur_obs, model=model, is_last_turn=iteration_cnt == MAX_ITERATION - 1)
            else:
                cur_response = self.llm_client.generate_response(cur_obs, model=model)

            # for the current implementation, use <think></think> for storing the internal state
            internal_state = extract_internal_state(cur_response, tag="think")
            
            if not is_compress_memory:
                memory = cur_obs[len(prompt):]
            else:
                memory = cur_obs
            if self.llm_client.has_memory and memory:
                self.llm_client.memory_system.add_note(memory)
            
            if internal_state:
                # Store summary in results dictionary
                results_dict[f"t{iteration_cnt}"] = internal_state
            else:
                results_dict[f"t{iteration_cnt}"] = ""
            
            if is_compress_memory:
                # clear all previous states by setting the cur_obs to empty
                cur_obs = prompt
            
            action_dict = act(cur_response)

            num_turns_left = MAX_ITERATION - iteration_cnt - 1
            if num_turns_left > 1:
                hint = f"[HINT]You have {num_turns_left} turns left.[/HINT]"
            else:
                hint = f"[HINT]You have {num_turns_left} turn left. You must answer the question now.[/HINT]"

            if action_dict is None:
                return None, results_dict
            elif action_dict["type"] == "search":
                search_results = action_dict["content"]
                search_results = f"<information>\n{hint}\n{search_results}\n</information>"
                # Store search query in results dictionary
                results_dict[f"r{iteration_cnt}"] = cur_response
                # Store information in results dictionary
                if iteration_cnt == MAX_ITERATION - 1:
                    results_dict[f"i{iteration_cnt}"] = ""
                else:
                    results_dict[f"i{iteration_cnt}"] = search_results
                next_obs = cur_obs + cur_response + search_results
            elif action_dict["type"] == "answer":
                # Store final answer in results dictionary
                results_dict[f"r{iteration_cnt}"] = cur_response
                return action_dict["content"], results_dict
            cur_obs = next_obs

            iteration_cnt += 1
        
        return None, results_dict

class Mem0Pipeline(Pipeline):
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.mem_lock = threading.Lock()
        self.thread_name = threading.current_thread().name
        
    def run_llm_loop(self, prompt, model="openai/gpt-4o-mini"):
        is_compress_memory = True

        cur_response = ""
        query_text = ""
        memory = ""
        cur_obs = prompt
        iteration_cnt = 0
        # Initialize results tracking dictionary
        results_dict = {"q": prompt}

        while iteration_cnt < MAX_ITERATION:
            # make summary and update the observation
            cur_response = self.llm_client.generate_response(query_text, cur_obs, thread_name=self.thread_name, model=model)

            # for the current implementation, use <think></think> for storing the internal state
            internal_state = extract_internal_state(cur_response, tag="think")
            print(f"[Debug] Iteration {iteration_cnt} Response: {cur_response}")
            
            action_dict = act(cur_response)
            if self.llm_client.has_memory and memory:
                try:
                    with self.mem_lock:
                        self.llm_client.memory_system.add(memory, user_id=f"agent_{self.thread_name}", infer=False)
                except Exception as e:
                    print(f"[Warning] Failed to add memory: {e}") 
            
            if internal_state:
                # Store summary in results dictionary
                results_dict[f"t{iteration_cnt}"] = internal_state
            else:
                results_dict[f"t{iteration_cnt}"] = ""

            cur_obs = prompt
            
            num_turns_left = MAX_ITERATION - iteration_cnt - 1
            if num_turns_left > 1:
                hint = f"[HINT]You have {num_turns_left} turns left.[/HINT]"
            else:
                hint = f"[HINT]You have {num_turns_left} turn left. You must answer the question now.[/HINT]"

            if action_dict is None:
                return None, results_dict
            elif action_dict["type"] == "search":
                search_results = action_dict["content"]
                search_results = f"<information>\n{hint}\n{search_results}\n</information>"
                # Store search query in results dictionary
                results_dict[f"r{iteration_cnt}"] = cur_response
                # Store information in results dictionary
                if iteration_cnt == MAX_ITERATION - 1:
                    results_dict[f"i{iteration_cnt}"] = ""
                else:
                    results_dict[f"i{iteration_cnt}"] = search_results
                memory = search_results
                next_obs = cur_obs + cur_response + search_results
                query_text = action_dict["query"]
            elif action_dict["type"] == "answer":
                # Store final answer in results dictionary
                query_text = ""
                results_dict[f"r{iteration_cnt}"] = cur_response
                return action_dict["content"], results_dict
            cur_obs = next_obs

            iteration_cnt += 1
        
        return None, results_dict    

class AyumuPipeline(Pipeline):
    def __init__(self, llm_client, inference_type: Literal["normal", "amem", "mem1", "ayumu"], slots, abstract_memories: bool = False):
        super().__init__(llm_client)
        self.inference_type = inference_type
        self.abstract_memories = abstract_memories
        self.search_query_cache = set()
        self.slots = slots

    def run_llm_loop(self, prompt, model):
        use_ayumu = True
        is_compress_memory = True
        is_collect_slot = True

        cur_response = ""
        query_text = ""
        cur_obs = prompt
        iteration_cnt = 0
        # Initialize results tracking dictionary
        results_dict = {"q": prompt}

        while iteration_cnt < MAX_ITERATION:
            cur_response = self.llm_client.generate_response(query_text=query_text, slots=self.slots, prompt=cur_obs, model=model)

            # for the current implementation, use <think></think> for storing the internal state
            internal_state = extract_internal_state(cur_response, tag="think")
            print(f"[Debug] Iteration {iteration_cnt} Response: {cur_response}")
            
            memory = cur_obs
            
            action_dict = act(cur_response)
            if self.llm_client.has_memory and memory: 
                if action_dict["type"] == "search":
                    search_query = action_dict["query"].lower().strip()
                    search_results = action_dict["content"]
                    cur_turn_result = cur_response + search_results
                    if search_query not in self.search_query_cache:
                        self.search_query_cache.add(search_query)
                        self.slots.extend(asyncio.run(self.llm_client.transfer_context_to_slots(context=memory)))
            
            if internal_state:
                # Store summary in results dictionary
                results_dict[f"t{iteration_cnt}"] = internal_state
            else:
                results_dict[f"t{iteration_cnt}"] = ""
            
            if is_compress_memory:
                # clear all previous states by setting the cur_obs to empty
                cur_obs = prompt
            

            num_turns_left = MAX_ITERATION - iteration_cnt - 1
            if num_turns_left > 1:
                hint = f"[HINT]You have {num_turns_left} turns left.[/HINT]"
            else:
                hint = f"[HINT]You have {num_turns_left} turn left. You must answer the question now.[/HINT]"

            if action_dict is None:
                return None, results_dict
            elif action_dict["type"] == "search":
                search_results = action_dict["content"]
                search_results = f"<information>\n{hint}\n{search_results}\n</information>"
                # Store search query in results dictionary
                results_dict[f"r{iteration_cnt}"] = cur_response
                # Store information in results dictionary
                if iteration_cnt == MAX_ITERATION - 1:
                    results_dict[f"i{iteration_cnt}"] = ""
                else:
                    results_dict[f"i{iteration_cnt}"] = search_results
                next_obs = cur_obs + cur_response + search_results
                query_text = action_dict["query"]
            elif action_dict["type"] == "answer":
                # Store final answer in results dictionary
                query_text = ""
                results_dict[f"r{iteration_cnt}"] = cur_response
                return action_dict["content"], results_dict
            cur_obs = next_obs

            iteration_cnt += 1
        
        return None, results_dict
