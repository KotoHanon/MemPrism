import pandas as pd
# set seed
import random
import numpy as np
random.seed(42)
np.random.seed(42)

pd.options.display.max_columns = 100
from models import LiteLLMClient, AMemClient, VLLMOpenAIClient, MemAlphaClient, AyumuClient, Mem0Client, VLLMClient
import argparse
import json
import numpy as np
from data_pipelines import Mem1Pipeline, Mem0Pipeline, AyumuPipeline, MemAlphaPipeline, model_estimated_match
import sys
try:
    sys.path.append("..")
    from train.rollout.env.webshop.webshop_manager import WebShopEnvManager
except Exception as e:
    print(f"Error importing WebShopEnvManager: {e}")
from tqdm import tqdm
import logging
import hashlib
from datetime import datetime
import os
import asyncio
from memory.memory_system.utils import setup_logger
from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from typing import List, Dict, Any
from itertools import repeat

# Set up logging
log_filename = datetime.now().strftime("train_%Y%m%d_%H%M%S.log")
logger = setup_logger("agent_output", log_path=os.path.join("inference/log/agent_output/", log_filename) ,level=logging.INFO)

# turn off logger
#logging.getLogger().setLevel(logging.WARNING)

########################################################
#  utils for reading the test data
########################################################
def read_nq_search_data(data_file):
    """
    Reads and returns the test data from the NQ search dataset.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the test data
    """
    file_path = data_file
    df = pd.read_parquet(file_path)
    size = len(df)
    # we only want 1000 rows
    frac = 1000 / size
    df = df.sample(frac=frac, random_state=42)
    
    # add hash to df
    df['hash'] = df['prompt'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())

    return df

def read_webshop_data(data_file):
    file_path = data_file
    df = pd.read_parquet(file_path)
    env_manager = WebShopEnvManager()
    df['hash'] = df['prompt'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    return df, env_manager

# JSON serialization helper
def json_serialize_helper(obj):
    """Helper function to make objects JSON serializable"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM loop with OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use: 'openai'")
    parser.add_argument("--use_ayumu", action="store_true", default=False,
                        help="Use Ayumu client")
    parser.add_argument("--use_amem", action="store_true", default=False,
                        help="Use Agentic Memory client")
    parser.add_argument("--use_mem0", action="store_true", default=False,
                        help="Use mem0 inference style")
    parser.add_argument("--use_mem1", action="store_true", default=False,
                        help="Use mem1 inference style")
    parser.add_argument("--use_litellm", action="store_true", default=False,
                        help="Use LiteLLM client")
    parser.add_argument("--use_memalpha", action="store_true", default=False,
                        help="Use MemAlpha client")
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="Use VLLM OpenAI client")
    parser.add_argument("--use_local_model", action="store_true", default=False,
                        help="Use VLLM OpenAI client")
    parser.add_argument("--use_graph", action="store_true", default=False,
                        help="Use graph memory in Mem0Client")
    parser.add_argument("--abstract_memories", action="store_true", default=False,
                        help="Use abstract memories (only for ayumu)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Maximum number of parallel workers")
    parser.add_argument("--resume_file", type=str, default=None,
                        help="Output file name")
    parser.add_argument("--data_file", type=str, default="Mem1/data/websearch_multi_3/test.parquet",
                        help="Data file to use")
    parser.add_argument("--task_type", type=str, default="rag", choices=["rag", "websearch", "webshop"], 
                        help="Task type")
    args = parser.parse_args()
    
    reconstruction_dicts = []

    if args.use_ayumu:
        inference_type = "ayumu"
    elif args.use_mem1:
        inference_type = "mem1"
    elif args.use_amem:
        inference_type = "amem"
    elif args.use_mem0:
        inference_type = "mem0"
    elif args.use_memalpha:
        inference_type = "mem-alpha"
    else:
        inference_type = "normal"

    assert not(not(args.use_ayumu) and args.abstract_memories), "Abstract memories can only be used with Ayumu"
    assert not(not(args.use_mem0) and args.use_graph), "Graph memory can only be used with Mem0Client"

    if args.resume_file:
        file_path = args.resume_file
        print(f"Resuming from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                reconstruction_dicts.append(json.loads(line))
        print(f"Loaded {len(reconstruction_dicts)} reconstruction dicts from {file_path}")
    else:
        file_path = f'{args.task_type}_train_reconstruction_dicts_{args.model.replace("/", "_")}.jsonl'

    # Read the test data
    if args.task_type == "rag" or args.task_type == "websearch":
        train_data = read_nq_search_data(args.data_file)
    elif args.task_type == "webshop":
        train_data, env_manager = read_webshop_data(args.data_file)
    original_len = len(train_data)

    if len(reconstruction_dicts) > 0:
        all_hashes = set()
        for row in reconstruction_dicts:
            all_hashes.add(row['hash'])
        train_data = train_data[~train_data['hash'].isin(all_hashes)]
        print(f"Filtered {len(train_data)} rows from {original_len} rows")

    if args.use_mem1:
        assert not args.use_amem, "Cannot use Agentic memory while mem1 style inference is on"


    # Run the LLM loop for each row in the test data
    reconstruction_dicts = []

    all_hashes = set()
    for row in reconstruction_dicts:
        all_hashes.add(row['hash'])

    import concurrent.futures
    import threading
    
    # Create a thread-safe list for results
    results_lock = threading.Lock()
    
    def process_row(func_args):
        index, row, client, model = func_args
        client.reset()
        try:
            prompt = row["prompt"][0]["content"]
            if args.use_mem0:
                pipeline = Mem0Pipeline(client)
            elif args.use_memalpha:
                pipeline = MemAlphaPipeline(client)
            else:
                pipeline = Mem1Pipeline(client, inference_type=inference_type)
            answer, results_dict = pipeline.run_llm_loop(prompt)
            logger.info(f"Generated answer: {answer}, Golden answer: {row['reward_model']['ground_truth']}")

            if "multi" in args.data_file:
                answers = str(answer).split(";")
                ground_truths = row['reward_model']['ground_truth']['target']
                exact_match = 0
                for idx, gt in enumerate(ground_truths):

                    gt = gt[0]
                    try:
                        if str(answers[idx]).lower().strip() in str(gt).lower().strip() or str(gt).lower().strip() in str(answers[idx]).lower().strip():
                            exact_match += 1
                        else:
                            exact_match += 0
                    except Exception as e:
                        exact_match = 0
                        break
                if exact_match == len(ground_truths):
                    print(f"Test {index} passed")
                else:
                    print(f"Test {index} failed")
            else:
                if str(answer).lower().strip() in str(row['reward_model']['ground_truth']).lower().strip():
                    print(f"Test {index} passed")
                    exact_match = True
                else:
                    print(f"Test {index} failed")
                    exact_match = False
            results_dict["index"] = index
            results_dict["hash"] = row["hash"]
            if "multi" in args.data_file:
                results_dict['Golden_answer'] = row['reward_model']['ground_truth']['target']
            else:
                results_dict['Golden_answer'] = row['golden_answers']
            results_dict['Exact_match'] = exact_match

            if client.has_memory:
                if inference_type in ["mem1", "amem"]:
                    results_dict["memories"] = client.memories
            
            try:
                if "multi" in args.data_file:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['reward_model']['ground_truth']['target'], prompt, client)
                else:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['golden_answers'], prompt, client)
            except Exception as e:
                results_dict['Model_estimated_match'] = False
            
            # Thread-safe append to the results list
            with results_lock:
                # add the entry
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            
            return index
        except Exception as e:
            # Add minimal error information to results
            result_dict = {
                    'index': index,
                    'hash': row["hash"],
                    'error': str(e),
                    'question': row.get('question', 'Unknown'),
                    'Golden_answer': row.get('golden_answers', 'Unknown'),
                    'Exact_match': False,
                    'Model_estimated_match': False
                }
            with results_lock:
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            return index

    def process_row_for_ayumu(func_args):
        index, row, client, model = func_args
        client.reset()
        try:
            prompt = row["prompt"][0]["content"]
            slots = []
            pipeline = AyumuPipeline(client, inference_type=inference_type, slots=slots, abstract_memories=args.abstract_memories)
            answer, results_dict = pipeline.run_llm_loop(prompt, model=model)
            logger.info(f"Generated answer: {answer}, Golden answer: {row['reward_model']['ground_truth']}")

            if "multi" in args.data_file:
                answers = str(answer).split(";")
                ground_truths = row['reward_model']['ground_truth']['target']
                exact_match = 0
                for idx, gt in enumerate(ground_truths):

                    gt = gt[0]
                    try:
                        if str(answers[idx]).lower().strip() in str(gt).lower().strip() or str(gt).lower().strip() in str(answers[idx]).lower().strip():
                            exact_match += 1
                        else:
                            exact_match += 0
                    except Exception as e:
                        exact_match = 0
                        break
                if exact_match == len(ground_truths):
                    print(f"Test {index} passed")
                else:
                    print(f"Test {index} failed")
            else:
                if str(answer).lower().strip() in str(row['reward_model']['ground_truth']).lower().strip():
                    print(f"Test {index} passed")
                    exact_match = True
                else:
                    print(f"Test {index} failed")
                    exact_match = False
            results_dict["index"] = index
            results_dict["hash"] = row["hash"]
            if "multi" in args.data_file:
                results_dict['Golden_answer'] = row['reward_model']['ground_truth']['target']
            else:
                results_dict['Golden_answer'] = row['golden_answers']
            results_dict['Exact_match'] = exact_match
            
            try:
                if "multi" in args.data_file:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['reward_model']['ground_truth']['target'], prompt, client)
                else:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['golden_answers'], prompt, client)
            except Exception as e:
                results_dict['Model_estimated_match'] = False
            
            # Thread-safe append to the results list
            with results_lock:
                # add the entry
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            
            return index
        except Exception as e:
            results_dict = {
                    'index': index,
                    'hash': row["hash"],
                    'error': str(e),
                    'question': row.get('question', 'Unknown'),
                    'Golden_answer': row.get('golden_answers', 'Unknown'),
                    'Exact_match': False,
                    'Model_estimated_match': False
                }
            with results_lock:
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            return index
    
    max_workers = args.max_workers
    # Use ThreadPoolExecutor to process rows in parallel
    if args.use_amem:
        # we must run in a single thread
        # otherwise chromadb will clash
        llm_client = AMemClient(args.use_local_model)
        row_data = [(index, row, llm_client, args.model) for index, row in train_data.iterrows()]
        for row in tqdm(row_data):
            process_row(row)

    elif args.use_memalpha:
        llm_client = MemAlphaClient("qwen3-4b-think-FC")
        row_data = [(index, row, llm_client, args.model) for index, row in train_data.iterrows()]
        for row in tqdm(row_data):
            process_row(row)

    elif args.use_ayumu:
        llm_client = AyumuClient(args.model, args.use_local_model)
        row_data = [(index, row, llm_client, args.model) for index, row in train_data.iterrows()]
        for batch in chunks(row_data, max_workers):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(tqdm(executor.map(process_row_for_ayumu, batch), total=len(batch)))
                working_slots = llm_client.slot_process.slot_container.values()
                num_slots = len(working_slots)
                # filter and route
                print(f"[Info] Filtering and routing {num_slots} slots")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(tqdm(executor.map(
                        llm_client.slot_process.multi_thread_filter_and_route_slot,
                        working_slots,
                    ),
                    total=num_slots
                    ))
                routed_slots = llm_client.slot_process.routed_slot_container
                num_routed_slots = len(routed_slots)
                print(f"[Info] Transferring memories from {num_routed_slots} slots to memory systems")
                # generate memories in multi-threaded way
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(tqdm(executor.map(
                        llm_client.slot_process.multi_thread_transfer_slot_to_memory,
                        routed_slots,
                    ),
                    total=num_routed_slots
                    ))
                # transfer memories to records
                asyncio.run(llm_client.multi_thread_transfer_dicts_to_memories(is_abstract=args.abstract_memories))
            except Exception as e:
                print(f"Error processing batch: {e}")

            print(f"[Info] Semantic memory size: {llm_client.semantic_memory_system.size}")
            print(f"[Info] Episodic memory size: {llm_client.episodic_memory_system.size}")
            
    else:
        if args.use_mem1:
            llm_client = VLLMOpenAIClient()
        elif args.use_mem0:
            llm_client = Mem0Client(args.use_local_model, use_graph=args.use_graph)
        elif args.use_vllm:
            llm_client = VLLMClient()
        else:
            llm_client = LiteLLMClient()
        # otherwise we can use parallel workers
        row_data = [(index, row, llm_client, args.model) for index, row in train_data.iterrows()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_row, row_data), total=len(row_data)))