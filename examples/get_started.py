import instruct_qa

from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

import pandas as pd
import glob
import os
import numpy as np
import traceback 
import string
import time
import itertools
from importlib import reload
import pickle

import proximipy
import random


def expand_question(q):
    return [x + q for x in 
        [
            "Please answer the following question : ", 
            "I'm stuck on this question, could you please give me a hand? The question is : ",
            "Answer this now please : ",
            ""
        ]
    ]

model = load_model("meta-llama/Meta-Llama-3.1-8B-Instruct", weights_path="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=15)
collection = load_collection("dpr_wiki_collection")
index = load_index("dpr-nq-multi-hnsw", index_path = "/mnt/nfs/home/randl/datasets/index.dpr")
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
prompt_template = load_template("llama_chat_qa")

mmlu_path = '/mnt/nfs/shared/mmlu/test/'

def find_best_tok(lis):
    for elem in lis:
        if elem.upper() in [" A", " B", " C", " D"]:
            return elem[1].upper()
    return None

print("LLM LOADING DONE")

#params for experiments
index = 18 # we work with econometrics
fraction = 1.0 # use all questions of the topic

seeds_range = list(range(42, 42 + 5))
cache_capacity_range  = [10, 50, 100, 200, 300]
cache_tolerance_range = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0]
db_k_range = [5]
rag_size = 5
should_expand = True #whether questions should be copied/modified to create new ones using prefixes

results = {}

for paramlist in itertools.product(seeds_range, cache_capacity_range, cache_tolerance_range, db_k_range):
    (seed, cache_capacity, cache_tolerance, db_k) = paramlist

    try:
        all_files = glob.glob(os.path.join(mmlu_path, "*.csv"))
        
        if(index >= 0):
            print(all_files[index])
            mmlu_qs = pd.concat((pd.read_csv(f, names=['question', 'a', 'b', 'c', 'd', 'correct']) for f in all_files[index:index+1]), ignore_index=True)
        else:
            print("loading all")
            mmlu_qs = pd.concat((pd.read_csv(f, names=['question', 'a', 'b', 'c', 'd', 'correct']) for f in all_files), ignore_index=True)

        cache = proximipy.FifoCache(cache_capacity, cache_tolerance)
        reload(instruct_qa)
        from instruct_qa.response_runner import ResponseRunner

        timings = {}
        t1 = time.time()


        if fraction >= 0.99:
            queries_df = mmlu_qs
        else:
            queries_df = mmlu_qs.sample(frac=fraction, random_state=seed)
        queries = [str(x) for x in queries_df.apply(lambda x: f'{x.question} The possible answers are : A) {x.a}; B) {x.b}; C) {x.c}; D) {x.d}. No further questions allowed. Only the first character of your answer will be considered. Please answer output a single character among the letters A, B, C, or D.', axis=1)]
        answers = list(queries_df.correct)

        if should_expand:
            queries = [x for query in queries for x in expand_question(query)]
            answers = [answer for answer in answers for i in range(4)] # repeat 4 times every answer

        to_shuffle = list(zip(queries, answers))
        random.seed(seed)
        random.shuffle(to_shuffle)
        queries, answers = zip(*to_shuffle)
        del to_shuffle

        batch_size= 32

        runner = ResponseRunner(
            model=model,
            retriever=retriever,
            document_collection=collection,
            prompt_template=prompt_template,
            queries=queries,
            batch_size=batch_size,
            timings=timings,
            use_rag=rag_size != 0, 
            k=rag_size if rag_size > 0 else 5,
            cache=cache,
            cache_depth=0,
            db_k = db_k
        )
        
        # get 30 most likely tokens and find which one does best
        responses, trags = runner.get_probas(30)
        with open(f"/mnt/nfs/home/randl/llm-rag/logs/fifo-timings-s{seed}cap{cache_capacity}tol{cache_tolerance}rerank{db_k}.pkl", "wb") as f:
            pickle.dump(trags, f)
        best_calls = [find_best_tok(toks) for toks in responses]

        # print(best_calls, answers)
        # print("rag time", np.mean(trags))
        # print("total time", time.time() - t1)
        # print("hit rate", runner.cache_hit / len(responses))
        # print(sum([1 if x == y else 0 for (x, y) in zip(best_calls, answers)]), "/", len(answers))
        accuracy = sum([1 if x == y else 0 for (x, y) in zip(best_calls, answers)]) / len(answers)

        results[paramlist] = {"hit rate" : runner.cache_hit / len(responses), "accuracy" : accuracy}
        print(paramlist, results[paramlist])
        

    except Exception:
        print(traceback.format_exc())

with open("/mnt/nfs/home/randl/llm-rag/logs/acc_hitrate.pkl", "wb") as f:
    pickle.dump(results, f)

