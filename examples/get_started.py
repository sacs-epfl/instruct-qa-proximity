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
from importlib import reload

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
while True:
    try:
        input("Ready to launch, please hit ENTER")
        reload(instruct_qa)
        from instruct_qa.response_runner import ResponseRunner
        
        all_files = glob.glob(os.path.join(mmlu_path, "*.csv"))
        index = int(input("index of file:"))
        if(index >= 0):
            print(all_files[index])
            mmlu_qs = pd.concat((pd.read_csv(f, names=['question', 'a', 'b', 'c', 'd', 'correct']) for f in all_files[index:index+1]), ignore_index=True)
        else:
            print("loading all")
            mmlu_qs = pd.concat((pd.read_csv(f, names=['question', 'a', 'b', 'c', 'd', 'correct']) for f in all_files), ignore_index=True)
        print(len(mmlu_qs))

        timings = {}
        t1 = time.time()


        fraction = float(input("fraction:"))
        if fraction >= 0.99:
            queries_df = mmlu_qs
        else:
            queries_df = mmlu_qs.sample(frac=fraction, random_state=999)
        queries = [str(x) for x in queries_df.apply(lambda x: f'{x.question} The possible answers are : A) {x.a}; B) {x.b}; C) {x.c}; D) {x.d}. No further questions allowed. Only the first character of your answer will be considered. Please answer output a single character among the letters A, B, C, or D.', axis=1)]
        
        rag_size = int(input("rag size:"))

        runner = ResponseRunner(
            model=model,
            retriever=retriever,
            document_collection=collection,
            prompt_template=prompt_template,
            queries=queries,
            batch_size=int(input("batch size:")),
            timings=timings,
            use_rag=rag_size != 0, 
            k=rag_size if rag_size > 0 else 5 
        )
        
        # get 30 most likely tokens and find which one does best
        responses, trags = runner.get_probas(30)
        best_calls = [find_best_tok(toks) for toks in responses]

        print(best_calls, list(queries_df.correct))
        print("rag time", np.mean(trags))
        print("total time", time.time() - t1)

        print(sum([1 if x == y else 0 for (x, y) in zip(best_calls, list(queries_df.correct))]), "/", len(queries))

    except Exception:
        print(traceback.format_exc())
