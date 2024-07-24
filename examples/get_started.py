import instruct_qa

from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

import pandas as pd
import glob
import os

import string
import time
from importlib import reload

model = load_model("meta-llama/Meta-Llama-3.1-8B-Instruct", weights_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
collection = load_collection("dpr_wiki_collection")
index = load_index("dpr-nq-multi-hnsw")
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
prompt_template = load_template("qa")

path = '/mnt/nfs/shared/mmlu/test/'

all_files = glob.glob(os.path.join(path, "*.csv"))
mmlu_qs =  pd.concat((pd.read_csv(f, names=['question', 'a', 'b', 'c', 'd', 'correct']) for f in all_files), ignore_index=True)
print(len(mmlu_qs))

while True:
    try:
        input("Ready to launch, please hit ENTER")
        reload(instruct_qa)
        from instruct_qa.response_runner import ResponseRunner
        timings = {}
        t1 = time.time()

        for queries in range(1):
            queries_df = mmlu_qs 
            queries = [str(x) for x in queries_df.apply(lambda x: f'Answer the following question: {x.question} The possible answers are : A) {x.a}; B) {x.b}; C) {x.c}; D) {x.d}. No further questions allowed. Please answer only using one of the letters A, B, C, or D.', axis=1)]
            
            runner = ResponseRunner(
                model=model,
                retriever=retriever,
                document_collection=collection,
                prompt_template=prompt_template,
                queries=queries,
                batch_size=int(input("batch size:")),
                timings=timings,
                use_rag=False
            )

            responses = runner()
            print(sum([1 if x == y else 0 for (x, y) in zip([r["response"][:5] for r in responses], list(queries_df.correct))]), "/", len(queries))
            print("")
            print("elasped time", str(time.time() - t1))
    except Exception as e:
        print("attempt failed", e)
