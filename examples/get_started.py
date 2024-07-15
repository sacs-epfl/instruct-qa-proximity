from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

import pandas as pd

collection = load_collection("dpr_wiki_collection")
index = load_index("dpr-nq-multi-hnsw")
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
model = load_model("flan-t5-xxl")
prompt_template = load_template("qa")

megaqueries = pd.read_csv('/mnt/nfs/shared/mmlu/test/high_school_world_history_test.csv', names=['question', 'a', 'b', 'c', 'd', 'correct'])

for queries in range(1):
    queries = megaqueries.iloc[:5]
    print(queries)
    queries = queries.apply(lambda x: f'This is a question about history. Here is the question : {x.question}. Please pick one of the following answers : A) {x.a} B) {x.b} C) {x.c} D) {x.d}.')
    print(queries)
    
    runner = ResponseRunner(
        model=model,
        retriever=retriever,
        document_collection=collection,
        prompt_template=prompt_template,
        queries=queries,
    )

    responses = runner()
    print(responses[0]["response"])
"""
Halley's Comet Halley's Comet or Comet Halley, officially designated 1P/Halley...
"""
