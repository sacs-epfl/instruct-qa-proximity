from instruct_qa.collections.utils import load_collection
from instruct_qa.retrieval.utils import load_retriever, load_index
from instruct_qa.prompt.utils import load_template
from instruct_qa.generation.utils import load_model
from instruct_qa.response_runner import ResponseRunner

import string
import time

print("started load_collection", time.time())
collection = load_collection("dpr_wiki_collection")
print("started load_index", time.time())
index = load_index("dpr-nq-multi-hnsw")
print("started load_retriever", time.time())
retriever = load_retriever("facebook-dpr-question_encoder-multiset-base", index)
print("started load_model", time.time())
model = load_model("flan-t5-xxl")
print("started load_template", time.time())
prompt_template = load_template("qa")
print("ram all good", time.time())

megaq = [
    ["what is haleys comet"],
    ["give some information about the letter " + str(letter) for letter in string.ascii_lowercase],
    ["what is love"]
]

for queries in megaq:

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
