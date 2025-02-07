import json
import os
from pathlib import Path
import numpy as np
import requests
import re
import time

from instruct_qa.retrieval.utils import dict_values_list_to_numpy
from instruct_qa.dataset.qa import GenericQADataset
from instruct_qa.generation import ProbabilityGenerator

from tqdm import tqdm



class ResponseRunner:
    def __init__(
        self,
        model,
        retriever,
        document_collection,
        prompt_template,
        timings,
        cache,
        cache_depth,
        db_k,
        use_rag=True,
        dataset=None,
        queries=None,
        output_path=None,
        k=10,
        batch_size=1,
        logging_interval=256,
        use_hosted_retriever=False,
        hosted_retriever_url="http://10.140.16.91:42010/search",
        use_cached_retrieved_results=False,
        post_process_response=False,
    ):
        self._model = model
        self._probamodel = ProbabilityGenerator(model.model, model.tokenizer)
        self._retriever = retriever
        self._document_collection = document_collection
        self._prompt_template = prompt_template
        self.timings = timings
        self.cache = cache
        self.cache_hit = 0
        self.cache_depth = cache_depth
        self.use_rag = use_rag
        self.db_k = db_k

        # either dataset or queries should be specified, but not both
        assert (dataset is None) != (queries is None), "Either dataset or queries should be specified, but not both"
        if queries:
            dataset = GenericQADataset(queries)
        self._dataset = dataset
        self._output_path = output_path
        self._k = k
        self._batch_size = batch_size
        self._logging_interval = logging_interval
        self._use_hosted_retriever = use_hosted_retriever
        self._hosted_retriever_url = hosted_retriever_url
        self._use_cached_retrieved_results = use_cached_retrieved_results
        self._collection_name = document_collection.get_name()
        self._post_process_response = post_process_response

    def post_process_response(self, response):
        return self._model.post_process_response(response)

    def rag_call(self, batch, queries):


        ## transform text to vectors
        t1 = time.time()
        encoded = self._retriever.encode_queries(queries)
        t2 = time.time()

        # check in the cache. Todo batch search in Rust code 
        cache_res = []
        for to_search in encoded:
            cache_res.append(self.cache.find(list(to_search)))
        indices_found = [i for i in range(len(encoded)) if cache_res[i] is not None]
        indices_not_found = [i for i in range(len(encoded)) if cache_res[i] is None]
        self.cache_hit += len(indices_found)

        # retrieved indices is the cache/db returned value for all vectors in batch
        # for now, it has results from the cache and None where there was no match
        # so we then ask the DB wherever it is None
        retrieved_indices = cache_res

        if len(indices_not_found) > 0:
            # db calls for the cache misses
            missed = np.array([encoded[i] for i in indices_not_found])
            r_dict = self._retriever.retrieve(missed, k=self.db_k)["indices"]

            # update the cache and the retrieved indices
            for (r_dict_i, cache_res_i) in enumerate(indices_not_found):
                retrieved_indices[cache_res_i] = r_dict[r_dict_i]
                self.cache.insert(list(encoded[cache_res_i]), list(retrieved_indices[cache_res_i]))

        t3 = time.time()

        passages = [
            self._document_collection.get_passages_from_indices(indices)
            for indices in retrieved_indices
        ]

        t4 = time.time()

        distances = []
        for i in range(len(encoded)):
            encoded_passages = self._retriever.encode_queries(passages[i]) #(20, 768)
            encoded_query = encoded[i]
            distances.append(np.linalg.norm(encoded_passages - encoded_query, axis=1))
        distances = np.mean(distances)

        prompts = [
            self._prompt_template(
                sample=sample,
                passages=p,
            )
            for sample, p in zip(batch, passages)
        ]
        return prompts, {"avg_dist" : distances, "hit" : len(indices_not_found) < len(indices_found), "encoding" : t2 - t1, "search" : t3 - t2, "fetch_doc" : t4 - t3}

    def get_probas(self, k):
        INTERNAL_BATCH_SIZE = self._batch_size
        batches = [
            self._dataset[i:i+INTERNAL_BATCH_SIZE]
            for i in range(0, len(self._dataset), INTERNAL_BATCH_SIZE)
        ]
        ret = []
        trags = []
        for batch in batches:
            queries = self._dataset.get_queries(batch)
            if self.use_rag:
                prompts, trag = self.rag_call(batch, queries)
            else:
                prompts = [
                    self._prompt_template(
                        sample=sample,
                        passages=[{"title" : "Not Found", "text" : "No corresponding source was found."}],
                    )
                    for sample in batch
                ]
                trag = {}
                retrieved_indices = [0] * self._k
            for p in prompts: # no LLM batching but we don't care, it's not part of the measured DB query latency
                ret.append(self._probamodel(p, k))
            trags.append(trag)
        return ret, trags

    def _write_results_to_file(self, results):
        # Use pathlib to create a folder of the output path if it is not created
        # already.
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "a") as f:
            f.writelines(json.dumps(result) + "\n" for result in results)

    def recompute_embeddings(passages):
        return self._retriever.encode_queries(passages)

    def rerank(target, embeddings):
        print(target)
        print("-----")
        print(embeddings)
        return range(len(embeddings))
