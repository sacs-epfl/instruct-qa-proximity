from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

class ProbabilityGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, sentence, searchlist, k=100):
        self.model.eval()
        input_ids = self.tokenizer(sentence, return_tensors='pt').input_ids.cuda()

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits


        last_token_logits = logits[0, -1, :]
        probabilities = F.softmax(last_token_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probabilities, k)

        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

        return [x for x in top_k_tokens if x in searchlist]