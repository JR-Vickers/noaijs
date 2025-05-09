import torch
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
from .perplexity import pseudo_perplexity  # import your function

class PerplexityGapDetector:
    def __init__(self, human_model_dir, detection_model_dir):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Load human model
        self.human_tokenizer = DistilBertTokenizerFast.from_pretrained(human_model_dir)
        self.human_model = DistilBertForMaskedLM.from_pretrained(human_model_dir).to(self.device)
        # Load detection model
        self.detection_tokenizer = DistilBertTokenizerFast.from_pretrained(detection_model_dir)
        self.detection_model = DistilBertForMaskedLM.from_pretrained(detection_model_dir).to(self.device)

    def score(self, text):
        human_ppl = pseudo_perplexity(text, self.human_tokenizer, self.human_model)
        detection_ppl = pseudo_perplexity(text, self.detection_tokenizer, self.detection_model)
        gap = human_ppl - detection_ppl
        return {
            "human_ppl": human_ppl,
            "detection_ppl": detection_ppl,
            "gap": gap
        }