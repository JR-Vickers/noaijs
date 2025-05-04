import torch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def pseudo_perplexity(text, tokenizer, model, max_length=512):
    """
    Compute pseudo-perplexity for a single text using a masked language model.
    Args:
        text (str): Input text.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace masked language model.
        max_length (int): Max sequence length (default 512).
    Returns:
        float: Pseudo-perplexity score.
    """

    # Tokenize and get input IDs
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc.input_ids[0].to(device)
    n_tokens = input_ids.size(0)

    log_probs = []
    # Loop over each token (skip [CLS] and [SEP] if present)
    for i in range(1, n_tokens - 1):
        masked = input_ids.clone()
        masked[i] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(masked.unsqueeze(0))
            logits = outputs.logits
            softmax = torch.nn.functional.softmax(logits[0, i], dim=-1)
            prob = softmax[input_ids[i]].item()
            log_probs.append(np.log(prob + 1e-12))  # add epsilon for stability

    avg_neg_log_prob = -np.mean(log_probs)
    ppl = np.exp(avg_neg_log_prob)
    return ppl