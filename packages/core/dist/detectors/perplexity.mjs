import { Pipeline, AutoTokenizer } from '@xenova/transformers';
import * as tf from '@tensorflow/tfjs';
export class PerplexityDetector {
    constructor() {
        this.smallModel = null;
        this.tokenizer = null;
        this.isInitialized = false;
        // Models will be loaded in init()
    }
    async init() {
        // Load GPT-2-small for perplexity calculation
        this.smallModel = await Pipeline.from_pretrained('Xenova/gpt2', {
            quantized: true, // Use quantized model for smaller size
            revision: 'v1.0'
        });
        // Load tokenizer
        this.tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2');
        this.isInitialized = true;
    }
    async getPerplexity(text) {
        if (!this.isInitialized || !this.smallModel || !this.tokenizer) {
            throw new Error('Detector not initialized. Call init() first.');
        }
        // Tokenize input
        const encoding = await this.tokenizer.encode(text);
        const tokens = encoding.input_ids;
        // Get logits for each position
        const output = await this.smallModel(text, {
            max_new_tokens: 0,
            return_dict: true,
            output_scores: true
        });
        const logits = output.logits;
        const logProbs = [];
        // Calculate log probabilities
        for (let i = 0; i < tokens.length - 1; i++) {
            const nextToken = tokens[i + 1];
            const positionLogits = logits[i];
            // Convert logits to probabilities using softmax
            const probs = tf.softmax(tf.tensor(positionLogits));
            const nextTokenProb = probs.gather([nextToken]).dataSync()[0];
            logProbs.push(Math.log(nextTokenProb));
            probs.dispose(); // Clean up tensor
        }
        // Calculate perplexity
        const avgNegLogProb = -logProbs.reduce((a, b) => a + b, 0) / logProbs.length;
        return Math.exp(avgNegLogProb);
    }
    async detect(text) {
        const perplexity = await this.getPerplexity(text);
        // Convert perplexity to probability using logistic function
        // These thresholds need to be calibrated with real data
        const normalizedPerp = (perplexity - 10) / 5; // Center around perplexity of 10
        const prob = 1 / (1 + Math.exp(-normalizedPerp));
        return prob;
    }
}
