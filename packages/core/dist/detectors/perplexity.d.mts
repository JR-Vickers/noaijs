interface TokenizerOutput {
    input_ids: number[];
    attention_mask: number[];
}
interface ModelOutput {
    logits: number[][];
}
declare module '@xenova/transformers' {
    interface Pipeline {
        (text: string, options: any): Promise<ModelOutput>;
    }
    interface AutoTokenizer {
        encode(text: string): Promise<TokenizerOutput>;
    }
}
export declare class PerplexityDetector {
    private smallModel;
    private tokenizer;
    private isInitialized;
    constructor();
    init(): Promise<void>;
    private getPerplexity;
    detect(text: string): Promise<number>;
}
export {};
