export declare class PerplexityDetector {
    private smallModel;
    private tokenizer;
    private isInitialized;
    constructor();
    init(): Promise<void>;
    private getPerplexity;
    detect(text: string): Promise<number>;
}
