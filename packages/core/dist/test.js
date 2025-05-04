"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const perplexity_1 = require("./detectors/perplexity");
async function main() {
    const detector = new perplexity_1.PerplexityDetector();
    await detector.initialize();
    const text = "This is some sample text to test";
    const score = await detector.detect(text);
    console.log(`AI probability: ${score}`);
}
main().catch(console.error);
