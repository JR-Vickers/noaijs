import { PerplexityDetector } from './detectors/perplexity';

async function main() {
  const detector = new PerplexityDetector();
  await detector.initialize();
  
  const text = "This is some sample text to test";
  const score = await detector.detect(text);
  console.log(`AI probability: ${score}`);
}

main().catch(console.error);