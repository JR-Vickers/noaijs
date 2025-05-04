import { PerplexityDetector } from './detectors/perplexity.mjs';
async function test() {
    const detector = new PerplexityDetector();
    await detector.init();
    // Test with human text
    const humanText = "The weather today is quite unpredictable, with sudden changes between sunshine and rain.";
    const humanProb = await detector.detect(humanText);
    console.log("Human text probability:", humanProb);
    // Test with AI text (this is GPT generated)
    const aiText = "The atmospheric conditions today exhibit significant variability, characterized by rapid alternations between solar illumination and precipitation.";
    const aiProb = await detector.detect(aiText);
    console.log("AI text probability:", aiProb);
}
test().catch(console.error);
