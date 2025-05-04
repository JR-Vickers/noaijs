import * as tf from '@tensorflow/tfjs';

export class PerplexityDetector {
  private model: tf.LayersModel | null;
  private readonly maxSequenceLength = 512;
  private readonly vocab: Map<string, number>;
  private readonly unkToken = 0;

  constructor() {
    this.model = null;
    this.vocab = new Map();
    // Initialize with basic vocabulary
    'the and of to a in is that for it as with was he on by at an are this from or had not be they his which have one you were all their can said there use do how if will each tell does set three want air well also play small end put home read hand port large spell add even land here must big high such follow act why ask men change went light kind off need house picture try us again animal point mother world near build self earth father head stand own page should country found answer school grow study still learn plant cover food sun four between state keep eye never last let thought city tree cross farm hard start might story saw far sea draw left late run don\'t while press close night real life few north open seem together next white children begin got walk example ease paper group always music those both mark often letter until mile river car feet care second book carry took science eat room friend began idea fish mountain stop once base hear horse cut sure watch color face wood main enough plain girl usual young ready above ever red list though feel talk bird soon body dog family direct pose leave song measure door product black short numeral class wind question happen complete ship area half rock order fire south problem piece told knew pass since top whole king space heard best hour better true during hundred five remember step early hold west ground interest reach fast verb sing listen six table travel less morning ten simple several vowel toward war lay against pattern slow center love person money serve appear road map rain rule govern pull cold notice voice unit power town fine certain fly fall lead cry dark machine note wait plan figure star box noun field rest correct able pound done beauty drive stood contain front teach week final gave green oh quick develop ocean warm free minute strong special mind behind clear tail produce fact street inch multiply nothing course stay wheel full force blue object decide surface deep moon island foot system busy test record boat common gold possible plane stead dry wonder laugh thousand ago ran check game shape equate hot miss brought heat snow tire bring yes distant fill east paint language among grand'.split(' ').forEach((word, i) => {
      this.vocab.set(word, i + 1); // Start from 1, 0 is UNK
    });
  }

  async initialize() {
    // Load a pre-trained language model (GPT-2 small)
    this.model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/gpt2/model.json');
  }

  private preprocessText(text: string): number[] {
    // Simple word-level tokenization
    const tokens = text.toLowerCase().split(/\s+/).map(word => 
      this.vocab.get(word) || this.unkToken
    );
    return tokens.slice(0, this.maxSequenceLength);
  }

  private async getPerplexity(text: string): Promise<number> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const tokens = this.preprocessText(text);
    if (tokens.length < 2) return 0;

    let totalLogProb = 0;
    let count = 0;

    for (let i = 0; i < tokens.length - 1; i++) {
      const input = tf.tensor2d([tokens.slice(0, i + 1)], [1, i + 1]);
      const target = tokens[i + 1];

      const predictions = this.model.predict(input) as tf.Tensor;
      const probs = await predictions.softmax().data();

      const tokenProb = probs[target];
      totalLogProb += Math.log(tokenProb);
      count++;

      input.dispose();
      predictions.dispose();
    }

    const avgNegLogProb = -totalLogProb / count;
    return Math.exp(avgNegLogProb);
  }

  async detect(text: string): Promise<number> {
    const perplexity = await this.getPerplexity(text);
    const normalizedScore = 1 - Math.min(perplexity / 100, 1);
    return normalizedScore;
  }
}

// Example usage:
const detector = new PerplexityDetector();
detector.initialize().then(() => {
  detector.detect("the cat sat").then(prob => {
    console.log("AI probability:", prob);
  });
}); 