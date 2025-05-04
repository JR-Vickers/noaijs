"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.PerplexityDetector = void 0;
const tf = __importStar(require("@tensorflow/tfjs"));
class PerplexityDetector {
    constructor() {
        this.maxSequenceLength = 512;
        this.unkToken = 0;
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
    preprocessText(text) {
        // Simple word-level tokenization
        const tokens = text.toLowerCase().split(/\s+/).map(word => this.vocab.get(word) || this.unkToken);
        return tokens.slice(0, this.maxSequenceLength);
    }
    async getPerplexity(text) {
        if (!this.model) {
            throw new Error('Model not initialized');
        }
        const tokens = this.preprocessText(text);
        if (tokens.length < 2)
            return 0;
        let totalLogProb = 0;
        let count = 0;
        for (let i = 0; i < tokens.length - 1; i++) {
            const input = tf.tensor2d([tokens.slice(0, i + 1)], [1, i + 1]);
            const target = tokens[i + 1];
            const predictions = this.model.predict(input);
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
    async detect(text) {
        const perplexity = await this.getPerplexity(text);
        const normalizedScore = 1 - Math.min(perplexity / 100, 1);
        return normalizedScore;
    }
}
exports.PerplexityDetector = PerplexityDetector;
// Example usage:
const detector = new PerplexityDetector();
detector.initialize().then(() => {
    detector.detect("the cat sat").then(prob => {
        console.log("AI probability:", prob);
    });
});
