# NoAI.js (WIP)

This will be a simple JS widget that detects and rejects AI-generated text in real time.

<a href="https://warpcast.com/balajis.eth/0x34042e98" target="_blank">Link to original idea.</a>

## Quickstart

```bash
git clone https://github.com/JR-Vickers/noaijs.git
cd noaijs
npm install
npm run build
```

## Usage

```js
import { PerplexityDetector } from 'noaijs';
const detector = new PerplexityDetector();
await detector.initialize();
const score = await detector.detect('your text here');
```

## Model Weights

Model files are not included. Download them with:
```bash
python scripts/download_models.py
```
or see [models/README.md](models/README.md) for details.

## Features

- Real-time AI text detection
- Paste-penalty logic
- Pluggable detection engines

## License

MIT