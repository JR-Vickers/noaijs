import os
import json
import asyncio
import random
import openai
import anthropic
import torch
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

class ModelGenerator(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.output_dir = Path(f"packages/training/src/data/raw/ai/{model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "samples.json"
        self.existing_samples = self._load_existing_samples()
        
    def _load_existing_samples(self) -> List[Dict]:
        if self.output_file.exists() and self.output_file.stat().st_size > 0:
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return []

    def _get_prompts(self) -> List[str]:
        with open("packages/training/src/data/prompts.json", 'r') as f:
            prompts_data = json.load(f)
            return [prompt for category in prompts_data.values() for prompt in category]

    @abstractmethod
    async def generate_sample(self, prompt: str, temperature: float) -> Dict:
        pass

    async def generate_samples(self, total_samples: int = 50):
        prompts = self._get_prompts()
        samples_needed = total_samples - len(self.existing_samples)
        if samples_needed <= 0:
            print(f"Already have {len(self.existing_samples)} samples. No more needed.")
            return False

        with tqdm(total=samples_needed, desc=f"Generating {self.model_name} samples") as pbar:
            prompt_idx = 0
            while samples_needed > 0:
                prompt = prompts[prompt_idx % len(prompts)]
                temperature = random.uniform(0.7, 1.0)
                result = await self.generate_sample(prompt, temperature)
                if result is not None:
                    self.existing_samples.append(result)
                    with open(self.output_file, 'w') as f:
                        json.dump(self.existing_samples, f, indent=2)
                    pbar.update(1)
                    samples_needed -= 1
                prompt_idx += 1
                await asyncio.sleep(3)  # Wait 3 seconds to respect rate limit
        return samples_needed > 0  # Return whether more samples are needed

class GPT35Generator(ModelGenerator):
    def __init__(self):
        super().__init__("gpt3.5")
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_sample(self, prompt: str, temperature: float) -> Dict:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            return {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": response.choices[0].message.content
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None
        
class GPT41Generator(ModelGenerator):
    def __init__(self):
        super().__init__("gpt4.1")
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_sample(self, prompt: str, temperature: float) -> Dict:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            return {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": response.choices[0].message.content
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

class ClaudeGenerator(ModelGenerator):
    def __init__(self):
        super().__init__("claude")
        self.client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def generate_sample(self, prompt: str, temperature: float) -> Dict:
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": response.content[0].text if response.content else ""
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None
        
    from llama_cpp import Llama

class LlamaGenerator(ModelGenerator):
    def __init__(self):
        super().__init__("llama-3.1-8b")
        model_dir = "/Users/jarrett/.llama/checkpoints/Llama3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def generate_sample(self, prompt: str, temperature: float) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()
        def sync_gen():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            text = await loop.run_in_executor(None, sync_gen)
            return {
                "prompt": prompt,
                "model": self.model_name,
                "temperature": temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text": text
            }
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

async def main():
    generators = [GPT35Generator(), GPT41Generator(), ClaudeGenerator()]
    total_samples = 150
    
    while True:
        tasks = []
        for generator in generators:
            tasks.append(generator.generate_samples(total_samples))
        
        results = await asyncio.gather(*tasks)
        if not any(results):  # If all generators are done
            break
        
        await asyncio.sleep(1)  # Small delay between cycles

if __name__ == "__main__":
    asyncio.run(main())