import wikipedia
import json
from datetime import datetime, timezone

prompts = [
    "Write a detailed explanation of how an artificial neural network works, suitable for a beginner.",
    "Explain the concept of blockchain technology in simple terms.",
    "Describe the process of photosynthesis in plants.",
    "Explain quantum computing as if you were telling a bedtime story.",
    "Describe the basics of machine learning to someone new to the field.",
    "What is deep learning and how does it differ from traditional machine learning?",
    "Explain what a genetic algorithm is and how it is used.",
    "Describe natural language processing and its applications.",
    "What is computer vision and how do machines interpret images?",
    "Explain the fundamentals of reinforcement learning.",
    "Describe what a support vector machine is and how it works.",
    "Explain the concept of a decision tree in machine learning.",
    "What is a random forest and how does it improve predictions?",
    "Describe gradient boosting and its advantages.",
    "Explain linear regression and when it is used.",
    "What is logistic regression and how does it work?",
    "Describe the k-means clustering algorithm.",
    "Explain principal component analysis and its purpose.",
    "What is a convolutional neural network and what is it used for?",
    "Describe a recurrent neural network and its applications.",
    "Explain the transformer model in machine learning.",
    "What is backpropagation and why is it important?",
    "Describe the problem of overfitting in machine learning.",
    "Explain the bias–variance tradeoff.",
    "What is cross-validation and why is it used?",
    "Describe hyperparameter optimization in machine learning.",
    "What is feature engineering and why is it important?",
    "Explain data augmentation and its benefits.",
    "What is big data and how is it managed?",
    "Describe cloud computing and its advantages.",
    "What is the Internet of Things (IoT)?",
    "Explain edge computing and its use cases.",
    "What is cybersecurity and why is it important?",
    "Describe encryption and how it protects data.",
    "What is public-key cryptography?",
    "Explain digital signatures and their role in security.",
    "What is a hash function and where is it used?",
    "Describe distributed ledger technology.",
    "What is a smart contract?",
    "Explain consensus algorithms in blockchain.",
    "What is proof of work?",
    "Describe proof of stake and how it differs from proof of work.",
    "What is a zero-knowledge proof?",
    "Explain homomorphic encryption.",
    "What is federated learning?",
    "Describe explainable artificial intelligence (XAI).",
    "What is the Turing test?",
    "Who was Alan Turing and what is his significance?",
    "Explain Moore's law.",
    "What is Shannon's information theory?"
]

topics = [
    "Artificial neural network",
    "Blockchain",
    "Photosynthesis",
    "Quantum computing",
    "Machine learning",
    "Deep learning",
    "Genetic algorithm",
    "Natural language processing",
    "Computer vision",
    "Reinforcement learning",
    "Support vector machine",
    "Decision tree",
    "Random forest",
    "Gradient boosting",
    "Linear regression",
    "Logistic regression",
    "K-means clustering",
    "Principal component analysis",
    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (machine learning model)",
    "Backpropagation",
    "Overfitting",
    "Bias–variance tradeoff",
    "Cross-validation (statistics)",
    "Hyperparameter optimization",
    "Feature engineering",
    "Data augmentation",
    "Big data",
    "Cloud computing",
    "Internet of things",
    "Edge computing",
    "Cybersecurity",
    "Encryption",
    "Public-key cryptography",
    "Digital signature",
    "Hash function",
    "Distributed ledger",
    "Smart contract",
    "Consensus algorithm",
    "Proof of work",
    "Proof of stake",
    "Zero-knowledge proof",
    "Homomorphic encryption",
    "Federated learning",
    "Explainable artificial intelligence",
    "Turing test",
    "Alan Turing",
    "Moore's law",
    "Shannon's information theory"
]

samples = []
for prompt, topic in zip(prompts, topics):
    try:
        summary = wikipedia.summary(topic, sentences=5)
        samples.append({
            "prompt": prompt,
            "model": "human",
            "temperature": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": summary
        })
    except Exception as e:
        print(f"Error fetching {topic}: {e}")

with open("../data/raw/human/technical/samples.json", "w") as f:
    json.dump(samples, f, indent=2)