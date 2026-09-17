from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.run_config import RunConfig

from ragas.metrics import context_recall

from datasets import Dataset

from langchain_ollama import ChatOllama


# =========================
# Local Ollama LLM
# =========================

llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0
)


# =========================
# Local Embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# =========================
# Ragas Run Configuration
# =========================

run_config = RunConfig(
    timeout=300,
    max_retries=2,
    max_workers=1
)


# =========================
# Evaluation Dataset
# =========================

data = {
    "question": [
        "What are the three types of machine learning?",
        "What is natural language processing?",
        "What does supervised learning use?",
        "What does reinforcement learning learn through?"
    ],

    "answer": [
        "The three types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
        "Natural language processing is a branch of AI that helps computers process and understand human language.",
        "Supervised learning uses labeled data.",
        "Reinforcement learning learns through interaction with an environment and feedback in the form of rewards or penalties."
    ],

    "contexts": [
        [
            "There are three common types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."
        ],
        [
            "Natural language processing (NLP) is a branch of AI that helps computers process and understand human language."
        ],
        [
            "Supervised learning uses labeled data."
        ],
        [
            "Reinforcement learning learns through interaction with an environment and feedback in the form of rewards or penalties."
        ]
    ],

    "ground_truth": [
        "Supervised learning, unsupervised learning, and reinforcement learning.",
        "Natural language processing is a branch of AI that helps computers process and understand human language.",
        "Supervised learning uses labeled data.",
        "Reinforcement learning learns through interaction with an environment and feedback in the form of rewards or penalties."
    ]
}


dataset = Dataset.from_dict(data)


# =========================
# Run Ragas Evaluation
# =========================

print("\nStarting Ragas evaluation...")
print("Using local Ollama Llama 3.2 3B.")
print("Using local all-MiniLM-L6-v2 embeddings.")
print("Evaluating context recall...\n")


result = evaluate(
    dataset,
    metrics=[
        context_recall
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
    raise_exceptions=True
)


# =========================
# Display Results
# =========================

print("\n===================================")
print("      AskMyDocs RAG Evaluation")
print("===================================\n")

print("Ragas Results:")
print(result)

print("\n===================================")
print("       Evaluation Completed")
print("===================================\n")