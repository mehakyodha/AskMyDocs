import sys
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# =========================================================
# CI EVALUATION DATA
# =========================================================

evaluation_data = [
    {
        "question": "What are the three types of machine learning?",
        "contexts": [
            "There are three common types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."
        ],
        "expected_terms": [
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning"
        ]
    },
    {
        "question": "What does supervised learning use?",
        "contexts": [
            "Supervised learning uses labeled data."
        ],
        "expected_terms": [
            "labeled data"
        ]
    },
    {
        "question": "What is natural language processing?",
        "contexts": [
            "Natural language processing (NLP) is a branch of AI that helps computers process and understand human language."
        ],
        "expected_terms": [
            "natural language processing",
            "human language"
        ]
    },
    {
        "question": "How does reinforcement learning learn?",
        "contexts": [
            "Reinforcement learning learns through interaction with an environment and feedback in the form of rewards or penalties."
        ],
        "expected_terms": [
            "interaction",
            "environment",
            "rewards",
            "penalties"
        ]
    }
]


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# RUN RETRIEVAL EVALUATION
# =========================================================

passed_tests = 0
total_tests = len(evaluation_data)


for test_number, item in enumerate(
    evaluation_data,
    start=1
):

    question = item["question"]

    contexts = item["contexts"]

    expected_terms = item["expected_terms"]


    print("\n-----------------------------------")
    print(f"Test {test_number}: {question}")
    print("-----------------------------------")


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    embeddings = model.encode(
        contexts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )


    # =====================================================
    # FAISS VECTOR SEARCH
    # =====================================================

    dimension = embeddings.shape[1]

    vector_index = faiss.IndexFlatIP(
        dimension
    )

    vector_index.add(
        embeddings
    )


    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )


    vector_scores, vector_indices = (
        vector_index.search(
            question_embedding,
            1
        )
    )


    vector_context = contexts[
        int(vector_indices[0][0])
    ]


    # =====================================================
    # BM25 SEARCH
    # =====================================================

    tokenized_contexts = [
        context.lower().split()
        for context in contexts
    ]

    bm25 = BM25Okapi(
        tokenized_contexts
    )

    tokenized_question = (
        question.lower().split()
    )

    bm25_scores = bm25.get_scores(
        tokenized_question
    )

    bm25_index = int(
        np.argmax(bm25_scores)
    )

    bm25_context = contexts[
        bm25_index
    ]


    # =====================================================
    # COMBINE RETRIEVED CONTEXT
    # =====================================================

    retrieved_context = (
        vector_context
        + " "
        + bm25_context
    ).lower()


    # =====================================================
    # CHECK EXPECTED TERMS
    # =====================================================

    matched_terms = [
        term
        for term in expected_terms
        if term.lower() in retrieved_context
    ]


    score = (
        len(matched_terms)
        / len(expected_terms)
    )


    print(
        f"Retrieval score: {score:.2f}"
    )

    print(
        f"Matched terms: "
        f"{len(matched_terms)}/"
        f"{len(expected_terms)}"
    )


    # =====================================================
    # CI THRESHOLD
    # =====================================================

    if score >= 0.75:

        print("✅ PASS")

        passed_tests += 1

    else:

        print("❌ FAIL")


# =========================================================
# FINAL CI RESULT
# =========================================================

overall_score = (
    passed_tests
    / total_tests
)


print("\n===================================")
print("       AskMyDocs CI Evaluation")
print("===================================")

print(
    f"\nTests passed: "
    f"{passed_tests}/{total_tests}"
)

print(
    f"Overall score: "
    f"{overall_score:.2f}"
)


# =========================================================
# CI GATE
# =========================================================

if overall_score >= 0.75:

    print(
        "\n✅ CI GATE PASSED"
    )

    print(
        "RAG retrieval quality meets "
        "the required threshold."
    )

    sys.exit(0)

else:

    print(
        "\n❌ CI GATE FAILED"
    )

    print(
        "RAG retrieval quality is below "
        "the required threshold."
    )

    sys.exit(1)