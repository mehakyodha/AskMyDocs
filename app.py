import streamlit as st
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import requests


st.set_page_config(
    page_title="AskMyDocs",
    page_icon="📚",
    layout="wide"
)


st.title("📚 AskMyDocs")
st.write("Hybrid RAG: Semantic Search + BM25 Keyword Search")

st.divider()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def generate_answer(question, retrieved_chunks):

    evidence = ""

    for i, chunk in enumerate(retrieved_chunks):

        evidence += (
            f"\n[Source {i + 1}]\n"
            f"File: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    prompt = f"""
    You are AskMyDocs, a document question-answering assistant.

    Answer the user's question using ONLY the evidence provided below.

    User question:
    {question}

    Evidence:
    {evidence}

    Instructions:
    - Read the evidence carefully.
    - Give a direct and clear answer.
    - Do not use information outside the evidence.
    - If the answer is present in the evidence, answer it.
    - Add citations such as [Source 1] after factual statements.
    - If the answer is genuinely not present in the evidence, say:
    "I couldn't find sufficient information in the uploaded documents."

    Answer:
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]

def validate_citations(answer, retrieved_chunks):
    """
    Check whether citations in the answer refer to
    valid retrieved sources.
    """

    import re

    citations = re.findall(r"\[Source (\d+)\]", answer)

    if not citations:
        return False, "No citations found in the answer."

    valid_sources = set(
        str(i + 1)
        for i in range(len(retrieved_chunks))
    )

    invalid_citations = [
        citation
        for citation in citations
        if citation not in valid_sources
    ]

    if invalid_citations:
        return False, (
            f"Invalid citation(s) found: "
            f"{', '.join('[Source ' + c + ']' for c in invalid_citations)}"
        )

    return True, "All citations are valid."

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf_document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "text": text,
                "page": page_number + 1
            })

    pdf_document.close()

    return pages


def create_chunks(text, chunk_size=400, overlap=75):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) uploaded successfully!"
    )

    all_chunks = []

    # -----------------------------
    # 1. Extract and chunk PDFs
    # -----------------------------

    for uploaded_file in uploaded_files:

        pages = extract_text_from_pdf(uploaded_file)

        for page_data in pages:

            chunks = create_chunks(
                page_data["text"]
            )

            for chunk in chunks:

                all_chunks.append({
                    "text": chunk,
                    "source": uploaded_file.name,
                    "page": page_data["page"]
                })


    st.write(
        f"📦 Total chunks: **{len(all_chunks)}**"
    )


    if all_chunks:

        # -----------------------------
        # 2. Create embeddings
        # -----------------------------

        st.subheader("🧠 Creating Vector Database...")

        model = load_embedding_model()

        texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype("float32")


        dimension = embeddings.shape[1]

        vector_index = faiss.IndexFlatIP(dimension)

        vector_index.add(embeddings)


        # -----------------------------
        # 3. Create BM25 index
        # -----------------------------

        tokenized_chunks = [
            text.lower().split()
            for text in texts
        ]

        bm25 = BM25Okapi(tokenized_chunks)


        st.success("✅ Vector + BM25 indexes created!")


        st.divider()


        # -----------------------------
        # 4. Ask question
        # -----------------------------

        question = st.text_input(
            "🔎 Ask a question about your PDFs:"
        )


        if question:

            # -----------------------------
            # Vector Search
            # -----------------------------

            question_embedding = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            question_embedding = question_embedding.astype(
                "float32"
            )


            vector_scores, vector_indices = vector_index.search(
                question_embedding,
                min(5, len(all_chunks))
            )


            # -----------------------------
            # BM25 Search
            # -----------------------------

            tokenized_question = question.lower().split()

            bm25_scores = bm25.get_scores(
                tokenized_question
            )


            bm25_indices = np.argsort(
                bm25_scores
            )[::-1][:min(5, len(all_chunks))]


            # -----------------------------
            # Reciprocal Rank Fusion (RRF)
            # -----------------------------

            rrf_scores = {}

            k = 60

            # Vector search contribution
            for rank, idx in enumerate(vector_indices[0]):

                idx = int(idx)

                score = 1 / (k + rank + 1)

                rrf_scores[idx] = rrf_scores.get(idx, 0) + score


            # BM25 contribution
            for rank, idx in enumerate(bm25_indices):

                idx = int(idx)

                score = 1 / (k + rank + 1)

                rrf_scores[idx] = rrf_scores.get(idx, 0) + score


            # Sort by combined RRF score
            combined_indices = sorted(
                rrf_scores,
                key=rrf_scores.get,
                reverse=True
            )


            # -----------------------------
            # Cross-Encoder Re-Ranking
            # -----------------------------

            st.subheader("🔄 Cross-Encoder Re-Ranking")

            reranker = load_reranker()

            #Take the top RRF candidates
            candidate_indices = combined_indices[:10]

            pairs = []

            for idx in candidate_indices:

                pairs.append([
                    question,
                    all_chunks[idx]["text"]
                ])


            # Calculate relevance scores
            reranker_scores = reranker.predict(pairs)


            # Sort candidates by cross-encoder score
            reranked_results = sorted(
                zip(candidate_indices, reranker_scores),
                key=lambda x: x[1],
                reverse=True
            )


            # Keep top 3
            final_results = reranked_results[:3]
            
            retrieved_chunks = []

            for index_number, score in final_results:

                retrieved_chunks.append(
                    all_chunks[index_number]
                )


            st.divider()

            st.subheader("🤖 AskMyDocs Answer")

            with st.spinner("Generating answer with local SLM..."):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

                is_valid, citation_message = validate_citations(
                    answer,
                    retrieved_chunks
                )

                if is_valid:
                    st.success("✅ Citation validation passed")
                    st.write(answer)
                else:
                    st.warning(f"⚠️ Citation validation failed: {citation_message}")
                    st.write(answer)


            st.success(
                f"Selected {len(final_results)} most relevant chunks"
            )


            # -----------------------------
            # Display Final Evidence
            # -----------------------------

            st.subheader("📚 Final Retrieved Evidence")


            for rank, (index_number, score) in enumerate(
                            final_results
            ):

                chunk = all_chunks[index_number]

                st.markdown(
                    f"### Result {rank + 1}"
                )

                st.write(
                    chunk["text"]
                )

                st.caption(
                    f"📄 {chunk['source']} | "
                    f"Page {chunk['page']} | "
                    f"Reranker Score: {score:.4f}"
                )            