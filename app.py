import os
import re
from pathlib import Path

import streamlit as st
import chromadb

from sentence_transformers import SentenceTransformer
from ollama import chat, ResponseError


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Company Policy RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Company Policy RAG")
st.write(
    "Ask questions about your company document using RAG."
)


# ============================================================
# PROJECT / DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "rag_database"

# Explicitly create the folder
DB_PATH.mkdir(
    parents=True,
    exist_ok=True
)

COLLECTION_NAME = "company_policy"


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# ============================================================
# GET CHROMA COLLECTION
# IMPORTANT:
# We do NOT cache the collection object.
# We always get the collection fresh by name.
# ============================================================

def get_collection():

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return client, collection


# ============================================================
# CREATE SMALL CHUNKS
# ============================================================

def create_chunks(text):
    """
    Convert a large company document into small chunks.

    Example:

    Large Document
         ↓
    Policy Section
         ↓
    Sentences
         ↓
    Small chunks
         ↓
    Embeddings
    """

    # --------------------------------------------------------
    # Normalize new lines
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.strip()


    # --------------------------------------------------------
    # Find numbered sections
    #
    # Example:
    #
    # 1. COMPANY OVERVIEW
    # 2. WORK FROM OFFICE POLICY
    # 3. LEAVE POLICY
    #
    # --------------------------------------------------------

    heading_pattern = re.compile(
        r"(?m)^\s*(\d+)\.\s+(.+?)\s*$"
    )

    matches = list(
        heading_pattern.finditer(text)
    )


    sections = []


    # --------------------------------------------------------
    # Build sections
    # --------------------------------------------------------

    if matches:

        for i, match in enumerate(matches):

            start = match.start()

            if i + 1 < len(matches):

                end = matches[i + 1].start()

            else:

                end = len(text)

            section_text = text[
                start:end
            ].strip()

            if section_text:

                heading = match.group(0).strip()

                body = section_text[
                    len(match.group(0)):
                ].strip()

                sections.append(
                    {
                        "heading": heading,
                        "body": body
                    }
                )

    else:

        # ----------------------------------------------------
        # Fallback if numbered headings are not found
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        for i, paragraph in enumerate(
            paragraphs,
            start=1
        ):

            paragraph = paragraph.strip()

            if paragraph:

                sections.append(
                    {
                        "heading": f"Section {i}",
                        "body": paragraph
                    }
                )


    # ========================================================
    # CREATE SMALL CHUNKS
    # ========================================================

    chunks = []
    metadatas = []

    MAX_CHARS = 350

    for section in sections:

        heading = section["heading"]
        body = section["body"]


        # ----------------------------------------------------
        # Split body into sentences
        # ----------------------------------------------------

        sentences = re.split(
            r"(?<=[.!?])\s+",
            body
        )

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue


            # ------------------------------------------------
            # Add sentence to current chunk
            # ------------------------------------------------

            possible_chunk = (
                current_chunk + " " + sentence
            ).strip()


            if len(possible_chunk) <= MAX_CHARS:

                current_chunk = possible_chunk

            else:

                # --------------------------------------------
                # Save previous chunk
                # --------------------------------------------

                if current_chunk:

                    final_text = (
                        f"{heading}\n"
                        f"{current_chunk}"
                    )

                    chunks.append(
                        final_text
                    )

                    metadatas.append(
                        {
                            "section": heading
                        }
                    )


                # --------------------------------------------
                # Start new chunk
                # --------------------------------------------

                current_chunk = sentence


        # ----------------------------------------------------
        # Save remaining text
        # ----------------------------------------------------

        if current_chunk:

            final_text = (
                f"{heading}\n"
                f"{current_chunk}"
            )

            chunks.append(
                final_text
            )

            metadatas.append(
                {
                    "section": heading
                }
            )


    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------

    unique_chunks = []
    unique_metadata = []

    seen = set()

    for chunk, metadata in zip(
        chunks,
        metadatas
    ):

        normalized = chunk.lower().strip()

        if normalized not in seen:

            seen.add(normalized)

            unique_chunks.append(
                chunk
            )

            unique_metadata.append(
                metadata
            )


    return (
        unique_chunks,
        unique_metadata
    )


# ============================================================
# CLEAR EXISTING DOCUMENTS
# IMPORTANT:
# We delete DOCUMENTS, not the collection itself.
# ============================================================

def clear_existing_documents(collection):

    existing = collection.get(
        include=[]
    )

    existing_ids = existing.get(
        "ids",
        []
    )

    if existing_ids:

        collection.delete(
            ids=existing_ids
        )


# ============================================================
# SESSION STATE
# ============================================================

if "document_ready" not in st.session_state:

    st.session_state.document_ready = False


if "chunk_count" not in st.session_state:

    st.session_state.chunk_count = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write("Database path:")

    st.code(
        str(DB_PATH)
    )

    st.write("Embedding model:")

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.write("Vector database:")

    st.code(
        "ChromaDB"
    )

    st.write("LLM:")

    st.code(
        "Ollama → llama3.2"
    )

    if st.session_state.document_ready:

        st.success(
            f"{st.session_state.chunk_count} "
            "chunks stored"
        )

    else:

        st.info(
            "No document processed yet."
        )


# ============================================================
# STEP 1 — DOCUMENT INPUT
# ============================================================

st.subheader(
    "📄 Step 1: Paste Your Document"
)

document = st.text_area(
    "Paste the company policy document here:",
    height=350,
    placeholder=(
        "Paste your TechNova company "
        "policy document here..."
    )
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if st.button(
    "📥 Process & Store Document",
    type="primary"
):

    if not document.strip():

        st.warning(
            "Please paste the document first."
        )

    else:

        with st.spinner(
            "Processing document..."
        ):

            # -----------------------------------------------
            # Create chunks
            # -----------------------------------------------

            chunks, metadatas = create_chunks(
                document
            )


            # -----------------------------------------------
            # Safety check
            # -----------------------------------------------

            if not chunks:

                st.error(
                    "No chunks were created."
                )

                st.stop()


            # -----------------------------------------------
            # Get collection
            # -----------------------------------------------

            client, collection = get_collection()


            # -----------------------------------------------
            # Remove OLD document chunks
            # -----------------------------------------------

            clear_existing_documents(
                collection
            )


            # -----------------------------------------------
            # Create embeddings
            # -----------------------------------------------

            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()


            # -----------------------------------------------
            # Create unique IDs
            # -----------------------------------------------

            ids = [
                f"chunk_{i}"
                for i in range(
                    len(chunks)
                )
            ]


            # -----------------------------------------------
            # Store in ChromaDB
            # -----------------------------------------------

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )


            # -----------------------------------------------
            # Create marker file
            # -----------------------------------------------

            marker_file = (
                DB_PATH /
                "database_created.txt"
            )

            marker_file.write_text(
                "ChromaDB database created successfully.\n"
                f"Total chunks: {len(chunks)}\n"
                f"Collection: {COLLECTION_NAME}\n",
                encoding="utf-8"
            )


            # -----------------------------------------------
            # Session state
            # -----------------------------------------------

            st.session_state.document_ready = True

            st.session_state.chunk_count = len(
                chunks
            )


        # ---------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------

        st.success(
            "✅ Document processed successfully!"
        )

        st.info(
            f"📦 {len(chunks)} chunks created "
            "and stored."
        )

        st.success(
            "📁 Database folder created at:"
        )

        st.code(
            str(DB_PATH)
        )


        # ---------------------------------------------------
        # SHOW CHUNKS
        # ---------------------------------------------------

        with st.expander(
            "🔎 View Created Chunks"
        ):

            for i, chunk in enumerate(
                chunks,
                start=1
            ):

                st.markdown(
                    f"**Chunk {i}**"
                )

                st.write(
                    chunk
                )

                st.divider()


# ============================================================
# STEP 2 — QUESTION
# ============================================================

st.subheader(
    "❓ Step 2: Ask a Question"
)

question = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: What are the company's working hours?"
    )
)


# ============================================================
# ASK AI
# ============================================================

if st.button(
    "🤖 Ask AI",
    type="primary"
):

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    if not st.session_state.get(
        "document_ready",
        False
    ):

        st.warning(
            "Please process the document first."
        )

        st.stop()


    # ========================================================
    # GET FRESH COLLECTION
    # ========================================================

    try:

        client, collection = get_collection()

    except Exception as e:

        st.error(
            "Could not open ChromaDB."
        )

        st.code(
            str(e)
        )

        st.stop()


    # ========================================================
    # CHECK DATABASE
    # ========================================================

    total_documents = collection.count()

    if total_documents == 0:

        st.warning(
            "The database is empty. "
            "Please process the document again."
        )

        st.stop()


    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    with st.spinner(
        "Creating question embedding..."
    ):

        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    with st.spinner(
        "Searching ChromaDB..."
    ):

        number_of_results = min(
            3,
            total_documents
        )

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=number_of_results,
            include=[
                "documents",
                "distances",
                "metadatas"
            ]
        )


    # ========================================================
    # GET RESULTS
    # ========================================================

    retrieved_documents = (
        results.get(
            "documents",
            [[]]
        )[0]
    )

    distances = (
        results.get(
            "distances",
            [[]]
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]]
        )[0]
    )


    # ========================================================
    # FILTER RESULTS
    # ========================================================

    relevant_results = []


    for doc, distance, metadata in zip(
        retrieved_documents,
        distances,
        metadatas
    ):

        # Cosine distance:
        # Smaller = more similar

        if distance <= 0.65:

            relevant_results.append(
                {
                    "document": doc,
                    "distance": distance,
                    "metadata": metadata
                }
            )


    # ========================================================
    # NO RELEVANT INFORMATION
    # ========================================================

    if not relevant_results:

        st.warning(
            "I couldn't find relevant information "
            "in the provided document."
        )

        st.caption(
            f"Best distance found: "
            f"{distances[0]:.4f}"
        )

        st.stop()


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    for item in relevant_results:

        context_parts.append(
            item["document"]
        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # SEND RETRIEVED CONTEXT TO OLLAMA
    # ========================================================

    st.subheader(
        "💬 Answer"
    )

    with st.spinner(
        "Ollama is generating the answer..."
    ):

        prompt = f"""
You are a company policy assistant.

Your job is to answer the user's question
using ONLY the retrieved context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not reproduce the entire document.
3. Do not list unrelated policies.
4. Answer only the question asked.
5. Keep the answer short and clear.
6. If the answer is not contained in the context,
   say exactly:

"I couldn't find that information in the document."

RETRIEVED CONTEXT:
------------------
{context}
------------------

USER QUESTION:
{question}

ANSWER:
"""


        try:

            response = chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise "
                            "company policy assistant."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )


            # ------------------------------------------------
            # Get response
            # ------------------------------------------------

            try:

                answer = (
                    response.message.content
                )

            except AttributeError:

                answer = (
                    response[
                        "message"
                    ][
                        "content"
                    ]
                )


            st.success(
                answer
            )


        except ResponseError as e:

            st.error(
                "Ollama returned an error."
            )

            st.code(
                str(e)
            )

            st.info(
                "Make sure Ollama is running and "
                "llama3.2 is installed."
            )


        except Exception as e:

            st.error(
                "Could not connect to Ollama."
            )

            st.code(
                str(e)
            )

            st.info(
                "Run: ollama list"
            )


    # ========================================================
    # RETRIEVED CONTEXT — HIDDEN BY DEFAULT
    # ========================================================

    with st.expander(
        "📚 View Retrieved Chunks"
    ):

        for i, item in enumerate(
            relevant_results,
            start=1
        ):

            st.markdown(
                f"### Retrieved Chunk {i}"
            )

            st.write(
                item["document"]
            )

            st.caption(
                f"Distance: "
                f"{item['distance']:.4f}"
            )

            st.caption(
                f"Section: "
                f"{item['metadata'].get('section', 'Unknown')}"
            )

            st.divider()


# ============================================================
# RAG FLOW
# ============================================================

st.divider()

st.subheader(
    "🧠 RAG Pipeline"
)

st.code(
    """
Company Document
       ↓
     Chunking
       ↓
Sentence Transformers
       ↓
    Embeddings
       ↓
    ChromaDB
       ↓
User Question
       ↓
Question Embedding
       ↓
Similarity Search
       ↓
Relevant Chunks
       ↓
     Ollama
       ↓
   Final Answer
""",
    language="text"
)