import os
import re
import math
from pathlib import Path
from collections import Counter

from werkzeug.utils import secure_filename
from quart import Quart, jsonify, request, send_from_directory
from openai import AsyncOpenAI
from pypdf import PdfReader

app = Quart(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")

UPLOAD_DIR = Path(__file__).resolve().parent / "local_documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200


@app.get("/auth_setup")
async def auth_setup():
    return jsonify({
        "useLogin": False,
        "requireAccessControl": False,
        "enableUnauthenticatedAccess": True,
        "msalConfig": {
            "auth": {
                "clientId": "",
                "authority": "",
                "redirectUri": "/redirect",
                "postLogoutRedirectUri": "/redirect",
                "navigateToLoginRequestUrl": False,
            },
            "cache": {
                "cacheLocation": "localStorage",
                "storeAuthStateInCookie": False,
            },
        },
        "loginRequest": {"scopes": []},
        "tokenRequest": {"scopes": []},
    })


@app.get("/config")
async def config():
    return jsonify({
        "showMultimodalOptions": False,
        "showSemanticRankerOption": False,
        "showQueryRewritingOption": False,
        "showReasoningEffortOption": False,
        "streamingEnabled": False,
        "defaultReasoningEffort": "",
        "reasoningEffortOptions": [],
        "defaultRetrievalReasoningEffort": "minimal",
        "showVectorOption": False,
        "showUserUpload": True,
        "showLanguagePicker": False,
        "showSpeechInput": False,
        "showSpeechOutputBrowser": False,
        "showSpeechOutputAzure": False,
        "showChatHistoryBrowser": True,
        "showChatHistoryCosmos": False,
        "showAgenticRetrievalOption": False,
        "ragSearchTextEmbeddings": False,
        "ragSearchImageEmbeddings": False,
        "ragSendTextSources": True,
        "ragSendImageSources": False,
        "webSourceEnabled": False,
        "sharepointSourceEnabled": False,
    })


@app.get("/")
async def index():
    return jsonify({"status": "Local Ollama backend is running"})


@app.post("/upload")
async def upload():
    files = await request.files
    file = files.get("file")

    if file is None or not file.filename:
        return jsonify({"message": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename or Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"message": "Only PDF, TXT and MD files are supported"}), 400

    await file.save(str(UPLOAD_DIR / filename))
    return jsonify({"message": f"File uploaded successfully: {filename}"}), 200


@app.get("/list_uploaded")
async def list_uploaded():
    files = sorted(
        p.name for p in UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    return jsonify(files)


@app.post("/delete_uploaded")
async def delete_uploaded():
    body = await request.get_json()
    filename = secure_filename((body or {}).get("filename", ""))

    if not filename:
        return jsonify({"message": "Filename is required"}), 400

    destination = UPLOAD_DIR / filename
    if not destination.is_file():
        return jsonify({"message": "File not found"}), 404

    destination.unlink()
    return jsonify({"message": f"File {filename} deleted successfully"})


@app.get("/content/<path:filename>")
async def content_file(filename):
    safe_name = secure_filename(filename)
    if not safe_name or not (UPLOAD_DIR / safe_name).is_file():
        return jsonify({"error": "File not found"}), 404
    return await send_from_directory(UPLOAD_DIR, safe_name)


def split_into_chunks(text, filename, page_number):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "filename": filename,
                "page": page_number,
            })

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def read_documents():
    chunks = []

    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file():
            continue

        try:
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(str(path))

                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    chunks.extend(
                        split_into_chunks(text, path.name, page_number)
                    )

            elif path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                chunks.extend(split_into_chunks(text, path.name, 1))

        except Exception:
            app.logger.exception("Could not read document: %s", path.name)

    return chunks



def search_documents(question, chunks, limit=4):
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "whose",
        "why", "how", "the", "and", "for", "from", "with", "this",
        "that", "are", "was", "were", "is", "to", "of", "in", "on",
        "a", "an", "do", "does", "did", "tell", "me", "about",
        "please", "explain", "according", "title", "page",
        "list", "give", "show", "find", "application", "applications",
        "kya", "hai", "ka", "ki", "ke", "ko", "mein", "se", "batao",
        "samjhao", "waht", "what"
    }

    def tokens(value):
        return [
            word for word in re.findall(r"[a-zA-Z0-9]+", value.lower())
            if len(word) > 1 and word not in stop_words
        ]

    # Restrict search to the requested filename when one is provided.
    filename_match = re.search(
        r"([a-zA-Z0-9_-]+\.pdf|[a-zA-Z0-9_-]+\.txt|[a-zA-Z0-9_-]+\.md)",
        question,
        re.IGNORECASE,
    )

    if filename_match:
        requested_filename = filename_match.group(1).lower()
        matching_chunks = [
            chunk for chunk in chunks
            if chunk["filename"].lower() == requested_filename
        ]

        # Never silently answer from a different document.
        if not matching_chunks:
            return []

        chunks = matching_chunks

    # Remove the filename and common question wording before ranking.
    search_question = re.sub(
        r"[a-zA-Z0-9_-]+\.(?:pdf|txt|md)",
        " ",
        question,
        flags=re.IGNORECASE,
    )
    query_words = set(tokens(search_question))

    if not query_words:
        return []

    # Important multi-word concepts should match together where possible.
    normalized_question = re.sub(
        r"\s+", " ", search_question.lower()
    ).strip()
    query_tokens = tokens(normalized_question)
    query_phrases = []

    for size in (3, 2):
        for index in range(len(query_tokens) - size + 1):
            phrase = " ".join(query_tokens[index:index + size])
            if len(phrase) >= 8:
                query_phrases.append(phrase)

    scored = []

    for chunk in chunks:
        text = chunk["text"]
        chunk_text = text.lower()
        chunk_words = tokens(chunk_text)
        chunk_word_set = set(chunk_words)

        overlap = query_words.intersection(chunk_word_set)
        if not overlap:
            continue

        coverage = len(overlap) / len(query_words)

        # Balance term coverage and term rarity; avoid long generic chunks.
        frequency_score = sum(
            1 + math.log1p(chunk_words.count(word))
            for word in overlap
        )
        score = coverage * 5 + frequency_score / math.sqrt(
            max(len(chunk_words), 1)
        )

        # Strongly reward exact multi-word topic matches.
        for phrase in query_phrases:
            if phrase in chunk_text:
                score += 4

        # Reward a chunk that covers all meaningful query terms.
        if query_words.issubset(chunk_word_set):
            score += 5

        # Prefer slide headings and compact passages over unrelated material.
        first_line = chunk_text.splitlines()[0] if chunk_text else ""
        heading_overlap = len(query_words.intersection(set(tokens(first_line))))
        score += heading_overlap * 1.5

        scored.append((score, coverage, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    if not scored:
        return []

    # If the query is weakly related to every passage, return no evidence.
    best_score, best_coverage, _ = scored[0]
    if best_coverage < 0.25:
        return []

    results = []
    for score, coverage, chunk in scored:
        # Avoid passing unrelated passages just to fill the result limit.
        if coverage < 0.25 or score < best_score * 0.45:
            continue

        results.append(chunk)
        if len(results) >= limit:
            break

    return results
@app.post("/chat")
async def chat():
    body = await request.get_json() or {}
    messages = body.get("messages", [])

    ollama_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("system", "user", "assistant")
        and isinstance(m.get("content"), str)
    ]

    if not ollama_messages:
        return jsonify({"error": "No chat messages provided"}), 400

    latest_question = next(
        (
            message["content"]
            for message in reversed(ollama_messages)
            if message["role"] == "user"
        ),
        "",
    )

    
    try:
        all_chunks = read_documents()
        relevant_chunks = search_documents(latest_question, all_chunks)

        document_context = ""
        citations = []

        if relevant_chunks:
            context_parts = []

            for chunk in relevant_chunks:
                reference = f'{chunk["filename"]}, page {chunk["page"]}'
                citations.append(reference)
                context_parts.append(
                    f"[Source: {reference}]\n{chunk['text']}"
                )

            document_context = "\n\n".join(context_parts)

        elif all_chunks:
            document_context = (
                "No sufficiently matching passage was found in the uploaded "
                "documents. Do not invent document facts. Tell the user that "
                "the answer was not found in the uploaded files."
            )

        else:
            document_context = (
                "There is no extractable text in the uploaded documents. "
                "If the user asks about a document, explain that its text "
                "could not be retrieved. Do not claim to have read it."
            )

        system_instruction = (
            "You are a helpful assistant in a local document question-answering "
            "application. Answer using only the provided document context. "
            "If the context does not contain the answer, say so. Do not invent "
            "facts or citations. Answer in the user's language.\n\n"
            "Retrieved document context:\n"
            f"{document_context}"
        )

        system_message = next(
            (m for m in ollama_messages if m["role"] == "system"),
            None,
        )

        if system_message:
            system_message["content"] += "\n\n" + system_instruction
        else:
            ollama_messages.insert(
                0, {"role": "system", "content": system_instruction}
            )

        result = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=ollama_messages,
            stream=False,
        )

        answer = result.choices[0].message.content or ""

        if relevant_chunks:
            source_markers = []
            for chunk in relevant_chunks:
                marker = f'[{chunk["filename"]}]'
                if marker not in source_markers:
                    source_markers.append(marker)

            answer = answer.rstrip() + "\n\nSources: " + " ".join(source_markers)

        return jsonify({
            "output_text": answer,
            "context": {
                "data_points": {
                    "text": [chunk["text"] for chunk in relevant_chunks],
                    "images": [],
                    "citations": list(dict.fromkeys(citations)),
                },
                "thoughts": [],
                "followup_questions": None,
            },
            "session_state": body.get("session_state"),
        })

    except Exception:
        app.logger.exception("Local Ollama chat failed")
        return jsonify({
            "error": (
                "Local chat failed. Check that Ollama is running, "
                "the model is installed, and uploaded files are readable."
            )
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=50505, debug=False)



