# Omesh

Omesh is an educational platform for students, parents, teachers, and school staff. Its long-term goal is to bring school communication, learning resources, and an AI study tutor into one platform.

## Product vision

- A school community space for announcements, news, and posts.
- Accounts and role-based access for students, parents, teachers, and administrators.
- Carefully controlled access to student profiles and grades through approved school systems.
- A study platform with an AI tutor grounded in GCE Advanced and Ordinary Level textbooks.
- An offline-capable tutor so students can use it when internet access is unreliable.

## Learning and build roadmap

We will build in small stages, understanding each part before adding the next:

1. **Project and Python foundations** — Git, Python modules, functions, types, configuration, errors, and readable code.
2. **Local AI tutor** — connect to Ollama, understand model inputs and outputs, and handle unavailable models cleanly.
3. **Textbook answers with sources** — extract and organize permitted textbook content, retrieve relevant passages, cite pages, and evaluate citation accuracy.
4. **Tutor interface** — create a small web app for asking questions and viewing grounded answers.
5. **Accounts and permissions** — add authentication and enforce student, parent, teacher, and administrator access in the backend.
6. **School data integration** — use sample data first, then plan an approved and audited integration with the school's systems.
7. **Community features** — add announcements and posts with appropriate publishing and moderation controls.
8. **Offline use and deployment** — choose a practical local model and delivery approach, then package, deploy, monitor, and maintain the system.

The order may change as we learn, but each stage should produce a working, understandable piece of Omesh.

## First milestone

Build a tutor that accepts a question about one selected textbook, finds relevant textbook passages, answers from those passages, and shows source and page references that can be checked. Start with a small amount of content whose use is permitted. Do not treat a generated citation as correct unless it can be traced to the source.

## Notes-backed RAG prototype

The study notes in `Documents/NOTES` are Omesh's first knowledge library. Omesh reads each topic PDF or scan in place, extracts text page by page, and stores searchable passages and embedding vectors in a local SQLite database at `data/notes.sqlite3`. The index retains the topic folder, source filename, and page number so answers can point back to the notes. The original PDFs and images remain in the NOTES folder.

The retriever combines SQLite full-text keyword search with semantic vector search from a local Ollama embedding model. It combines both result rankings before sending passages to the local chat model. The chat prompt requires answers to use the retrieved passages and cite their source labels. By default, indexing and questions use `nomic-embed-text`; pull that model once with `ollama pull nomic-embed-text`. Use the same embedding model when indexing and asking questions.

Scanned pages need OCR. PDF ingestion uses Poppler's `pdfinfo` and `pdftoppm` plus Tesseract OCR. HEIC-only topics also need `heif-convert`. These tools can be found on `PATH` or supplied through `OMESH_PDFINFO`, `OMESH_PDFTOPPM`, `OMESH_TESSERACT`, and `OMESH_HEIF_CONVERT`.

Start with the small Phase Equilibrium pilot:

    .\.venv\Scripts\python.exe notes_rag.py ingest --topic "CH 7 PHASE EQUILIBRUM"
    .\.venv\Scripts\python.exe notes_rag.py ask "Explain dynamic equilibrium" --topic "CH 7 PHASE EQUILIBRUM"

Use `notes_rag.py ingest --all` to index every topic folder after the pilot is ready. Set `OMESH_NOTES_DIR` if the NOTES folder is in a different location, or `OMESH_EMBED_MODEL` to select another embedding model. The local SQLite database is ignored by Git.

## Current starting point

`test_ollama.py` is the initial experiment that sends one question to the local Ollama HTTP API using `qwen3:4b`. `notes_rag.py` is the first notes-grounded command-line tutor prototype; it still needs OCR and the embedding model before the scanned notes can be indexed.

## Working principles

- Build one understandable feature at a time.
- Keep secrets, private student data, and local environments out of source control.
- Use sample data while developing; connect real school records only with the school's approval and a defined access policy.
- Verify tutor answers and citations against their source material.
- Record design decisions and explain the trade-offs as the project grows.
