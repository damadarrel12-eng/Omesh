"""Build and query a local, page-cited RAG index from the NOTES folder."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import requests


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_NOTES_DIR = Path(
    os.environ.get("OMESH_NOTES_DIR", str(Path.home() / "Documents" / "NOTES"))
).expanduser()
DEFAULT_DB_PATH = Path(
    os.environ.get("OMESH_RAG_DB", str(PROJECT_DIR / "data" / "notes.sqlite3"))
).expanduser()
DEFAULT_CHAT_MODEL = os.environ.get("OMESH_CHAT_MODEL", "qwen3:4b")
DEFAULT_OLLAMA_URL = os.environ.get("OMESH_OLLAMA_URL", "http://localhost:11434")
DEFAULT_OCR_LANGUAGE = os.environ.get("OMESH_OCR_LANGUAGE", "eng")
MIN_TEXT_CHARS = 40
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 240
WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class PipelineError(RuntimeError):
    """A readable setup or source-file error for the command-line user."""


def find_tool(env_name: str, executable: str) -> str | None:
    """Find a command, allowing an explicit executable path through the environment."""
    override = os.environ.get(env_name)
    candidates = [override] if override else [executable]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            return str(candidate_path)
    return None


def run_command(args: list[str], tool_name: str) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise PipelineError(f"Could not start {tool_name}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PipelineError(f"{tool_name} failed: {detail}")
    return result.stdout


def ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS note_chunks USING fts5(
                       topic UNINDEXED,
                       source_ref UNINDEXED,
                       page_number UNINDEXED,
                       chunk_number UNINDEXED,
                       content,
                       tokenize = 'unicode61'
                   )"""
            )
    except sqlite3.OperationalError as error:
        raise PipelineError(
            "This Python SQLite build does not include FTS5, which Omesh uses "
            "for its local notes search index."
        ) from error


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    """Split one page into overlapping search passages without losing its page link."""
    text = clean_ocr_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def get_pdf_page_count(pdfinfo: str, pdf_path: Path) -> int:
    output = run_command([pdfinfo, str(pdf_path)], "pdfinfo")
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if not match:
        raise PipelineError(f"Could not read the page count for {pdf_path.name}.")
    return int(match.group(1))


def extract_with_pypdf(pdf_path: Path) -> list[str] | None:
    """Use embedded PDF text when the optional pypdf package is already available."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return None


def ocr_pdf_page(
    pdf_path: Path,
    page_number: int,
    dpi: int,
    language: str,
    pdftoppm: str,
    tesseract: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="omesh-ocr-") as temp_name:
        prefix = Path(temp_name) / "page"
        run_command(
            [
                pdftoppm,
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-r",
                str(dpi),
                "-png",
                str(pdf_path),
                str(prefix),
            ],
            "pdftoppm",
        )
        image_path = prefix.with_suffix(".png")
        if not image_path.is_file():
            raise PipelineError(f"PDF page {page_number} was not rendered to an image.")
        return run_command(
            [tesseract, str(image_path), "stdout", "-l", language, "--psm", "6"],
            "Tesseract OCR",
        )


def read_pdf_pages(
    pdf_path: Path,
    dpi: int,
    language: str,
    pdfinfo: str,
    pdftoppm: str,
    tesseract: str | None,
) -> list[str]:
    extracted = extract_with_pypdf(pdf_path)
    page_count = len(extracted) if extracted is not None else get_pdf_page_count(pdfinfo, pdf_path)
    if page_count == 0:
        raise PipelineError(f"No pages were found in {pdf_path.name}.")

    pages = extracted if extracted is not None else [""] * page_count
    if len(pages) != page_count:
        raise PipelineError(f"The page count changed while reading {pdf_path.name}.")

    pages_needing_ocr = [
        index for index, text in enumerate(pages) if len(clean_ocr_text(text)) < MIN_TEXT_CHARS
    ]
    if pages_needing_ocr and not tesseract:
        raise PipelineError(
            "This notes PDF contains scanned pages. Omesh needs Tesseract OCR to read them. "
            "No Tesseract executable is currently available."
        )

    if tesseract:
        for index in pages_needing_ocr:
            pages[index] = ocr_pdf_page(
                pdf_path,
                index + 1,
                dpi,
                language,
                pdftoppm,
                tesseract,
            )
    return [clean_ocr_text(page) for page in pages]


def ocr_image(
    image_path: Path,
    language: str,
    tesseract: str,
    heif_convert: str | None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="omesh-image-ocr-") as temp_name:
        prepared_path = image_path
        if image_path.suffix.casefold() == ".heic":
            if not heif_convert:
                raise PipelineError(
                    f"{image_path.name} is HEIC. Set OMESH_HEIF_CONVERT to the "
                    "heif-convert executable so Omesh can prepare it for OCR."
                )
            prepared_path = Path(temp_name) / "page.png"
            run_command([heif_convert, str(image_path), str(prepared_path)], "heif-convert")
        return run_command(
            [tesseract, str(prepared_path), "stdout", "-l", language, "--psm", "6"],
            "Tesseract OCR",
        )


def store_document(
    db_path: Path,
    topic: str,
    source_ref: str,
    pages: Iterable[str],
) -> int:
    rows: list[tuple[str, str, int, int, str]] = []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_number, chunk in enumerate(split_into_chunks(page_text), start=1):
            rows.append((topic, source_ref, page_number, chunk_number, chunk))

    ensure_schema(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM note_chunks WHERE source_ref = ?", (source_ref,))
        connection.executemany(
            """INSERT INTO note_chunks
                   (topic, source_ref, page_number, chunk_number, content)
               VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        connection.commit()
    return len(rows)


def process_pdf(
    pdf_path: Path,
    topic_dir: Path,
    notes_root: Path,
    db_path: Path,
    dpi: int,
    language: str,
    pdfinfo: str,
    pdftoppm: str,
    tesseract: str | None,
) -> tuple[int, int]:
    pages = read_pdf_pages(pdf_path, dpi, language, pdfinfo, pdftoppm, tesseract)
    source_ref = pdf_path.relative_to(notes_root).as_posix()
    chunk_count = store_document(
        db_path, topic_dir.name, source_ref, pages
    )
    return len(pages), chunk_count


def natural_sort_key(path: Path) -> list[object]:
    return [
        int(piece) if piece.isdigit() else piece.casefold()
        for piece in re.split(r"(\d+)", path.name)
    ]


def process_images(
    topic_dir: Path,
    notes_root: Path,
    db_path: Path,
    language: str,
    tesseract: str | None,
    heif_convert: str | None,
) -> tuple[int, int]:
    images = sorted(
        (
            path
            for path in topic_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".heic"}
        ),
        key=natural_sort_key,
    )
    if not images:
        return 0, 0
    if not tesseract:
        raise PipelineError(
            "Image notes need Tesseract OCR. No Tesseract executable is currently available."
        )

    total_chunks = 0
    for page_number, image_path in enumerate(images, start=1):
        text = ocr_image(image_path, language, tesseract, heif_convert)
        source_ref = image_path.relative_to(notes_root).as_posix()
        total_chunks += store_document(
            db_path,
            topic_dir.name,
            source_ref,
            [text],
        )
        print(f"  Read scan {page_number}/{len(images)}: {image_path.name}")
    return len(images), total_chunks


def ingest(args: argparse.Namespace) -> None:
    notes_root = args.source_dir.resolve()
    if not notes_root.is_dir():
        raise PipelineError(f"Notes folder not found: {notes_root}")

    topic_dirs = sorted((path for path in notes_root.iterdir() if path.is_dir()), key=natural_sort_key)
    if args.topic:
        selected = [path for path in topic_dirs if path.name.casefold() == args.topic.casefold()]
        if not selected:
            available = ", ".join(path.name for path in topic_dirs)
            raise PipelineError(f"Topic folder not found: {args.topic}. Available folders: {available}")
        topic_dirs = selected

    pdfinfo = find_tool("OMESH_PDFINFO", "pdfinfo")
    pdftoppm = find_tool("OMESH_PDFTOPPM", "pdftoppm")
    tesseract = find_tool("OMESH_TESSERACT", "tesseract")
    heif_convert = find_tool("OMESH_HEIF_CONVERT", "heif-convert")
    processed_topics = 0
    for topic_dir in topic_dirs:
        pdfs = sorted(topic_dir.glob("*.pdf"), key=natural_sort_key)
        if pdfs:
            if not pdfinfo or not pdftoppm:
                raise PipelineError(
                    "PDF ingestion requires Poppler's pdfinfo and pdftoppm commands. "
                    "Set OMESH_PDFINFO and OMESH_PDFTOPPM if they are not on PATH."
                )
            for pdf_path in pdfs:
                print(f"Ingesting {topic_dir.name} / {pdf_path.name}")
                page_count, chunk_count = process_pdf(
                    pdf_path,
                    topic_dir,
                    notes_root,
                    args.db,
                    args.dpi,
                    args.language,
                    pdfinfo,
                    pdftoppm,
                    tesseract,
                )
                print(f"  Pages: {page_count}; searchable passages: {chunk_count}")
                processed_topics += 1
            continue

        page_count, chunk_count = process_images(
            topic_dir, notes_root, args.db, args.language, tesseract, heif_convert
        )
        if page_count:
            print(f"Ingested {topic_dir.name}: {page_count} scans; {chunk_count} passages")
            processed_topics += 1

    if processed_topics == 0:
        raise PipelineError("No PDF or supported image notes were found in the selected folder(s).")
    print(f"Index saved to: {args.db}")
    print("Original note files were read in place and were not changed.")


def make_fts_query(question: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for term in WORD_PATTERN.findall(question.casefold()):
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    if not terms:
        return ""
    return "content : (" + " OR ".join(f'"{term}"' for term in terms) + ")"


def search_notes(
    db_path: Path, question: str, topic: str | None, limit: int
) -> list[dict[str, object]]:
    if not db_path.is_file():
        raise PipelineError(f"Notes index not found: {db_path}. Run the ingest command first.")
    match_expression = make_fts_query(question)
    if not match_expression:
        return []

    db_uri = db_path.resolve().as_uri() + "?mode=ro"
    sql = """SELECT topic, source_ref, page_number, content, bm25(note_chunks) AS score
             FROM note_chunks WHERE note_chunks MATCH ?"""
    values: list[object] = [match_expression]
    if topic:
        sql += " AND topic = ? COLLATE NOCASE"
        values.append(topic)
    sql += " ORDER BY score LIMIT ?"
    values.append(max(limit * 4, limit))

    with sqlite3.connect(db_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, values).fetchall()

    results: list[dict[str, object]] = []
    seen_pages: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["source_ref"]), int(row["page_number"]))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        results.append(dict(row))
        if len(results) == limit:
            break
    return results


def ask_ollama(question: str, passages: list[dict[str, object]], model: str, base_url: str) -> str:
    context_parts: list[str] = []
    for index, passage in enumerate(passages, start=1):
        context_parts.append(
            f"[S{index}] Topic: {passage['topic']}\n"
            f"Source file: {passage['source_ref']}\n"
            f"Page: {passage['page_number']}\n"
            f"Passage: {passage['content']}"
        )
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Omesh, a careful study tutor. Answer using only the supplied "
                        "study-note passages. Explain clearly at A Level when the notes support "
                        "it. Cite each factual point with the passage label, such as [S1]. "
                        "If the passages do not contain enough information, say so instead of "
                        "guessing. Preserve uncertainty when OCR text is unclear. Never invent "
                        "page numbers or sources."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nStudy-note passages:\n\n"
                    + "\n\n".join(context_parts),
                },
            ],
            "stream": False,
        },
        timeout=(5, 300),
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        raise PipelineError(f"Ollama returned an error: {detail}") from error
    try:
        return str(response.json()["message"]["content"])
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError("Ollama's response did not include message.content.") from error


def answer(args: argparse.Namespace) -> None:
    passages = search_notes(args.db, args.question, args.topic, args.limit)
    if not passages:
        print("I could not find a matching passage in the indexed notes.")
        print("Try another wording or ingest the relevant topic first.")
        return

    text = ask_ollama(args.question, passages, args.model, args.ollama_url)
    valid_labels = {f"S{index}" for index in range(1, len(passages) + 1)}
    cited_labels = set(re.findall(r"\[S(\d+)\]", text))
    cited_labels = {f"S{label}" for label in cited_labels}
    unknown_labels = sorted(cited_labels - valid_labels)

    print(text.strip())
    if unknown_labels:
        print("\nWarning: the answer included a source label that was not retrieved.")
    elif not cited_labels:
        print("\nThe answer did not include inline source labels; retrieved notes are listed below.")
    print("\nRetrieved notes:")
    for index, passage in enumerate(passages, start=1):
        print(
            f"[S{index}] {passage['topic']} — {passage['source_ref']}, "
            f"page {passage['page_number']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index the user's study notes and ask a source-grounded Omesh question."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="OCR and index note pages")
    selection = ingest_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--topic", help="One exact topic-folder name")
    selection.add_argument("--all", action="store_true", help="Index every topic folder")
    ingest_parser.add_argument("--source-dir", type=Path, default=DEFAULT_NOTES_DIR)
    ingest_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ingest_parser.add_argument("--dpi", type=int, default=220)
    ingest_parser.add_argument("--language", default=DEFAULT_OCR_LANGUAGE)
    ingest_parser.set_defaults(handler=ingest)

    ask_parser = commands.add_parser("ask", help="Retrieve notes and ask the local Ollama model")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--topic", help="Restrict retrieval to one exact topic-folder name")
    ask_parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ask_parser.add_argument("--limit", type=int, default=5)
    ask_parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    ask_parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ask_parser.set_defaults(handler=answer)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "db"):
        args.db = args.db.expanduser().resolve()
    if hasattr(args, "source_dir"):
        args.source_dir = args.source_dir.expanduser()
    if hasattr(args, "limit") and args.limit < 1:
        parser.error("--limit must be at least 1")
    if hasattr(args, "dpi") and args.dpi < 100:
        parser.error("--dpi must be at least 100")
    try:
        args.handler(args)
        return 0
    except PipelineError as error:
        print(f"Omesh notes RAG: {error}", file=sys.stderr)
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Could not reach Ollama: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
