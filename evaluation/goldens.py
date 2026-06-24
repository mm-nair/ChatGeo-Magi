import os
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is missing in .env!"

client = OpenAI(api_key=OPENAI_API_KEY)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except:
            pass
    return "\n".join(text)


def load_text_file(path: Path) -> str:
    return path.read_text(errors="ignore")


def load_corpus() -> str:
    folder = Path("data")
    assert folder.exists(), "data/ folder does not exist!"

    corpus_parts = []

    for file in folder.iterdir():
        if file.suffix.lower() in [".pdf"]:
            print(f"[PDF] Loading {file.name}")
            corpus_parts.append(load_pdf(file))
        elif file.suffix.lower() in [".txt", ".md"]:
            print(f"[TXT] Loading {file.name}")
            corpus_parts.append(load_text_file(file))
        else:
            print(f"[SKIP] Unknown format: {file.name}")

    corpus = "\n\n".join(corpus_parts)
    print(f"Loaded {len(corpus)} total characters from data/")
    return corpus

GENERATION_PROMPT = """
You are generating GOLDEN evaluation questions for a Retrieval-Augmented Generation (RAG) system.

Your task:
- Read the provided document corpus.
- Produce exactly **25 Q&A pairs**.
- Each question MUST be answerable solely from the corpus.
- Questions must test whether the retrieval system can recall *concepts, definitions, procedures, reasoning, and major facts*.
- DO NOT ask hyper-specific trivia such as “what number is on page 12” or “what is the 3rd bullet point.”
- DO ask questions that require retrieving meaningful content (procedures, explanations, scientific values, relationships, definitions, summary facts, etc.)
- Answers must be correct, grounded, and self-contained.
- Format your output as a JSON list of objects with "question" and "answer".

Example format:
[
  {"question": "...", "answer": "..."},
  ...
]

Return ONLY valid JSON. No commentary.
"""

def generate_goldens(context: str):
    messages = [
        {"role": "system", "content": GENERATION_PROMPT},
        {"role": "user", "content": context[:50000]}
    ]

    print("Calling OpenAI model...")

    response = client.responses.create(
        model="gpt-4.1",
        input=messages,
        temperature=0.2,
        max_output_tokens=4096,
    )

    output = response.output_text
    return json.loads(output)

def save_to_csv(items, path="goldens.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        for qa in items:
            writer.writerow([qa["question"], qa["answer"]])
    print(f"Saved {len(items)} rows to {path}")


if __name__ == "__main__":
    context = load_corpus()
    goldens = generate_goldens(context)
    save_to_csv(goldens)