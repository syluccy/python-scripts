#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SRT_BATCH_SIZE = int(os.getenv("SRT_BATCH_SIZE", "20"))
TXT_BATCH_LINES = int(os.getenv("TXT_BATCH_LINES", "40"))
MAX_RETRIES = int(os.getenv("SRT_MAX_RETRIES", "4"))
REQUEST_TIMEOUT = float(os.getenv("SRT_REQUEST_TIMEOUT", "180"))

TIME_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$"
)

SYSTEM_PROMPT = """Te professzionális magyar fordító vagy.

Feladat:
Spanyol vagy angol szöveget kell magyarra fordítanod.

KÖTELEZŐ SZABÁLYOK:
- Csak a megadott szövegsorokat fordítsd.
- Ne írj plusz kommentárt vagy magyarázatot.
- Minden blokkhoz pontosan ugyanannyi sort adj vissza, mint amennyit kaptál.
- A hangnem legyen kontextusfüggő és következetes.
- Az akronimákat általában ne fordítsd le (pl. FBI, CIA).
- Hivatalok, szakmák, beosztások fordíthatók.
- Közterületneveket ne fordítsd le.
- Dalszöveget ne fordítsd le.
- A dátumok magyar formátumban jelenjenek meg.
- A magyar szöveg legyen természetes és nyelvhelyes.
- Ha a jelentés és a sorstruktúra ütközik, tömörebben fogalmazz, de a sorok számát tartsd meg.
- Ne hagyj ki egyetlen blokkot sem.
- Ne vonj össze blokkokat.
- Ne bonts szét blokkokat.
"""

TRANSLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "lines": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["id", "lines"],
            },
        }
    },
    "required": ["items"],
}


@dataclass
class SRTBlock:
    index: str
    timecode: str
    lines: List[str]


@dataclass
class TextChunk:
    index: int
    lines: List[str]


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_preserve_bom(path: Path) -> Tuple[str, bool]:
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    return text, has_bom


def write_text_preserve_bom(path: Path, text: str, add_bom: bool) -> None:
    data = text.encode("utf-8")
    if add_bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)


def parse_srt(text: str) -> List[SRTBlock]:
    text = normalize_newlines(text).strip("\n")
    if not text:
        return []

    raw_blocks = re.split(r"\n{2,}", text)
    blocks: List[SRTBlock] = []

    for n, raw_block in enumerate(raw_blocks, start=1):
        lines = raw_block.split("\n")
        if len(lines) < 3:
            raise ValueError(f"Érvénytelen SRT blokk #{n}: túl rövid.")

        index = lines[0].strip()
        timecode = lines[1].strip()
        text_lines = lines[2:]

        if not index.isdigit():
            raise ValueError(f"Érvénytelen sorszám a(z) {n}. blokkban: {index!r}")

        if not TIME_RE.match(timecode):
            raise ValueError(f"Érvénytelen időzítés a(z) {n}. blokkban: {timecode!r}")

        if not text_lines:
            raise ValueError(f"Nincs szövegsor a(z) {n}. blokkban.")

        blocks.append(SRTBlock(index=index, timecode=timecode, lines=text_lines))

    return blocks


def build_srt(blocks: List[SRTBlock]) -> str:
    out: List[str] = []
    for block in blocks:
        out.append(block.index)
        out.append(block.timecode)
        out.extend(block.lines)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def parse_txt_to_chunks(text: str, chunk_size: int) -> Tuple[List[TextChunk], bool]:
    text = normalize_newlines(text)
    had_trailing_newline = text.endswith("\n")
    lines = text.split("\n")

    # Ha a file newline-ra végződött, split ad egy üres utolsó sort.
    # Ezt megtartjuk, hogy visszaépítéskor se vesszen el.
    chunks: List[TextChunk] = []
    idx = 1
    for i in range(0, len(lines), chunk_size):
        chunks.append(TextChunk(index=idx, lines=lines[i:i + chunk_size]))
        idx += 1

    return chunks, had_trailing_newline


def build_txt_from_chunks(chunks: List[TextChunk], had_trailing_newline: bool) -> str:
    lines: List[str] = []
    for chunk in chunks:
        lines.extend(chunk.lines)

    text = "\n".join(lines)
    if had_trailing_newline and not text.endswith("\n"):
        text += "\n"
    return text


def build_output_name(input_path: Path) -> Path:
    name = input_path.name

    if name.endswith(".hu.srt") or name.endswith(".hu.txt"):
        return input_path

    if name.endswith(".es.srt"):
        output_name = name[:-7] + ".hu.srt"
    elif name.endswith(".en.srt"):
        output_name = name[:-7] + ".hu.srt"
    elif name.endswith(".srt"):
        output_name = input_path.stem + ".hu.srt"
    elif name.endswith(".txt"):
        output_name = input_path.stem + ".hu.txt"
    else:
        output_name = input_path.stem + ".hu" + input_path.suffix

    return input_path.with_name(output_name)


def batched(items: List, size: int):
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def validate_batch_lines(original_batch: List[List[str]], translated_items: List[dict], expected_ids: List[int]) -> List[str]:
    errors: List[str] = []

    if len(original_batch) != len(translated_items):
        errors.append(
            f"Eltérő elemszám: eredeti={len(original_batch)}, fordítás={len(translated_items)}"
        )
        return errors

    for pos, (orig_lines, tr, expected_id) in enumerate(zip(original_batch, translated_items, expected_ids), start=1):
        if not isinstance(tr, dict):
            errors.append(f"{pos}. elem nem objektum.")
            continue

        got_id = tr.get("id")
        if got_id != expected_id:
            errors.append(f"{pos}. elem hibás id: várt={expected_id}, kapott={got_id}")
            continue

        lines = tr.get("lines")
        if not isinstance(lines, list):
            errors.append(f"{expected_id}. blokk: a 'lines' nem lista.")
            continue

        if len(lines) != len(orig_lines):
            errors.append(
                f"{expected_id}. blokk: eltérő sorszám a blokkban: "
                f"eredeti={len(orig_lines)}, fordítás={len(lines)}"
            )
            continue

        for i, line in enumerate(lines, start=1):
            if not isinstance(line, str):
                errors.append(f"{expected_id}. blokk {i}. sora nem string.")

    return errors


def save_debug_files(batch_label: str, request_payload: dict, response_text: str) -> None:
    debug_dir = Path("debug_batches")
    debug_dir.mkdir(parents=True, exist_ok=True)

    safe_label = batch_label.replace("/", "_")
    req_path = debug_dir / f"batch_{safe_label}.request.json"
    resp_path = debug_dir / f"batch_{safe_label}.response.json"

    req_path.write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resp_path.write_text(response_text, encoding="utf-8")


def save_partial_output_srt(
    output_path: Path,
    translated_blocks: List[SRTBlock],
    has_bom: bool,
) -> None:
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.srt")
    partial_content = build_srt(translated_blocks)
    write_text_preserve_bom(partial_path, partial_content, has_bom)


def save_partial_output_txt(
    output_path: Path,
    translated_chunks: List[TextChunk],
    has_bom: bool,
    had_trailing_newline: bool,
) -> None:
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.txt")
    partial_content = build_txt_from_chunks(translated_chunks, had_trailing_newline)
    write_text_preserve_bom(partial_path, partial_content, has_bom)


def translate_batch_once(
    client: OpenAI,
    batch_label: str,
    ids: List[int],
    lines_list: List[List[str]],
) -> List[List[str]]:
    payload = {
        "items": [
            {
                "id": item_id,
                "lines": lines,
            }
            for item_id, lines in zip(ids, lines_list)
        ]
    }

    user_prompt = (
        "Fordítsd le magyarra a következő szövegblokkok szövegsorait.\n"
        "Csak JSON-t adj vissza a megadott schema szerint.\n"
        "Minden id maradjon ugyanaz.\n"
        "Minden blokkban ugyanannyi line legyen, mint az inputban.\n"
        "Ne hagyj ki egyetlen elemet sem.\n\n"
        f"INPUT JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "translation_batch",
                        "schema": TRANSLATION_SCHEMA,
                        "strict": True,
                    }
                },
                timeout=REQUEST_TIMEOUT,
            )

            raw_text = response.output_text.strip()
            save_debug_files(batch_label, payload, raw_text)

            data = json.loads(raw_text)
            items = data["items"]

            errors = validate_batch_lines(lines_list, items, ids)
            if errors:
                last_error = " | ".join(errors)
                print(
                    f"  Batch {batch_label}, próbálkozás {attempt}/{MAX_RETRIES}: "
                    f"szerkezeti hiba: {last_error}"
                )
                time.sleep(min(8, 1.5 * attempt))
                continue

            return [item["lines"] for item in items]

        except Exception as e:
            last_error = str(e)
            print(
                f"  Batch {batch_label}, próbálkozás {attempt}/{MAX_RETRIES}: "
                f"API/JSON hiba: {last_error}"
            )
            time.sleep(min(8, 1.5 * attempt))

    raise RuntimeError(f"Batch {batch_label} sikertelen: {last_error}")


def translate_batch_resilient(
    client: OpenAI,
    batch_label: str,
    ids: List[int],
    lines_list: List[List[str]],
) -> List[List[str]]:
    try:
        return translate_batch_once(client, batch_label, ids, lines_list)
    except Exception as e:
        if len(ids) == 1:
            raise RuntimeError(
                f"Batch {batch_label} egyetlen blokkra bontva is sikertelen: {e}"
            )

        mid = len(ids) // 2
        left_ids = ids[:mid]
        right_ids = ids[mid:]
        left_lines = lines_list[:mid]
        right_lines = lines_list[mid:]

        print(
            f"  Figyelem: batch {batch_label} hibás volt ({len(ids)} blokk). "
            f"Újrapróbálom kisebb részekben: {len(left_ids)} + {len(right_ids)}"
        )

        left_result = translate_batch_resilient(client, f"{batch_label}.1", left_ids, left_lines)
        right_result = translate_batch_resilient(client, f"{batch_label}.2", right_ids, right_lines)

        return left_result + right_result


def process_srt_file(client: OpenAI, input_path: Path) -> bool:
    output_path = build_output_name(input_path)

    print(f"\n=== SRT feldolgozás: {input_path} ===")
    original_text, has_bom = read_text_preserve_bom(input_path)

    try:
        blocks = parse_srt(original_text)
    except Exception as e:
        print(f"Hiba az SRT beolvasásakor: {e}")
        return False

    if not blocks:
        print("Üres SRT fájl, kihagyva.")
        return False

    translated_blocks: List[SRTBlock] = []
    total_batches = (len(blocks) + SRT_BATCH_SIZE - 1) // SRT_BATCH_SIZE

    try:
        for batch_no, (_, batch) in enumerate(batched(blocks, SRT_BATCH_SIZE), start=1):
            print(f"Fordítás: batch {batch_no}/{total_batches} ({len(batch)} blokk)")

            ids = [int(b.index) for b in batch]
            lines_list = [b.lines for b in batch]

            translated_lines_list = translate_batch_resilient(
                client, f"{input_path.name}-srt-{batch_no}", ids, lines_list
            )

            for original_block, translated_lines in zip(batch, translated_lines_list):
                translated_blocks.append(
                    SRTBlock(
                        index=original_block.index,
                        timecode=original_block.timecode,
                        lines=translated_lines,
                    )
                )

            save_partial_output_srt(output_path, translated_blocks, has_bom)

    except Exception as e:
        print(f"Hiba fordítás közben: {e}")
        if translated_blocks:
            save_partial_output_srt(output_path, translated_blocks, has_bom)
            print(
                f"Részleges kimenet elmentve ide: "
                f"{output_path.with_suffix(output_path.suffix + '.partial.srt')}"
            )
        return False

    result = build_srt(translated_blocks)
    write_text_preserve_bom(output_path, result, has_bom)

    partial_path = output_path.with_suffix(output_path.suffix + ".partial.srt")
    if partial_path.exists():
        partial_path.unlink()

    print(f"Kész: {output_path}")
    return True


def process_txt_file(client: OpenAI, input_path: Path) -> bool:
    output_path = build_output_name(input_path)

    print(f"\n=== TXT feldolgozás: {input_path} ===")
    original_text, has_bom = read_text_preserve_bom(input_path)

    chunks, had_trailing_newline = parse_txt_to_chunks(original_text, TXT_BATCH_LINES)

    if not chunks:
        print("Üres TXT fájl, kihagyva.")
        return False

    translated_chunks: List[TextChunk] = []
    total_batches = len(chunks)

    try:
        for batch_no, chunk in enumerate(chunks, start=1):
            print(f"Fordítás: chunk {batch_no}/{total_batches} ({len(chunk.lines)} sor)")

            ids = [chunk.index]
            lines_list = [chunk.lines]

            translated_lines_list = translate_batch_resilient(
                client, f"{input_path.name}-txt-{batch_no}", ids, lines_list
            )

            translated_chunks.append(
                TextChunk(index=chunk.index, lines=translated_lines_list[0])
            )

            save_partial_output_txt(output_path, translated_chunks, has_bom, had_trailing_newline)

    except Exception as e:
        print(f"Hiba fordítás közben: {e}")
        if translated_chunks:
            save_partial_output_txt(output_path, translated_chunks, has_bom, had_trailing_newline)
            print(
                f"Részleges kimenet elmentve ide: "
                f"{output_path.with_suffix(output_path.suffix + '.partial.txt')}"
            )
        return False

    result = build_txt_from_chunks(translated_chunks, had_trailing_newline)
    write_text_preserve_bom(output_path, result, has_bom)

    partial_path = output_path.with_suffix(output_path.suffix + ".partial.txt")
    if partial_path.exists():
        partial_path.unlink()

    print(f"Kész: {output_path}")
    return True


def should_skip_file(path: Path) -> bool:
    lower = path.name.lower()
    if lower.endswith(".hu.srt") or lower.endswith(".hu.txt"):
        return True
    return False


def collect_files(folder: Path) -> List[Path]:
    files = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".srt", ".txt"):
            continue
        if should_skip_file(path):
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    if len(sys.argv) != 2:
        print("Használat:")
        print('  python3 translate.py "/utvonal/a/mappahoz"')
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Hiba: a mappa nem található vagy nem mappa: {input_folder}")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Hiba: nincs beállítva az OPENAI_API_KEY környezeti változó.")
        sys.exit(1)

    files = collect_files(input_folder)
    if not files:
        print("Nem találtam fordítandó .srt vagy .txt fájlokat.")
        sys.exit(0)

    print(f"Modell: {MODEL}")
    print(f"SRT batch méret: {SRT_BATCH_SIZE}")
    print(f"TXT chunk sorok: {TXT_BATCH_LINES}")
    print(f"Max retry: {MAX_RETRIES}")
    print(f"Talált fájlok: {len(files)}")

    client = OpenAI(api_key=api_key)

    ok_count = 0
    fail_count = 0

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".srt":
                ok = process_srt_file(client, file_path)
            else:
                ok = process_txt_file(client, file_path)

            if ok:
                ok_count += 1
            else:
                fail_count += 1

        except KeyboardInterrupt:
            print("\nMegszakítva.")
            sys.exit(130)
        except Exception as e:
            print(f"\nVáratlan hiba ennél a fájlnál: {file_path}")
            print(f"Hiba: {e}")
            fail_count += 1

    print("\n=== Összegzés ===")
    print(f"Sikeres: {ok_count}")
    print(f"Hibás:   {fail_count}")


if __name__ == "__main__":
    main()
