#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
BATCH_SIZE = int(os.getenv("SRT_BATCH_SIZE", "20"))
MAX_RETRIES = int(os.getenv("SRT_MAX_RETRIES", "4"))
REQUEST_TIMEOUT = float(os.getenv("SRT_REQUEST_TIMEOUT", "180"))

TIME_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$"
)

SYSTEM_PROMPT = """Te professzionális magyar feliratfordító vagy.

Feladat:
Spanyol vagy angol feliratszöveget kell magyarra fordítanod.

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


def build_output_name(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".es.srt"):
        output_name = name[:-7] + ".hu.srt"
    elif name.endswith(".en.srt"):
        output_name = name[:-7] + ".hu.srt"
    else:
        output_name = input_path.stem + ".hu.srt"
    return input_path.with_name(output_name)


def batched(items: List[SRTBlock], size: int):
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def validate_batch(original_batch: List[SRTBlock], translated_items: List[dict]) -> List[str]:
    errors: List[str] = []

    if len(original_batch) != len(translated_items):
        errors.append(
            f"Eltérő elemszám: eredeti={len(original_batch)}, fordítás={len(translated_items)}"
        )
        return errors

    for pos, (orig, tr) in enumerate(zip(original_batch, translated_items), start=1):
        expected_id = int(orig.index)

        if not isinstance(tr, dict):
            errors.append(f"{pos}. elem nem objektum.")
            continue

        got_id = tr.get("id")
        if got_id != expected_id:
            errors.append(
                f"{pos}. elem hibás id: várt={expected_id}, kapott={got_id}"
            )
            continue

        lines = tr.get("lines")
        if not isinstance(lines, list):
            errors.append(f"{expected_id}. blokk: a 'lines' nem lista.")
            continue

        if len(lines) != len(orig.lines):
            errors.append(
                f"{expected_id}. blokk: eltérő sorszám a blokkban: "
                f"eredeti={len(orig.lines)}, fordítás={len(lines)}"
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


def save_partial_output(
    output_path: Path,
    translated_blocks: List[SRTBlock],
    has_bom: bool,
) -> None:
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.srt")
    partial_content = build_srt(translated_blocks)
    write_text_preserve_bom(partial_path, partial_content, has_bom)


def translate_batch_once(
    client: OpenAI,
    batch_label: str,
    batch: List[SRTBlock],
) -> List[List[str]]:
    payload = {
        "items": [
            {
                "id": int(block.index),
                "lines": block.lines,
            }
            for block in batch
        ]
    }

    user_prompt = (
        "Fordítsd le magyarra a következő feliratblokkok szövegsorait.\n"
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
                        "name": "srt_translation_batch",
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

            errors = validate_batch(batch, items)
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
    batch: List[SRTBlock],
) -> List[List[str]]:
    try:
        return translate_batch_once(client, batch_label, batch)
    except Exception as e:
        if len(batch) == 1:
            raise RuntimeError(
                f"Batch {batch_label} egyetlen blokkra bontva is sikertelen: {e}"
            )

        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]

        print(
            f"  Figyelem: batch {batch_label} hibás volt ({len(batch)} blokk). "
            f"Újrapróbálom kisebb részekben: {len(left)} + {len(right)}"
        )

        left_result = translate_batch_resilient(client, f"{batch_label}.1", left)
        right_result = translate_batch_resilient(client, f"{batch_label}.2", right)

        return left_result + right_result


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("Használat:")
        print('  python3 translate.py "input.es.srt"')
        print('  python3 translate.py "input.es.srt" "output.hu.srt"')
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Hiba: a fájl nem található: {input_path}")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else build_output_name(input_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Hiba: nincs beállítva az OPENAI_API_KEY környezeti változó.")
        sys.exit(1)

    original_text, has_bom = read_text_preserve_bom(input_path)

    try:
        blocks = parse_srt(original_text)
    except Exception as e:
        print(f"Hiba az SRT beolvasásakor: {e}")
        sys.exit(1)

    if not blocks:
        print("Hiba: üres SRT fájl.")
        sys.exit(1)

    print(f"Beolvasva: {len(blocks)} blokk")
    print(f"Modell: {MODEL}")
    print(f"Batch méret: {BATCH_SIZE}")
    print(f"Max retry: {MAX_RETRIES}")

    client = OpenAI(api_key=api_key)

    translated_blocks: List[SRTBlock] = []
    total_batches = (len(blocks) + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        for batch_no, (_, batch) in enumerate(batched(blocks, BATCH_SIZE), start=1):
            print(f"Fordítás: batch {batch_no}/{total_batches} ({len(batch)} blokk)")
            translated_lines_list = translate_batch_resilient(client, str(batch_no), batch)

            for original_block, translated_lines in zip(batch, translated_lines_list):
                translated_blocks.append(
                    SRTBlock(
                        index=original_block.index,
                        timecode=original_block.timecode,
                        lines=translated_lines,
                    )
                )

            save_partial_output(output_path, translated_blocks, has_bom)

    except Exception as e:
        print(f"Hiba fordítás közben: {e}")
        if translated_blocks:
            save_partial_output(output_path, translated_blocks, has_bom)
            print(
                f"Részleges kimenet elmentve ide: "
                f"{output_path.with_suffix(output_path.suffix + '.partial.srt')}"
            )
        sys.exit(2)

    result = build_srt(translated_blocks)
    write_text_preserve_bom(output_path, result, has_bom)

    partial_path = output_path.with_suffix(output_path.suffix + ".partial.srt")
    if partial_path.exists():
        partial_path.unlink()

    print(f"Kész: {output_path}")


if __name__ == "__main__":
    main()
