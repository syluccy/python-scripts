#!/usr/bin/env python3
"""
DOCX -> magyar fordítás OpenAI API-val.

Mit csinál?
- A .docx szöveges bekezdéseit és táblacelláit magyarra fordítja.
- A képeket, alakzatokat, oldaltöréseket, fejlécet/láblécet és a dokumentum csomagját érintetlenül hagyja.
- Megpróbálja megtartani a stílusokat: a fordított szöveget az első run helyére írja.
- Tulajdonneveket, helységneveket, utcaneveket, közterület-neveket, intézményneveket nem fordít le.

Telepítés:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install openai python-docx tqdm

Használat:
    export OPENAI_API_KEY="sk-..."
    python translate_docx_hu.py modulo2.docx modulo2_hu.docx

Opcionális:
    python translate_docx_hu.py modulo2.docx modulo2_hu.docx --model gpt-5-mini
"""

import argparse
import os
import time
from typing import Iterable, List, Tuple

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table, _Cell
from openai import OpenAI
from tqdm import tqdm


SYSTEM_INSTRUCTIONS = """
Te profi spanyol-magyar fordító vagy.
Feladat: a megadott DOCX-dokumentum szövegét magyarra fordítani.

Szigorú szabályok:
- Csak magyar fordítást adj vissza, magyarázat nélkül.
- Ne fordítsd le a tulajdonneveket.
- Ne fordítsd le a helységneveket, kerületneveket, utcaneveket, tereket, sugárutakat, közterület-neveket.
- Ne fordítsd le az intézményneveket, múzeumokat, színházakat, templomokat, éttermeket, hoteleket, műemlékeket.
- Példák, amelyeket eredetiben kell hagyni: Madrid, Puerta del Sol, Calle Alcalá, Plaza Mayor, Paseo del Prado,
  Museo del Prado, Teatro Español, Casa Lucio, Gran Vía, Aranjuez, San Lorenzo de El Escorial.
- A rövidítéseket hagyd meg: DGT, ZBE, VTC.
- A felsorolásjeleket és számozást jelentés szerint tartsd meg.
- Vizsgafelkészülési anyaghoz természetes, pontos, könnyen tanulható magyar fordítást készíts.
- Ha egy sor csak név vagy cím, hagyd eredetiben.
"""


def iter_block_items(parent):
    """Bekezdések és táblák dokumentumsorrendben."""
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.document import Document as _Document

    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        return

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def collect_paragraphs(parent) -> List[Paragraph]:
    items = []
    for block in iter_block_items(parent):
        if isinstance(block, Paragraph):
            items.append(block)
        elif isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    items.extend(collect_paragraphs(cell))
    return items


def should_translate(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if len(s) <= 1:
        return False
    # Csak szám / írásjel / nagyon rövid jelölés
    if all(ch.isdigit() or ch.isspace() or ch in ".:,;-/()€" for ch in s):
        return False
    return True


def translate_text(client: OpenAI, model: str, text: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=text,
            )
            out = response.output_text.strip()
            return out if out else text
        except Exception as e:
            if attempt == retries - 1:
                print(f"\nHIBA, eredeti szöveg marad: {text[:80]!r}\n{e}")
                return text
            time.sleep(2 ** attempt)


def set_paragraph_text_keep_first_run(paragraph: Paragraph, new_text: str) -> None:
    """
    Megőrzi a bekezdés stílusát. A run-szintű vegyes formázás részben elveszhet,
    de a képek és nem szöveges objektumok a dokumentumban megmaradnak.
    """
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(new_text)
        return

    runs[0].text = new_text
    for r in runs[1:]:
        r.text = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_docx", help="Forrás .docx")
    ap.add_argument("output_docx", help="Kimeneti .docx")
    ap.add_argument("--model", default="gpt-5-mini", help="OpenAI modell, pl. gpt-5-mini vagy gpt-5.5")
    ap.add_argument("--limit", type=int, default=0, help="Teszteléshez: max. ennyi bekezdést fordítson. 0 = mind")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Hiányzik az OPENAI_API_KEY környezeti változó.")

    client = OpenAI()
    doc = Document(args.input_docx)

    paragraphs = [p for p in collect_paragraphs(doc) if should_translate(p.text)]
    if args.limit:
        paragraphs = paragraphs[: args.limit]

    for p in tqdm(paragraphs, desc="Fordítás"):
        original = p.text
        translated = translate_text(client, args.model, original)
        set_paragraph_text_keep_first_run(p, translated)

    doc.save(args.output_docx)
    print(f"Kész: {args.output_docx}")


if __name__ == "__main__":
    main()
