#!/usr/bin/env python3
"""
Block-shift Hungarian subtitles to an English reference SRT.

The script uses the OpenAI API only to find semantic anchor points between the
Hungarian subtitle and the English reference. It then estimates piecewise
constant time shifts and applies them to the original Hungarian cues.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # Keep --analyze-only usable even without optional dotenv support.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclasses.dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclasses.dataclass
class Anchor:
    hu_index: int
    hu_start_ms: int
    hu_end_ms: int
    en_start_index: int
    en_end_index: int
    en_start_ms: int
    en_end_ms: int
    shift_ms: int
    confidence: float
    reason: str


@dataclasses.dataclass
class ShiftBlock:
    block_id: int
    start_hu_ms: int
    end_hu_ms: int
    shift_ms: int
    anchor_count: int
    confidence_median: float


def parse_time(value: str) -> int:
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def format_time(ms: int) -> str:
    ms = max(0, int(round(ms)))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def read_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []

    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.split("\n")
        if len(lines) < 2:
            continue
        time_line_index = 1 if lines[0].strip().isdigit() else 0
        match = TIME_RE.search(lines[time_line_index])
        if not match:
            continue
        index_text = lines[0].strip() if time_line_index == 1 else str(len(cues) + 1)
        try:
            index = int(index_text)
        except ValueError:
            index = len(cues) + 1
        text = "\n".join(lines[time_line_index + 1 :]).strip()
        cues.append(
            Cue(
                index=index,
                start_ms=parse_time(match.group("start")),
                end_ms=parse_time(match.group("end")),
                text=text,
            )
        )
    return cues


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u200b", "")
    return text


def normalize_for_matching(text: str) -> str:
    text = strip_tags(text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = text.replace("♪", " ")
    text = re.sub(r"^\s*[A-Z][A-Za-z .'-]{1,30}:\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_cue(text: str) -> bool:
    clean = strip_tags(text).strip()
    if not clean:
        return True
    if "♪" in clean:
        return True
    no_brackets = re.sub(r"\[[^\]]+\]", "", clean).strip()
    if not no_brackets:
        return True
    if re.fullmatch(r"[-–—\s.?!,;:()\[\]]+", no_brackets):
        return True
    return False


def dialogue_cues(cues: Iterable[Cue]) -> list[Cue]:
    result = []
    for cue in cues:
        text = normalize_for_matching(cue.text)
        if text and not is_noise_cue(cue.text):
            result.append(dataclasses.replace(cue, text=text))
    return result


def describe_stats(label: str, cues: list[Cue]) -> str:
    durations = [cue.end_ms - cue.start_ms for cue in cues]
    gaps = [b.start_ms - a.end_ms for a, b in zip(cues, cues[1:])]
    noise_count = sum(1 for cue in cues if is_noise_cue(cue.text))
    if not cues:
        return f"{label}: no cues"
    return (
        f"{label}: {len(cues)} cues, first {format_time(cues[0].start_ms)}, "
        f"last end {format_time(cues[-1].end_ms)}, noise/SFX-like {noise_count}, "
        f"median duration {format_time(int(statistics.median(durations)))}, "
        f"large gaps >25s {sum(1 for gap in gaps if gap > 25_000)}"
    )


def selected_hu_anchor_cues(
    hu_dialogue: list[Cue], anchor_step: int, large_gap_ms: int
) -> list[Cue]:
    selected: dict[int, Cue] = {}
    for i, cue in enumerate(hu_dialogue):
        if i == 0 or i == len(hu_dialogue) - 1 or i % anchor_step == 0:
            selected[cue.index] = cue

    previous: Cue | None = None
    for cue in hu_dialogue:
        if previous and cue.start_ms - previous.end_ms >= large_gap_ms:
            selected[previous.index] = previous
            selected[cue.index] = cue
        previous = cue

    return sorted(selected.values(), key=lambda cue: cue.start_ms)


def compact_cue(cue: Cue, prefix: str) -> str:
    text = normalize_for_matching(cue.text)
    text = text.replace("\n", " ")
    if len(text) > 220:
        text = text[:217] + "..."
    return f"{prefix}{cue.index} [{format_time(cue.start_ms)}] {text}"


def batched(items: list[Cue], batch_size: int) -> Iterable[list[Cue]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_prompt(
    hu_batch: list[Cue], en_candidates: list[Cue], min_confidence: float
) -> tuple[str, str]:
    system = (
        "You align Hungarian subtitle cues to an English reference subtitle. "
        "A Hungarian cue can translate one English cue or a consecutive range of "
        "English cues. Match by meaning, not by timestamps alone. Keep matches "
        "monotonic in story order. If the Hungarian cue is a title card, a very "
        "generic line, or has no reliable English equivalent, return nulls. "
        "Return only valid JSON."
    )
    user = {
        "task": (
            "For each Hungarian cue, choose the best matching consecutive English "
            "cue range from the candidates. Use confidence 0..1. Only use a match "
            f"when confidence is at least about {min_confidence}; otherwise use null."
        ),
        "output_schema": {
            "matches": [
                {
                    "hu_id": "integer Hungarian cue id",
                    "en_start_id": "integer English cue id or null",
                    "en_end_id": "integer English cue id or null",
                    "confidence": "number from 0 to 1",
                    "reason": "short reason in English or Hungarian",
                }
            ]
        },
        "hungarian_cues": [compact_cue(cue, "HU") for cue in hu_batch],
        "english_candidates": [compact_cue(cue, "EN") for cue in en_candidates],
    }
    return system, json.dumps(user, ensure_ascii=False, indent=2)


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start_positions = [pos for pos in [text.find("{"), text.find("[")] if pos >= 0]
    if not start_positions:
        raise ValueError(f"No JSON found in model response: {text[:300]}")
    start = min(start_positions)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError(f"No complete JSON found in model response: {text[:300]}")
    return json.loads(text[start : end + 1])


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    try:
        return response.choices[0].message.content
    except Exception as exc:  # pragma: no cover - defensive SDK compatibility path.
        raise RuntimeError(f"Could not read model response text: {response!r}") from exc


def call_openai_json(model: str, system: str, user: str, retries: int) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `pip install -r requirements.txt`."
        ) from exc

    client = OpenAI()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if hasattr(client, "responses"):
                try:
                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0,
                    )
                except TypeError:
                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            return extract_json(response_text(response))
        except Exception as exc:  # Network/API/JSON issues are retried together.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI call failed after {retries + 1} attempts: {last_error}")


def load_anchor_cache(path: Path | None) -> dict[int, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in data.items()}


def save_anchor_cache(path: Path | None, cache: dict[int, dict[str, Any]]) -> None:
    if not path:
        return
    serializable = {str(key): value for key, value in sorted(cache.items())}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def find_anchors(
    en_dialogue: list[Cue],
    hu_anchor_cues: list[Cue],
    model: str,
    batch_size: int,
    search_window_ms: int,
    min_confidence: float,
    retries: int,
    cache_path: Path | None,
) -> list[Anchor]:
    en_by_index = {cue.index: cue for cue in en_dialogue}
    hu_by_index = {cue.index: cue for cue in hu_anchor_cues}
    cache = load_anchor_cache(cache_path)
    anchors: list[Anchor] = []

    for batch_number, hu_batch in enumerate(batched(hu_anchor_cues, batch_size), start=1):
        uncached = [cue for cue in hu_batch if cue.index not in cache]
        if uncached:
            min_t = min(cue.start_ms for cue in uncached) - search_window_ms
            max_t = max(cue.end_ms for cue in uncached) + search_window_ms
            candidates = [
                cue for cue in en_dialogue if cue.end_ms >= min_t and cue.start_ms <= max_t
            ]
            if not candidates:
                continue

            print(
                f"API batch {batch_number}: {len(uncached)} HU anchors, "
                f"{len(candidates)} EN candidates",
                file=sys.stderr,
            )
            system, user = build_prompt(uncached, candidates, min_confidence)
            data = call_openai_json(model=model, system=system, user=user, retries=retries)
            matches = data.get("matches", data if isinstance(data, list) else [])
            for item in matches:
                try:
                    hu_id = int(item["hu_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                cache[hu_id] = item
            save_anchor_cache(cache_path, cache)

        for cue in hu_batch:
            item = cache.get(cue.index)
            if not item:
                continue
            try:
                hu_id = int(item["hu_id"])
                en_start_id = item.get("en_start_id")
                en_end_id = item.get("en_end_id")
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                continue
            if confidence < min_confidence or en_start_id is None or en_end_id is None:
                continue
            try:
                en_start_id = int(en_start_id)
                en_end_id = int(en_end_id)
            except (TypeError, ValueError):
                continue
            if hu_id not in hu_by_index or en_start_id not in en_by_index:
                continue
            if en_end_id not in en_by_index:
                en_end_id = en_start_id
            if en_end_id < en_start_id:
                en_start_id, en_end_id = en_end_id, en_start_id

            hu = hu_by_index[hu_id]
            en_start = en_by_index[en_start_id]
            en_end = en_by_index[en_end_id]
            anchors.append(
                Anchor(
                    hu_index=hu.index,
                    hu_start_ms=hu.start_ms,
                    hu_end_ms=hu.end_ms,
                    en_start_index=en_start.index,
                    en_end_index=en_end.index,
                    en_start_ms=en_start.start_ms,
                    en_end_ms=en_end.end_ms,
                    shift_ms=en_start.start_ms - hu.start_ms,
                    confidence=confidence,
                    reason=str(item.get("reason") or ""),
                )
            )

    anchors = sorted(anchors, key=lambda anchor: (anchor.hu_start_ms, anchor.en_start_ms))
    monotonic: list[Anchor] = []
    last_en = -math.inf
    for anchor in anchors:
        if anchor.en_start_ms >= last_en:
            monotonic.append(anchor)
            last_en = anchor.en_start_ms
    return monotonic


def median_ms(values: Iterable[int]) -> int:
    return int(round(statistics.median(list(values))))


def build_shift_blocks(
    anchors: list[Anchor], all_hu: list[Cue], break_threshold_ms: int
) -> list[ShiftBlock]:
    if not anchors:
        raise SystemExit("No reliable anchors found. Try lowering --min-confidence or raising --search-window-sec.")

    groups: list[list[Anchor]] = []
    current: list[Anchor] = []
    for anchor in anchors:
        if not current:
            current.append(anchor)
            continue
        current_median = median_ms(a.shift_ms for a in current)
        if abs(anchor.shift_ms - current_median) > break_threshold_ms:
            groups.append(current)
            current = [anchor]
        else:
            current.append(anchor)
    if current:
        groups.append(current)

    boundaries: list[tuple[int, int]] = []
    episode_start = all_hu[0].start_ms if all_hu else 0
    episode_end = all_hu[-1].end_ms if all_hu else 0
    for i, group in enumerate(groups):
        if i == 0:
            start = episode_start
        else:
            start = (groups[i - 1][-1].hu_start_ms + group[0].hu_start_ms) // 2
        if i == len(groups) - 1:
            end = episode_end + 1
        else:
            end = (group[-1].hu_start_ms + groups[i + 1][0].hu_start_ms) // 2
        boundaries.append((start, end))

    blocks: list[ShiftBlock] = []
    for block_id, (group, (start, end)) in enumerate(zip(groups, boundaries), start=1):
        blocks.append(
            ShiftBlock(
                block_id=block_id,
                start_hu_ms=start,
                end_hu_ms=end,
                shift_ms=median_ms(a.shift_ms for a in group),
                anchor_count=len(group),
                confidence_median=float(statistics.median(a.confidence for a in group)),
            )
        )
    return blocks


def shift_for_cue(cue: Cue, blocks: list[ShiftBlock]) -> tuple[int, int]:
    midpoint = (cue.start_ms + cue.end_ms) // 2
    for block in blocks:
        if block.start_hu_ms <= midpoint < block.end_hu_ms:
            return block.shift_ms, block.block_id
    nearest = min(blocks, key=lambda block: abs(midpoint - (block.start_hu_ms + block.end_hu_ms) // 2))
    return nearest.shift_ms, nearest.block_id


def apply_blocks(
    hu_cues: list[Cue], blocks: list[ShiftBlock], drop_noise: bool, min_gap_ms: int
) -> list[tuple[Cue, Cue, int]]:
    shifted: list[tuple[Cue, Cue, int]] = []
    for cue in hu_cues:
        if drop_noise and is_noise_cue(cue.text):
            continue
        shift_ms, block_id = shift_for_cue(cue, blocks)
        new_cue = dataclasses.replace(
            cue,
            start_ms=max(0, cue.start_ms + shift_ms),
            end_ms=max(0, cue.end_ms + shift_ms),
        )
        shifted.append((cue, new_cue, block_id))

    for i in range(1, len(shifted)):
        previous_original, previous, previous_block = shifted[i - 1]
        original, current, block_id = shifted[i]
        if current.start_ms < previous.end_ms + min_gap_ms:
            max_previous_end = current.start_ms - min_gap_ms
            if max_previous_end - previous.start_ms >= 500:
                previous.end_ms = max_previous_end
            else:
                delta = previous.end_ms + min_gap_ms - current.start_ms
                current.start_ms += delta
                current.end_ms += delta
        shifted[i - 1] = (previous_original, previous, previous_block)
        shifted[i] = (original, current, block_id)
    return shifted


def write_srt(path: Path, shifted: list[tuple[Cue, Cue, int]]) -> None:
    blocks = []
    for new_index, (_, cue, _) in enumerate(shifted, start=1):
        blocks.append(
            f"{new_index}\n"
            f"{format_time(cue.start_ms)} --> {format_time(cue.end_ms)}\n"
            f"{cue.text.strip()}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def write_report(
    path: Path,
    anchors: list[Anchor],
    blocks: list[ShiftBlock],
    shifted: list[tuple[Cue, Cue, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "field_1", "field_2", "field_3", "field_4", "field_5", "field_6"])
        writer.writerow([])
        writer.writerow(["blocks", "block_id", "hu_start", "hu_end", "shift_sec", "anchors", "median_confidence"])
        for block in blocks:
            writer.writerow(
                [
                    "block",
                    block.block_id,
                    format_time(block.start_hu_ms),
                    format_time(block.end_hu_ms),
                    round(block.shift_ms / 1000, 3),
                    block.anchor_count,
                    round(block.confidence_median, 3),
                ]
            )
        writer.writerow([])
        writer.writerow(["anchors", "hu_id", "hu_time", "en_range", "shift_sec", "confidence", "reason"])
        for anchor in anchors:
            writer.writerow(
                [
                    "anchor",
                    anchor.hu_index,
                    format_time(anchor.hu_start_ms),
                    f"{anchor.en_start_index}-{anchor.en_end_index} @ {format_time(anchor.en_start_ms)}",
                    round(anchor.shift_ms / 1000, 3),
                    round(anchor.confidence, 3),
                    anchor.reason,
                ]
            )
        writer.writerow([])
        writer.writerow(["cues", "old_id", "old_time", "new_time", "block_id", "shift_sec", "text"])
        for old, new, block_id in shifted:
            writer.writerow(
                [
                    "cue",
                    old.index,
                    f"{format_time(old.start_ms)} --> {format_time(old.end_ms)}",
                    f"{format_time(new.start_ms)} --> {format_time(new.end_ms)}",
                    block_id,
                    round((new.start_ms - old.start_ms) / 1000, 3),
                    normalize_for_matching(old.text)[:180],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retiming Hungarian SRT by block-shifting it to an English reference SRT."
    )
    parser.add_argument("--english", required=True, type=Path, help="English reference .srt")
    parser.add_argument("--hungarian", required=True, type=Path, help="Hungarian .srt to retime")
    parser.add_argument("--output", required=True, type=Path, help="Output retimed Hungarian .srt")
    parser.add_argument("--report", type=Path, default=Path("alignment_report.csv"))
    parser.add_argument("--anchor-cache", type=Path, default=Path("anchor_cache.json"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--anchor-step", type=int, default=3, help="Use every Nth Hungarian dialogue cue as an API anchor candidate")
    parser.add_argument("--search-window-sec", type=float, default=900.0)
    parser.add_argument("--break-threshold-sec", type=float, default=2.5)
    parser.add_argument("--large-gap-sec", type=float, default=25.0)
    parser.add_argument("--min-confidence", type=float, default=0.62)
    parser.add_argument("--min-gap-ms", type=int, default=40)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--keep-noise", action="store_true", help="Keep music/SFX-like cues from the Hungarian source")
    parser.add_argument("--analyze-only", action="store_true", help="Print SRT stats without calling the API")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    en_cues = read_srt(args.english)
    hu_cues = read_srt(args.hungarian)

    print(describe_stats("English", en_cues), file=sys.stderr)
    print(describe_stats("Hungarian", hu_cues), file=sys.stderr)

    en_dialogue = dialogue_cues(en_cues)
    hu_dialogue = dialogue_cues(hu_cues)
    hu_anchor_cues = selected_hu_anchor_cues(
        hu_dialogue=hu_dialogue,
        anchor_step=max(1, args.anchor_step),
        large_gap_ms=int(args.large_gap_sec * 1000),
    )
    print(
        f"Alignment input: {len(en_dialogue)} EN dialogue cues, "
        f"{len(hu_dialogue)} HU dialogue cues, {len(hu_anchor_cues)} HU anchor candidates",
        file=sys.stderr,
    )

    if args.analyze_only:
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    anchors = find_anchors(
        en_dialogue=en_dialogue,
        hu_anchor_cues=hu_anchor_cues,
        model=args.model,
        batch_size=max(1, args.batch_size),
        search_window_ms=int(args.search_window_sec * 1000),
        min_confidence=args.min_confidence,
        retries=max(0, args.retries),
        cache_path=args.anchor_cache,
    )
    print(f"Reliable monotonic anchors: {len(anchors)}", file=sys.stderr)

    blocks = build_shift_blocks(
        anchors=anchors,
        all_hu=hu_cues,
        break_threshold_ms=int(args.break_threshold_sec * 1000),
    )
    for block in blocks:
        print(
            f"Block {block.block_id}: {format_time(block.start_hu_ms)}-"
            f"{format_time(block.end_hu_ms)} shift {block.shift_ms / 1000:+.3f}s "
            f"from {block.anchor_count} anchors",
            file=sys.stderr,
        )

    shifted = apply_blocks(
        hu_cues=hu_cues,
        blocks=blocks,
        drop_noise=not args.keep_noise,
        min_gap_ms=args.min_gap_ms,
    )
    write_srt(args.output, shifted)
    write_report(args.report, anchors, blocks, shifted)
    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"Wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
