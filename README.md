# SRT block alignment

This folder contains `align_srt_blocks.py`, a Debian/venv-friendly script that
retimes a Hungarian subtitle to an English reference subtitle.

The script uses the OpenAI API to find semantic anchor points, then applies
piecewise constant time shifts to the Hungarian subtitle. It keeps the Hungarian
wording and cue durations, and ignores music/SFX-like cues while aligning.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key either as a normal environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

or put it in a local `.env` file next to the script:

```bash
OPENAI_API_KEY=sk-...
```

Optional model override:

```bash
export OPENAI_MODEL="gpt-4.1-mini"
```

You can also pass `--model` explicitly.

## Example

```bash
python align_srt_blocks.py \
  --english "/path/to/episode.en.srt" \
  --hungarian "/path/to/episode.hu.srt" \
  --output "episode.hu.aligned.srt" \
  --report "alignment_report.csv"
```

For subtitle versions that may differ by larger ad-break cuts, increase the
search window:

```bash
python align_srt_blocks.py \
  --english "/path/to/episode.en.srt" \
  --hungarian "/path/to/episode.hu.srt" \
  --output "episode.hu.aligned.srt" \
  --search-window-sec 1200
```

## Useful options

- `--analyze-only`: read both SRT files and print stats without API calls.
- `--anchor-step 2`: ask the API to inspect more Hungarian anchor candidates.
- `--break-threshold-sec 2.5`: lower value creates more shift blocks.
- `--min-confidence 0.62`: lower value keeps more uncertain anchors.
- `--keep-noise`: keep music/SFX-like cues from the Hungarian source.
- `--anchor-cache anchor_cache.json`: cache API match results for repeat runs.

Inspect `alignment_report.csv` after a run. The `blocks` section is the most
important part: it shows where the script found timing blocks and how many
anchors supported each shift.
