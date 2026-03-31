import csv
from pathlib import Path

import voice_session


def test_chatgpt_csv_has_no_literal_escaped_newlines():
    with Path("scripts/chatgpt.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    offenders = []
    for row_index, row in enumerate(rows, start=1):
        for col_index, cell in enumerate(row, start=1):
            if "\\n" in cell:
                offenders.append((row_index, col_index, cell))

    assert offenders == []


def test_normalize_chatgpt_string_converts_escaped_newlines():
    value = 'Line one\\n\\nLine two\\nLine three'

    normalized = voice_session.VoiceSession._normalize_chatgpt_string(value)

    assert normalized == "Line one\n\nLine two\nLine three"
