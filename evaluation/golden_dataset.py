import csv
from pathlib import Path


def load_goldens(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.reader(stream))
    if rows and rows[0] == ['question', 'answer']:
        rows = rows[1:]
    if any(len(row) != 2 or not all(value.strip() for value in row) for row in rows):
        raise ValueError('Invalid golden row')
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('Duplicate golden question')
    return dict(rows)

