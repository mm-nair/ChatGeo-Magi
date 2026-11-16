import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = '01-03-Potemra_Geomag.pdf'
SOURCE_URL = 'https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V01-N03/01-03-Potemra_Geomag.pdf'
ANCHORS = {
    1: [(1, 'was conducted by William', 125), (1, 'in his De Magnete', 80)],
    2: [(1, 'the Magsat system to obtain', 155)],
    3: [(2, 'simple dipole', 90), (2, 'series of spherical harmonic functions', 90)],
    4: [(2, 'This unit of induction', 95), (2, 'In SI units', 55)],
    5: [(2, 'The intensity of the surface', 175)],
    6: [(2, 'A decrease in the moment', 120), (2, 'A westward precessional rotation', 110), (2, 'A rotation of the dipole toward', 110), (2, 'A westward drift of the nondipole', 105), (2, 'Growth and decay of features', 130)],
    7: [(3, 'Seismological evidence reveals', 200), (3, 'mainly of iron', 90)],
    8: [(4, 'but is severely distorted into a comet-shaped', 160), (4, 'magnetic field configuration called', 110)],
    9: [(4, 'called the Chapman-Ferraro', 100), (4, '"Birkeland"', 145), (4, 'the ring current', 110)],
    10: [(4, 'Widespread magnetic disturbances', 180), (4, 'The solar plasma accompanying', 145)],
    11: [(4, 'satellites in 1972', 210)],
    12: [(5, 'amount to make the orbit plane follow', 220)],
    13: [(7, 'An electronic and optical system', 160)],
    14: [(6, 'The vector sensor consisted of three', 225)],
    15: [(6, 'It used Zeeman', 150)],
    16: [(6, 'satellite upside down', 200), (6, 'the zero-g calibration', 125)],
    17: [(7, 'We experienced some operational problems', 240), (7, 'The sunshades of the star cameras', 110)],
    18: [(8, 'This rotation provided angular momentum', 175)],
    19: [(8, 'a pendulous mechanical damper', 110), (8, 'closed-loop attitude control', 110)],
    20: [(5, 'near-zero coefficient of thermal', 100), (8, 'Optical Bench - A structural', 240)],
    21: [(9, 'Transponder - A combined', 130), (9, 'for transmitting the telemetry', 110)],
    25: [(9, 'Tape Recorder - A device', 150)],
}


def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('\xad\n', '').replace('\xad', '')).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=ROOT / 'revision_workspace/retrieval_evaluation/chunk_catalog.json')
    parser.add_argument('--pages', type=Path, default=ROOT / 'revision_workspace/retrieval_evaluation/potemra_pages.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'revision_workspace/retrieval_evaluation/annotated_benchmark.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'revision_workspace'):
        parser.error('Output must remain in revision_workspace')
    catalog = json.loads(args.catalog.read_text())
    pages = [normalize(page) for page in json.loads(args.pages.read_text())]
    benchmark = json.loads((ROOT / 'evaluation/benchmarks/retrieval.json').read_text())
    for question in benchmark:
        number = int(question['id'][1:])
        if number not in ANCHORS:
            question['annotation_note'] = 'Outside source-checked 22-question subset; no ground truth assigned.'
            continue
        facts = []
        for page, anchor, length in ANCHORS[number]:
            start = pages[page - 1].find(anchor)
            if start < 0:
                raise ValueError(f'{question["id"]}: missing source anchor {anchor}')
            quote = pages[page - 1][start:start + length]
            candidates = [chunk for chunk in catalog if Path(chunk['metadata']['source']).name == SOURCE_NAME and chunk['metadata']['page'] == page - 1]
            chunk_ids = [chunk['chunk_id'] for chunk in candidates if quote in normalize(chunk['text'])]
            if not chunk_ids:
                raise ValueError(f'{question["id"]}: source quote crosses every chunk boundary: {anchor}')
            facts.append({'passage': quote, 'page_1based': page, 'chunk_ids': chunk_ids})
        question.update({'source_document': SOURCE_NAME, 'source_url': SOURCE_URL, 'supporting_passage': '\n'.join(fact['passage'] for fact in facts), 'relevant_chunk_ids': sorted({identifier for fact in facts for identifier in fact['chunk_ids']}), 'support_facts': facts, 'verification_status': 'source_verified', 'annotation_note': 'Machine/source checked against original public PDF before retrieving benchmark queries; not human verified. Relevant chunks contain complete annotated evidence excerpts. Coverage measures these excerpts, not exhaustive semantic answer entailment.'})
    args.output.write_text(json.dumps(benchmark, indent=2))
    print('Annotated', sum(question['verification_status'] == 'source_verified' for question in benchmark))


if __name__ == '__main__':
    main()
