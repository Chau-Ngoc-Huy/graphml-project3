"""Freeze candidate captions and hard-negative suggestions before XAI scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


RELATION = re.compile(r"\b(on|under|behind|beside|next to|between|above|below|holding|holds|carrying|wearing|riding|sitting|standing|walking|playing|looking|in front of|inside|through|over|near|with)\b", re.I)
EXCLUDED_QUERY_TERMS = re.compile(r"\b(nude|naked|nudity)\b", re.I)


def _image_url(image):
    if image.get('url'):
        return image['url']
    if image.get('coco_file'):
        return 'https://images.cocodataset.org/' + image['coco_file']
    return image.get('coco_url', '')


def freeze_registry(bundle_dirs: dict[str, Path], baseline_selected: Path, output: Path,
                    per_group: int = 10, selection_seed: int = 20261121) -> Path:
    """Pick new source captions deterministically; leave all visual labels blank.

    Requires numpy and the bundle's own mmxai_eval.Protocol. The source
    direct/indirect flag is retained but is not treated as verified semantics.
    A negative is a review candidate based on object overlap, not gold.
    """
    import importlib.util
    import sys

    excluded = {(x['dataset'], int(x['caption_id'])) for x in json.loads(Path(baseline_selected).read_text())}
    rows = []
    for dataset, root in bundle_dirs.items():
        root = Path(root)
        spec = importlib.util.spec_from_file_location(f'mmxai_eval_{dataset}', root / 'mmxai_eval.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        protocol = module.Protocol.load(root)
        candidates = protocol.candidates('test', None)
        for group in ('direct', 'indirect'):
            picked = 0
            candidate_rows = list(range(len(candidates['q_cap'])))
            random.Random(selection_seed + (0 if dataset == 'vg' else 100) +
                          (0 if group == 'direct' else 1)).shuffle(candidate_rows)
            for j in candidate_rows:
                cap_id = int(candidates['q_cap'][j])
                if (dataset, cap_id) in excluded or str(candidates['group'][j]) != group:
                    continue
                cap = protocol.captions[cap_id]
                text = cap['text']
                if EXCLUDED_QUERY_TERMS.search(text) or not RELATION.search(text) or len(cap.get('concepts', [])) < 2:
                    continue
                gt_idx = int(candidates['q_img'][j])
                gt = protocol.images[gt_idx]
                concepts = set(map(str.lower, cap.get('concepts', [])))
                options = []
                for rank, image_idx in enumerate(candidates['cand'][j], 1):
                    image_idx = int(image_idx)
                    if image_idx == gt_idx:
                        continue
                    image = protocol.images[image_idx]
                    overlap = len(concepts & set(map(str.lower, image.get('objects', []))))
                    if overlap >= 2:
                        options.append((-overlap, rank, image_idx))
                if not options:
                    continue
                _, neg_rank, neg_idx = min(options)
                negative = protocol.images[neg_idx]
                rows.append({
                    'dataset': dataset, 'query_id': cap_id, 'candidate_row': j,
                    'query': text, 'source_group': group, 'verified_group': '',
                    'claim_subject': '', 'claim_predicate': '', 'claim_object': '',
                    'gt_image_id': gt_idx, 'gt_source_url': _image_url(gt),
                    'hard_negative_candidate_id': neg_idx,
                    'hard_negative_source_url': _image_url(negative),
                    'hard_negative_clip_rank': neg_rank,
                    'gt_relevance': '', 'negative_relevance': '',
                    'same_objects_different_relation': '', 'visual_reason': '',
                    'reviewer': '', 'review_status': 'pending',
                    'split': 'test', 'candidate_pool': 'test_full_top100',
                    'clip_embedding_row': cap_id,
                    'selection_seed': selection_seed,
                    'concepts_from_bundle': '|'.join(cap.get('concepts', [])),
                    'provenance': 'bundle_caption_and_test_candidate_pool',
                })
                picked += 1
                if picked == per_group:
                    break
            if picked < per_group:
                raise ValueError(f'{dataset}/{group}: only {picked} eligible query suggestions')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (output.with_suffix('.sha256')).write_text(hashlib.sha256(output.read_bytes()).hexdigest() + '  ' + output.name + '\n')
    return output


def validated_registry_rows(path: Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row['review_status'] != 'approved' or row['verified_group'] not in ('direct', 'relational'):
            raise ValueError(f"query {row['dataset']}/{row['query_id']} awaits human review")
        if row['gt_relevance'] != 'supported' or row['negative_relevance'] != 'contradicted':
            raise ValueError(f"query {row['dataset']}/{row['query_id']} lacks verified positive/negative")
        if row['same_objects_different_relation'] != 'supported':
            raise ValueError(f"query {row['dataset']}/{row['query_id']} lacks controlled hard negative")
        if not all(row[k].strip() for k in ('claim_subject', 'claim_predicate', 'claim_object',
                                           'visual_reason', 'reviewer')):
            raise ValueError(f"query {row['dataset']}/{row['query_id']} lacks claim or review provenance")
    for dataset in {r['dataset'] for r in rows}:
        subset = [r for r in rows if r['dataset'] == dataset]
        for group in ('direct', 'relational'):
            if sum(r['verified_group'] == group for r in subset) < 10:
                raise ValueError(f'{dataset}: fewer than 10 verified {group} queries')
    return rows


def read_case_review_workbook(workbook: Path) -> list[dict[str, str]]:
    """Read Case review cells from a workbook saved by this project or Excel."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(workbook) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared = [''.join(t.text or '' for t in item.findall('.//m:t', ns))
                      for item in root.findall('m:si', ns)]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    table = []
    for row in root.findall('.//m:sheetData/m:row', ns):
        cells = {}
        for cell in row.findall('m:c', ns):
            letters = ''.join(c for c in cell.attrib['r'] if c.isalpha())
            index = 0
            for letter in letters:
                index = index * 26 + ord(letter) - 64
            typ = cell.attrib.get('t')
            if typ == 'inlineStr':
                value = ''.join(t.text or '' for t in cell.findall('.//m:t', ns))
            else:
                node = cell.find('m:v', ns)
                value = node.text if node is not None and node.text is not None else ''
                if typ == 's' and value:
                    value = shared[int(value)]
            cells[index - 1] = value
        table.append(cells)
    headers = {v: k for k, v in table[0].items()}
    return [{name: str(cells.get(index, '')).strip() for name, index in headers.items()}
            for cells in table[1:]]


def import_reviewed_queries(registry: Path, workbook: Path, output: Path) -> Path:
    """Transfer human Excel judgements into a new CSV; keep the frozen source intact."""
    reviewed_rows = read_case_review_workbook(workbook)
    needed = ('dataset', 'query_id', 'verified_group', 'claim_subject', 'claim_predicate',
              'claim_object', 'review_status', 'reviewer', 'gt_relevance',
              'retrieved_relevance', 'hard_negative_same_objects', 'relation_differs',
              'visual_reason')
    if not reviewed_rows or not set(needed) <= set(reviewed_rows[0]):
        raise ValueError('review workbook is missing required case columns')
    reviews = {}
    for item in reviewed_rows:
        review = {name: item[name] for name in needed}
        key = (review['dataset'], review['query_id'])
        if key in reviews:
            raise ValueError(f'duplicate review row: {key}')
        reviews[key] = review
    with Path(registry).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if set(reviews) != {(r['dataset'], r['query_id']) for r in rows}:
        raise ValueError('review workbook query IDs differ from frozen registry')
    for row in rows:
        review = reviews[row['dataset'], row['query_id']]
        for key in ('verified_group', 'claim_subject', 'claim_predicate', 'claim_object',
                    'review_status', 'reviewer', 'gt_relevance', 'visual_reason'):
            row[key] = review[key]
        row['negative_relevance'] = review['retrieved_relevance']
        row['same_objects_different_relation'] = (
            'supported' if review['hard_negative_same_objects'] == 'supported'
            and review['relation_differs'] == 'supported' else 'uncertain')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (output.with_suffix('.sha256')).write_text(hashlib.sha256(output.read_bytes()).hexdigest() + '  ' + output.name + '\n')
    return output
