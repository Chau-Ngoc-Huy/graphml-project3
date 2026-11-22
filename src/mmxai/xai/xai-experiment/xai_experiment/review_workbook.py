"""Create a self-contained Excel review pack without optional dependencies.

The workbook embeds local GT/retrieved JPEGs. Existing model labels and path
hints are context, never gold evidence.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .image_download import _candidates


def _col(index: int) -> str:
    result = ""
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def _cell(value, address: str, style: int = 0) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{address}" s="{style}"><v>{value}</v></c>'
    if isinstance(value, str) and value.startswith(('https://', 'http://')):
        url = escape(_candidates(value)[0].replace('"', '""'))
        return f'<c r="{address}" s="{style}"><f>HYPERLINK("{url}","open image")</f><v></v></c>'
    safe = escape(str(value), {'"': '&quot;'})
    return f'<c r="{address}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{safe}</t></is></c>'


def _sheet(rows, widths, *, image_rows=(), dropdown_columns=()):
    cols = ''.join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
                   for i, w in enumerate(widths, 1))
    row_xml = []
    for r, row in enumerate(rows, 1):
        cells = ''.join(_cell(v, f'{_col(c)}{r}', 1 if r == 1 else 0)
                        for c, v in enumerate(row, 1))
        height = ' ht="145" customHeight="1"' if r in image_rows else ''
        row_xml.append(f'<row r="{r}"{height}>{cells}</row>')
    last = f'{_col(max(len(x) for x in rows))}{len(rows)}'
    validations = ''
    if dropdown_columns:
        rules = ''.join(
            f'<dataValidation type="list" allowBlank="1" sqref="{_col(c)}2:{_col(c)}{len(rows)}">'
            '<formula1>"supported,contradicted,uncertain"</formula1></dataValidation>'
            for c in dropdown_columns)
        validations = f'<dataValidations count="{len(dropdown_columns)}">{rules}</dataValidations>'
    drawing = '<drawing r:id="rId1"/>' if image_rows else ''
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<dimension ref="A1:{last}"/><sheetViews><sheetView workbookViewId="0"/></sheetViews>'
            f'<sheetFormatPr defaultRowHeight="15"/><cols>{cols}</cols>'
            f'<sheetData>{"".join(row_xml)}</sheetData>{validations}{drawing}</worksheet>')


def create_review_workbook(cases: list[dict], steps: list[dict], output: Path,
                           image_root: Path) -> Path:
    """Write workbook with one case row and one independent judgement row per step."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    case_headers = ["dataset", "query_id", "query", "source_group", "verified_group",
                    "claim_subject", "claim_predicate", "claim_object", "review_status", "model_seed",
                    "gt_image_id", "retrieved_image_id", "retrieval_outcome",
                    "gt_rank", "target_rank", "explanation_level", "path_or_edge_text",
                    "gt_source_url", "retrieved_source_url", "GT image", "Retrieved image",
                    "reviewer", "gt_relevance", "retrieved_relevance", "hard_negative_same_objects",
                    "relation_differs", "visual_reason", "review_timestamp",
                    "validation_origin", "vlm_confidence", "vlm_uncertainty",
                    "vlm_image_a_observation", "vlm_image_b_observation"]
    case_rows = [case_headers]
    drawings = []
    media = []
    for idx, case in enumerate(cases, 2):
        row = [case.get(k, "") for k in ("dataset", "query_id", "query", "source_group",
               "verified_group", "claim_subject", "claim_predicate", "claim_object", "review_status", "model_seed",
               "gt_image_id", "retrieved_image_id", "retrieval_outcome", "gt_rank", "target_rank",
               "explanation_level", "path_or_edge_text", "gt_source_url", "retrieved_source_url")]
        row.extend(["embedded below", "embedded below"] +
                   [case.get(k, "") for k in ("reviewer", "gt_relevance", "retrieved_relevance",
                                               "hard_negative_same_objects", "relation_differs",
                                               "visual_reason", "review_timestamp",
                                               "validation_origin", "vlm_confidence", "vlm_uncertainty",
                                               "vlm_image_a_observation", "vlm_image_b_observation")])
        case_rows.append(row)
        for col, field in ((19, "gt_image_file"), (20, "retrieved_image_file")):
            path_text = case.get(field, "")
            if not path_text:
                case_rows[-1][col] = case.get('gt_source_url' if col == 19 else 'retrieved_source_url', '')
                continue
            path = Path(path_text)
            path = path if path.is_absolute() else image_root / path
            if not path.is_file():
                case_rows[-1][col] = case.get('gt_source_url' if col == 19 else 'retrieved_source_url', '')
                continue
            name = f'image{len(media) + 1}.jpg'
            media.append((name, path.read_bytes()))
            drawings.append((idx - 1, col, len(media), name))
    step_headers = ["dataset", "query_id", "target_image_id", "path_id", "step_index",
                    "edge_id", "source_node", "predicate_hint", "target_node", "edge_type",
                    "edge_provenance", "model_score_drop", "reviewer", "object_supported",
                    "visual_evidence", "step_decision", "support_source", "evidence_reason"]
    step_rows = [step_headers]
    for step in steps:
        step_rows.append([step.get(k, "") for k in ("dataset", "query_id", "target_image_id",
                         "path_id", "step_index", "edge_id", "source_node", "predicate_hint",
                         "target_node", "edge_type", "edge_provenance", "model_score_drop")]
                         + ["", "", "", "", "", ""])
    rubric = [
        ["Field", "Rule"],
        ["Purpose", "Judge visual and logical support independently of model fidelity and graph structure."],
        ["Direct object mode", "For a contains edge, verify the object in the image and its relevance to the query. Predicate direction and full-caption relation are outside this mode."],
        ["GT/retrieved relevance", "supported = image satisfies the full query; contradicted = does not; uncertain = cannot tell."],
        ["GT ID mismatch", "A top-1 image with a different GT ID may still satisfy the query. Judge retrieved relevance separately from exact-ID rank."],
        ["Hard negative", "Optional for direct retrieval; if reported, confirm the same objects and a different relation or role visually."],
        ["Predicate hint", "Display only. Current QA scorer does not consume predicate identity."],
        ["Object supported", "Check source and target objects independently in the image or independent annotation."],
        ["Visual evidence", "Check the object in the image. A contains edge does not prove the caption's action or relation."],
        ["Review provenance", "Record the reviewer and visual reason for each judgement. Agreement or adjudication is not required in this protocol."],
        ["Provenance", "Source URL and model labels are context, not independent ground truth."],
    ]
    ns = 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing'
    anchors = []
    for row0, col0, media_id, _ in drawings:
        anchors.append(
            f'<xdr:oneCellAnchor><xdr:from><xdr:col>{col0}</xdr:col><xdr:colOff>0</xdr:colOff>'
            f'<xdr:row>{row0}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>'
            '<xdr:ext cx="1714500" cy="1714500"/>'
            f'<xdr:pic><xdr:nvPicPr><xdr:cNvPr id="{media_id}" name="image{media_id}"/>'
            '<xdr:cNvPicPr/></xdr:nvPicPr><xdr:blipFill>'
            f'<a:blip r:embed="rId{media_id}"/><a:stretch><a:fillRect/></a:stretch>'
            '</xdr:blipFill><xdr:spPr><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</xdr:spPr></xdr:pic><xdr:clientData/></xdr:oneCellAnchor>')
    drawing_xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   f'<xdr:wsDr xmlns:xdr="{ns}" '
                   'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                   + ''.join(anchors) + '</xdr:wsDr>')
    sheets = [
        _sheet(case_rows, [12, 14, 55, 16, 18, 20, 22, 20, 16, 12, 16, 20, 18, 11, 12,
                           24, 65, 38, 38, 28, 28, 18, 18, 20, 22, 18, 60, 22,
                           20, 16, 50, 55, 55],
               image_rows=range(2, len(case_rows) + 1),
               dropdown_columns=(23, 24, 25, 26)),
        _sheet(step_rows, [12, 14, 20, 30, 12, 14, 30, 28, 30, 15, 28, 18, 18,
                           20, 22, 20, 25, 55],
               dropdown_columns=(14, 15, 16)),
        _sheet(rubric, [27, 110]),
    ]
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Default Extension="jpg" ContentType="image/jpeg"/>'
                     '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                     '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                     + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, 4))
                     + '<Override PartName="/xl/drawings/drawing1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
                     '</Types>')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   '<sheet name="Case review" sheetId="1" r:id="rId1"/><sheet name="Step review" sheetId="2" r:id="rId2"/>'
                   '<sheet name="Rubric" sheetId="3" r:id="rId3"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, 4))
                   + '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml', '<?xml version="1.0"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                   '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
                   '<fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders>'
                   '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                   '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                   '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>')
        for i, sheet in enumerate(sheets, 1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', sheet)
        z.writestr('xl/worksheets/_rels/sheet1.xml.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/drawing1.xml"/></Relationships>')
        z.writestr('xl/drawings/drawing1.xml', drawing_xml)
        z.writestr('xl/drawings/_rels/drawing1.xml.rels', '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{name}"/>'
                             for i, (name, _) in enumerate(media, 1)) + '</Relationships>')
        for name, data in media:
            z.writestr(f'xl/media/{name}', data)
    return output


def baseline_review_pack(snapshot: Path, output: Path) -> Path:
    """Package ten legacy case studies for blind visual review; no new labels."""
    snapshot = Path(snapshot)
    analysis = snapshot / 'analysis'
    with (analysis / 'case_studies_10.csv').open(newline='', encoding='utf-8') as f:
        source_cases = list(csv.DictReader(f))
    with (analysis / 'representative_cases.csv').open(newline='', encoding='utf-8') as f:
        representatives = {(r['dataset'], r['caption_id']): r for r in csv.DictReader(f)}
    explanations = json.loads((snapshot / 'explanations.json').read_text())
    index = {(x['dataset'], str(x['caption_id'])): x['explanation'] for x in explanations
             if int(x['seed']) == 0 and int(x['requested_k']) == 3}
    cases, steps = [], []
    for item in source_cases:
        dataset, query_id = item['dataset'], item['caption_id']
        exp = index.get((dataset, query_id), {})
        representative = representatives.get((dataset, query_id), {})
        cases.append({
            'dataset': dataset, 'query_id': query_id, 'query': item['query'],
            'source_group': item['group'], 'model_seed': 0,
            'gt_image_id': item['ground_truth_image_id'],
            'retrieved_image_id': item['top1_image_id'],
            'retrieval_outcome': item['seed0_outcome'],
            'gt_rank': item['gt_rank_before'], 'target_rank': 1,
            'explanation_level': 'legacy_top_edges; no semantic claim',
            'path_or_edge_text': representative.get('edges', ''),
            'gt_source_url': item['gt_source_url'],
            'retrieved_source_url': item['top1_source_url'],
            'gt_image_file': item['gt_image_file'],
            'retrieved_image_file': item['top1_image_file'],
        })
        for i, edge in enumerate(exp.get('edges', []), 1):
            if edge['id'] not in exp.get('edge_ids', []):
                continue
            steps.append({
                'dataset': dataset, 'query_id': query_id,
                'target_image_id': item['top1_image_id'], 'path_id': 'legacy_top_edges',
                'step_index': i, 'edge_id': edge['id'],
                'source_node': edge['source'],
                'predicate_hint': edge.get('metadata', {}).get('pair_relation_hint', ''),
                'target_node': edge['target'], 'edge_type': edge['relation'],
                'edge_provenance': edge.get('provenance', ''),
                'model_score_drop': next((x['final_contribution'] for x in exp.get('contributions', [])
                                          if x['edge_id'] == edge['id']), ''),
            })
    return create_review_workbook(cases, steps, output, analysis)
