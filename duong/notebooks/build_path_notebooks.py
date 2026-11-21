"""Build reproducible, output-free XAI notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def M(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def C(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
def save(name,cells):
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (ROOT/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')

save('04_direct_query_audit.ipynb',[
M('''# A–B · Audit baseline và kiểm chứng direct retrieval

**Mục tiêu:** giữ nguyên snapshot 24 caption, kiểm tra 40 ứng viên đã khóa và chọn các query direct có ảnh dương được người chấm xác minh. `source_group` là nhãn bundle, không phải mức suy luận đã kiểm chứng. Ảnh âm chỉ là ứng viên tham khảo. Notebook không tái tạo hoặc ghi đè workbook đã điền.
'''),
C('''from pathlib import Path
import hashlib, sys
import pandas as pd
ROOT = Path('/content/drive/MyDrive') if Path('/content/drive/MyDrive/graphml-project3').is_dir() else next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'graphml-project3').is_dir())
DUONG = ROOT/'graphml-project3/duong'
sys.path.insert(0,str(DUONG))
from xai.review_workbook import baseline_review_pack, create_review_workbook
from xai.image_download import fetch_review_image
from xai.query_registry import validated_registry_rows, import_reviewed_queries, read_case_review_workbook, direct_eligibility_reason
SNAP = DUONG/'runs/qa_vg_psg_xai/xai-20261121-130842'
REG = DUONG/'registry/query_candidates_frozen.csv'
assert SNAP.is_dir() and REG.is_file()
print('Frozen registry SHA256:',hashlib.sha256(REG.read_bytes()).hexdigest())
'''),
M('''## 1 · Audit snapshot cũ

Kiểm tra 216 dòng, 12 query/bộ × 3 seed × 3 mức k. Đây là leave-one-edge-out; path validity và evidence được báo riêng. Mẫu số sparsity cũ chỉ là vùng sàng lọc.'''),
C('''old=pd.read_csv(SNAP/'analysis/xai_results_corrected.csv')
assert len(old)==216 and old.groupby('dataset').caption_id.nunique().to_dict()=={'vg':12,'psg':12}
display(old[old.requested_k==3].groupby('dataset').agg(n=('caption_id','size'),query_count=('caption_id','nunique'),screen_coverage=('screen_coverage','mean'),final_drop=('final_score_drop','mean'),random_drop=('random_final_score_drop_mean','mean'),evidence_known=('evidence_known_rate','mean')))
'''),
M('''## 2 · Cổng query/ảnh direct

40 ứng viên được chọn trước khi xem XAI. Chỉ `verified_group=direct`, `gt_relevance=supported` và `review_status=approved` với người chấm rõ ràng mới vào lượt mới; cần ít nhất 10 query mỗi dataset. Ảnh âm khác quan hệ không còn là điều kiện của thí nghiệm này. Giữ mọi nhãn nguồn và kết quả không đủ điều kiện để báo cáo attrition.
'''),
C('''registry=pd.read_csv(REG,keep_default_na=False)
assert len(registry)==40
display(registry.groupby(['dataset','source_group']).size())
review_path=DUONG/'review/query_candidates_40.xlsx'
review_rows=read_case_review_workbook(review_path)
review=pd.DataFrame(review_rows)
display(review.groupby(['dataset','verified_group','review_status'],dropna=False).size())
review['direct_gt_candidate']=(review.verified_group=='direct') & (review.gt_relevance=='supported')
review['eligibility_reason']=review.apply(direct_eligibility_reason,axis=1)
display(review.groupby(['dataset','eligibility_reason']).size())
display(review.groupby('dataset').agg(direct_gt_candidates=('direct_gt_candidate','sum'),approved=('review_status',lambda x:(x=='approved').sum())))
print('Direct with GT supported:',review.direct_gt_candidate.sum(),'/ 40')
'''),
M('''## 3 · Xuất registry đã duyệt

`query_candidates_40.xlsx` chứa 40 nhãn đã duyệt. Notebook xuất registry đầy đủ và audit lý do chọn/loại; 34 query direct có GT `supported` đi tiếp. Nếu sửa nhãn, chạy lại cell này để làm mới CSV.
'''),
C('''REVIEWED_WORKBOOK=DUONG/'review/query_candidates_40.xlsx'
assert REVIEWED_WORKBOOK.is_file(), REVIEWED_WORKBOOK
EXPORT_DIRECT_REGISTRY=True  # workbook chuẩn đã được duyệt
if EXPORT_DIRECT_REGISTRY:
    verified=import_reviewed_queries(REG,REVIEWED_WORKBOOK,DUONG/'registry/queries_direct_verified.csv')
    audit=pd.read_csv(verified,keep_default_na=False)
    audit['eligibility_reason']=audit.apply(direct_eligibility_reason,axis=1)
    audit.to_csv(DUONG/'registry/direct_query_eligibility_audit.csv',index=False)
    selected=validated_registry_rows(verified)
    print('Approved direct queries:',len(selected))
    display(pd.DataFrame(selected).groupby('dataset').size())
else:
    print('Workbook hiện tại được giữ nguyên. Chưa xuất registry đã duyệt.')
'''),
])

save('05_direct_object_xai_colab.ipynb',[
M('''# C–D · Direct retrieval: can thiệp cạnh đối tượng

**Câu hỏi:** các cạnh `contains` nối khái niệm query với ảnh top-1 có ảnh hưởng tới điểm và hạng trên cùng top-100 không? Chạy Colab GPU. Mọi query, kể cả không có cạnh phù hợp, nằm trong mẫu số coverage. Kết quả đo tác động của bằng chứng đối tượng; cạnh `contains` không tự chứng minh toàn bộ hành động hay vị trí trong caption.
'''),
C('''from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, importlib.util, json, subprocess, sys
import numpy as np
if Path('/content').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT=Path('/content/drive/MyDrive')
else:
    ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'graphml-project3').is_dir())
REPO=ROOT/'graphml-project3'; DUONG=REPO/'duong'; RUNS=DUONG/'runs/qa_vg_psg_xai'
sys.path[:0]=[str(DUONG),str(REPO/'huy/src')]
if importlib.util.find_spec('torch_geometric') is None:
    subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(REPO/'huy')],check=True)
import torch
from xai.model import HuyQAPathCaseModel, MODEL_SPECS
from xai.path_protocol import evaluate_case
from xai.query_registry import validated_registry_rows
from xai.perturbation import rank_of
from mmxai.data.task1 import load_task1
from mmxai.graph.build import build_graph
from mmxai.graph.qa import QAData, QAScorer, qa_scores
from mmxai.experiments.t1_graph import load_eval_module
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',DEVICE)
'''),
M('''## 1 · Chốt protocol và provenance

Mặc định chạy 34 query direct đã duyệt từ notebook 01; dùng `baseline24` nếu muốn kiểm tra code trên tập cũ. Budget 1/2/3 là số cạnh `contains` duy nhất. Random control chọn cạnh `contains` khác trên **cùng ảnh đích** và cùng budget; thiếu đối chứng độc lập ghi N/A. Hash bundle/checkpoint được xác minh; lượt mới không ghi đè baseline.
'''),
C('''QUERY_MODE='verified_direct'  # approved direct set; use 'baseline24' for smoke test
MODEL_SEEDS=(0,1,2); PATH_SCREEN_LIMITS=(40,); EDGE_BUDGETS=(1,2,3)
RANDOM_REPEATS=20; RANDOM_SEED=20261121
MANIFEST=RUNS/'dual-20261121-125423_manifest.json'
manifest=json.loads(MANIFEST.read_text())
records={(r['dataset'],int(r['seed'])):r for r in manifest['runs']}
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
for rel,digest in manifest['source_sha256'].items():
    if rel.startswith('huy/'): assert sha256(REPO/rel)==digest,rel
for dataset,spec in MODEL_SPECS.items():
    bundle_dir=ROOT/spec.bundle_dir
    for rel,digest in manifest['bundle_sha256'][dataset].items():
        assert sha256(bundle_dir/rel)==digest,(dataset,rel)
    for seed in MODEL_SEEDS:
        rec=records[dataset,seed]
        ckpt=RUNS/spec.experiment/Path(rec['run_dir']).name/'model.pt'
        assert sha256(ckpt)==rec['checkpoint_sha256']
assert QUERY_MODE in ('baseline24','verified_direct')
if QUERY_MODE=='baseline24':
    selected=json.loads((RUNS/'xai-20261121-130842/selected_cases.json').read_text())
else:
    selected=validated_registry_rows(DUONG/'registry/queries_direct_verified.csv')
assert len(selected)==24 if QUERY_MODE=='baseline24' else len(selected)>=20
for dataset,spec in MODEL_SPECS.items():
    seed0_dir=RUNS/spec.experiment/Path(records[dataset,0]['run_dir']).name
    ranks0=json.loads((seed0_dir/'ranks_test_full.json').read_text())['ranks']
    subset=[r for r in selected if r['dataset']==dataset]
    success=sum(int(ranks0[int(r['candidate_row'])])==1 for r in subset)
    failure=sum(int(ranks0[int(r['candidate_row'])])>1 for r in subset)
    print(dataset,'seed-0 success/failure:',success,failure)
    if success<5 or failure<5: print('Case-study quota unavailable for',dataset,'; main run continues')
OUT=RUNS/('path-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'protocol.json').write_text(json.dumps({'query_mode':QUERY_MODE,'explanation_mode':'object','manifest':str(MANIFEST),'seeds':MODEL_SEEDS,'path_screen_limits':PATH_SCREEN_LIMITS,'edge_budgets':EDGE_BUDGETS,'random_repeats':RANDOM_REPEATS,'random_seed':RANDOM_SEED,'selected_query_ids':{d:[int(r['caption_id'] if QUERY_MODE=='baseline24' else r['query_id']) for r in selected if r['dataset']==d] for d in MODEL_SPECS}},indent=2))
print('Output:',OUT)
'''),
M('''## 2 · Score gốc và cạnh đối tượng đủ điều kiện

`E_eligible` gồm mọi cạnh `contains` của ảnh đích và vùng nhãn–nhãn có thể ảnh hưởng đến scorer, nhưng explanation chính chỉ chọn cạnh `contains` khớp khái niệm query. Vùng screen không dùng score can thiệp hoặc relevance test. Kiểm tra QA score và hạng GT so với checkpoint cho từng query/seed.
'''),
C('''result_rows=[]; path_rows=[]; case_rows=[]
for dataset,spec in MODEL_SPECS.items():
    bundle_dir=ROOT/spec.bundle_dir
    ev=load_eval_module(bundle_dir); protocol=ev.Protocol.load(bundle_dir)
    bundle=load_task1(bundle_dir,bundle_dir/'huy_extra_cn')
    cap_concepts={int(c['caption_id']):c['concepts'] for c in bundle.captions}
    cands=protocol.candidates('test',None)
    chosen=[r for r in selected if r['dataset']==dataset]
    image_tensor=torch.from_numpy(bundle.image_emb).to(DEVICE)
    for seed in MODEL_SEEDS:
        rec=records[dataset,seed]
        run_dir=RUNS/spec.experiment/Path(rec['run_dir']).name
        cfg=json.loads((run_dir/'config.json').read_text())
        graph=build_graph(bundle,variant=cfg['graph']['variant'],min_rel_count=int(cfg['graph'].get('min_rel_count',1)),seed=seed,rel_merge=cfg['graph'].get('rel_merge','directed'))
        assert cfg['graph']['variant']==spec.graph_variant and cfg['gnn']['kind']==spec.gnn_kind
        assert graph.stats['n_label']==spec.expected_labels and graph.stats['E_concept']==spec.expected_concept_edges
        qd=QAData(graph,cap_concepts,DEVICE)
        model=QAScorer(bundle.image_emb.shape[1],**cfg['gnn']).to(DEVICE)
        model.load_state_dict(torch.load(run_dir/'model.pt',map_location='cpu',weights_only=True)); model.eval()
        saved_ranks=json.loads((run_dir/'ranks_test_full.json').read_text())['ranks']
        for item in chosen:
            j=int(item['candidate_row']); cap_id=int(item['caption_id'] if QUERY_MODE=='baseline24' else item['query_id'])
            assert int(cands['q_cap'][j])==cap_id
            query=protocol.captions[cap_id]['text']
            assert QUERY_MODE=='baseline24' or query==item['query']
            gt_id=str(int(cands['q_img'][j]))
            group=str(cands['group'][j]) if QUERY_MODE=='baseline24' else item['source_group']
            for screen_limit in PATH_SCREEN_LIMITS:
                case=HuyQAPathCaseModel(model,graph,qd,bundle,protocol,cands,ev,j,rec['alpha'],DEVICE,max_paths=screen_limit,path_mode='object',image_tensor=image_tensor)
                reference=qa_scores(model,qd,image_tensor,np.asarray(protocol.cap[cap_id:cap_id+1]),np.asarray([cap_id]),case.rows_np[None],DEVICE)[0]
                np.testing.assert_allclose(case.score_removed()['qa'],reference,rtol=1e-4,atol=1e-5)
                ids=tuple(map(str,case.ids)); before=case.scorer.score(query,ids,case.intervention_graph)
                if gt_id in ids: assert rank_of(before,gt_id,ids)==int(saved_ranks[j])
                else: assert int(saved_ranks[j])>len(ids)
                rr,pp=evaluate_case(case,query,gt_id,dataset=dataset,model_seed=seed,query_id=str(cap_id),group=group,budgets=EDGE_BUDGETS,random_repeats=RANDOM_REPEATS,random_seed=RANDOM_SEED+seed*100000+j*10+screen_limit,hard_negative_image_id=None,explanation_mode='object')
                for r in rr: r['screen_path_limit']=screen_limit
                for p in pp: p.update(dataset=dataset,model_seed=seed,screen_path_limit=screen_limit,query=query,ground_truth_image_id=gt_id)
                result_rows.extend(rr); path_rows.extend(pp)
                def source_url(im): return im.get('url') or ('https://images.cocodataset.org/'+im['coco_file'] if im.get('coco_file') else im.get('coco_url',''))
                case_rows.append({'dataset':dataset,'query_id':cap_id,'query':query,'source_group':group,'model_seed':seed,'screen_path_limit':screen_limit,'gt_image_id':gt_id,'retrieved_image_id':case.top_id,'exact_id_outcome':'match' if case.top_id==int(gt_id) else 'mismatch','retrieval_outcome':'success' if case.top_id==int(gt_id) else 'failure','gt_rank':int(saved_ranks[j]),'target_rank':1,'gt_source_url':source_url(protocol.images[int(gt_id)]),'retrieved_source_url':source_url(protocol.images[case.top_id])})
        print('Completed:',dataset,seed,len(chosen),'queries')
'''),
M('''## 3 · Lưu artifact bất biến

`paths.jsonl` lưu ứng viên cạnh đối tượng, đóng góp và lý do chọn. `xai_results.csv` lưu can thiệp thật, random cùng ảnh đích, score/hạng trước–sau và mẫu số cạnh đủ điều kiện. `case_manifest.csv` giữ ảnh cho workbook chấm. Lượt baseline cũ không bị ghi đè.
'''),
C('''assert result_rows and path_rows and case_rows
with (OUT/'xai_results.csv').open('w',newline='',encoding='utf8') as f:
    keys=list(dict.fromkeys(k for row in result_rows for k in row)); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(result_rows)
with (OUT/'case_manifest.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(case_rows[0])); w.writeheader(); w.writerows(case_rows)
with (OUT/'paths.jsonl').open('w',encoding='utf8') as f:
    for row in path_rows: f.write(json.dumps(row,ensure_ascii=False)+'\\n')
(OUT/'complete.json').write_text(json.dumps({'n_result_rows':len(result_rows),'n_path_rows':len(path_rows),'n_cases':len(case_rows)},indent=2))
print('Saved:',OUT,len(result_rows),len(path_rows),len(case_rows))
'''),
])

save('06_direct_xai_analysis.ipynb',[
M('''# E · Direct retrieval: fidelity, coverage và căn cứ ảnh

Notebook CPU đọc lượt `path-*` có `explanation_mode=object`. Bootstrap theo query, giữ seed trong cụm; coverage gồm cả query không có cạnh đối tượng phù hợp. Fidelity so với cạnh `contains` khác trên cùng ảnh đích khi có đối chứng độc lập. Chấm ảnh riêng để biết đối tượng có thật; score drop không phải validity ngữ nghĩa của toàn caption. `failure` trong benchmark cũ chỉ nghĩa là top-1 khác ID GT; ảnh đó vẫn có thể phù hợp với query. Các output lưu trong notebook từ lượt cũ cần chạy lại sau thay đổi này.
'''),
C('''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path('/content/drive/MyDrive') if Path('/content/drive/MyDrive/graphml-project3').is_dir() else next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'graphml-project3').is_dir())
DUONG=ROOT/'graphml-project3/duong'; RUNS=DUONG/'runs/qa_vg_psg_xai'
sys.path.insert(0,str(DUONG))
from xai.review_workbook import create_review_workbook
from xai.image_download import fetch_review_image
RESULTS_DIR=None  # set an explicit completed path-* directory for the report
if RESULTS_DIR is None:
    complete=sorted(p.parent for p in RUNS.glob('path-*/complete.json')
                    if json.loads((p.parent/'protocol.json').read_text()).get('explanation_mode')=='object')
    assert complete,'Run notebook 02 in object mode first.'
    RESULTS_DIR=complete[-1]
RESULTS_DIR=Path(RESULTS_DIR)
assert json.loads((RESULTS_DIR/'protocol.json').read_text()).get('explanation_mode')=='object'
results=pd.read_csv(RESULTS_DIR/'xai_results.csv')
cases=pd.read_csv(RESULTS_DIR/'case_manifest.csv')
cases['exact_id_outcome']=np.where(cases.gt_image_id.astype(str)==cases.retrieved_image_id.astype(str),'match','mismatch')
if 'retrieval_outcome' in cases:
    legacy=np.where(cases.retrieval_outcome=='success','match','mismatch')
    assert (legacy==cases.exact_id_outcome).all(), 'Legacy success/failure must mean exact GT ID match only'
paths=[json.loads(line) for line in (RESULTS_DIR/'paths.jsonl').open(encoding='utf8')]
meta=json.loads((RESULTS_DIR/'complete.json').read_text())
assert len(results)==meta['n_result_rows'] and len(paths)==meta['n_path_rows']
analysis=RESULTS_DIR/'analysis'; analysis.mkdir(exist_ok=True)
print(RESULTS_DIR,len(results),len(cases),len(paths))
'''),
M('''## 1 · Mẫu số, object yield và đối chứng

Mỗi `dataset × budget` giữ toàn bộ case đã đăng ký. `exact_id_outcome` chỉ so ID top-1 với GT và không đo mức phù hợp ngữ nghĩa. Sparsity = 1 − số cạnh explanation duy nhất / `E_eligible_contains`; báo thêm số cạnh `contains` đủ điều kiện. Random control chỉ lấy cạnh `contains` khác trên chính ảnh đích. Thiếu cạnh đối chứng thì fidelity advantage là N/A, không loại case khỏi coverage.
'''),
C('''main=results[results.intervention=='object_path'].copy()
assert not main.duplicated(['dataset','query_id','model_seed','screen_path_limit','budget']).any()
assert (main.n_screened_contains<=main.n_eligible_explanation).all() and (main.n_removed_unique<=main.budget).all()
main['matched_advantage']=main.final_score_drop-main.random_final_drop_mean
coverage=main.groupby(['dataset','screen_path_limit','budget']).agg(n_cases=('query_id','size'),n_queries=('query_id','nunique'),positive_object_rate=('has_positive_object_path','mean'),selected_path_rate=('n_paths',lambda x:(x>0).mean()),no_explanation_rate=('status',lambda x:(x!='ok').mean()),structural_no_path_rate=('status',lambda x:x.isin(['no_object_path','no_path']).mean()),mean_eligible=('n_eligible_explanation','mean'),mean_screen_coverage=('screen_coverage','mean'),mean_sparsity=('sparsity','mean'),matched_cases=('random_n',lambda x:(x>0).sum()),unique_random_sets=('random_unique_n','sum'),mean_fidelity_advantage=('matched_advantage','mean')).reset_index()
display(coverage)
exact_id_summary=cases.groupby('dataset').agg(n_rows=('query_id','size'),n_queries=('query_id','nunique'),exact_id_match_rate=('exact_id_outcome',lambda x:(x=='match').mean())).reset_index()
display(exact_id_summary)
display(main.groupby(['dataset','screen_path_limit','budget']).status.value_counts())
'''),
M('''## 2 · CI 95% theo query và hình fidelity–sparsity–coverage

Lấy lại query ID, không lấy lại từng dòng seed. CI fidelity chỉ tính trên query có random match và kèm `n_matched_queries`. Yield/coverage dùng toàn bộ query. Không so trực tiếp thang fuse VG và PSG.
'''),
C('''rng=np.random.default_rng(20261121); ci=[]
for key,part in main.groupby(['dataset','screen_path_limit','budget']):
    by_query=part.groupby('query_id').agg(coverage=('n_paths',lambda x:(x>0).mean()),advantage=('matched_advantage','mean'))
    out={'dataset':key[0],'screen_path_limit':key[1],'budget':key[2],'n_queries':len(by_query),'n_matched_queries':by_query.advantage.notna().sum()}
    for field in ('coverage','advantage'):
        values=by_query[field].dropna().to_numpy(); out[field+'_mean']=float(values.mean()) if len(values) else np.nan
        if len(values):
            draws=rng.choice(values,size=(5000,len(values)),replace=True).mean(axis=1)
            out[field+'_ci_low'],out[field+'_ci_high']=map(float,np.quantile(draws,[.025,.975]))
        else: out[field+'_ci_low']=out[field+'_ci_high']=np.nan
    ci.append(out)
ci=pd.DataFrame(ci); display(ci)
coverage.to_csv(analysis/'coverage.csv',index=False); ci.to_csv(analysis/'cluster_bootstrap_ci.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for (dataset,limit),sub in coverage.groupby(['dataset','screen_path_limit']):
    sub=sub.sort_values('budget'); label=f'{dataset.upper()} screen={limit}'
    for ax,y in zip(axes,['mean_fidelity_advantage','mean_sparsity','selected_path_rate']): ax.plot(sub.budget,sub[y],marker='o',label=label)
for ax,title in zip(axes,['Matched final drop advantage','Sparsity / target contains edges','Object evidence yield / all cases']):
    ax.set_title(title); ax.set_xlabel('Unique-edge budget'); ax.grid(alpha=.3)
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(analysis/'fidelity_sparsity_coverage.png',dpi=160)
display(fig); plt.close(fig)
'''),
M('''## 3 · Tối đa 10 case study và Excel chấm ảnh

Chọn 5 GT-ID match/5 GT-ID mismatch theo seed 0, gồm cả case không có cạnh đối tượng giải thích. Nhúng GT và top-1 vào workbook khi tải được. Ở tab Case review, chấm riêng `retrieved_relevance` cho ảnh top-1 khác ID GT, ghi `visual_reason` và `reviewer`. Phần bằng chứng đối tượng có thể chấm trên một workbook; cạnh `contains` không tự xác minh hành động/vị trí của cả caption.
'''),
C('''limit=int(cases.screen_path_limit.max()); pool=cases[(cases.model_seed==0)&(cases.screen_path_limit==limit)]
def choose_outcome(outcome):
    part=pool[pool.exact_id_outcome==outcome].sort_values(['dataset','query_id'])
    preferred=pd.concat([x.head(2) for _,x in part.groupby('dataset')]) if len(part) else part
    remaining=part.drop(preferred.index).sort_values(['query_id','dataset'])
    return pd.concat([preferred,remaining.head(max(0,5-len(preferred)))]).head(5)
chosen=pd.concat([choose_outcome('match'),choose_outcome('mismatch')])
print('Case studies:',len(chosen),'| GT-ID match:',(chosen.exact_id_outcome=='match').sum(),'| GT-ID mismatch:',(chosen.exact_id_outcome=='mismatch').sum())
if len(chosen)<10: print('Insufficient GT-ID match/mismatch cases; reporting the available cases only')
image_dir=analysis/'case_images'; image_dir.mkdir(exist_ok=True)
ALLOW_UNVERIFIED_TLS=True  # public dataset images only; provenance is saved below
path_index={}
for p in paths:
    if p.get('model_seed')==0 and p.get('screen_path_limit')==limit: path_index.setdefault((p['dataset'],str(p['query_id'])),[]).append(p)
verified=pd.read_csv(DUONG/'registry/queries_direct_verified.csv',keep_default_na=False,dtype={'query_id':str})
gt_labels={(r['dataset'],r['query_id']):r['gt_relevance'] for r in verified.to_dict('records')}
review_cases=[]; review_steps=[]; download_records=[]
for row in chosen.to_dict('records'):
    row['gt_relevance']=gt_labels.get((row['dataset'],str(row['query_id'])),'')
    row['retrieved_relevance']='supported' if row['exact_id_outcome']=='match' and row['gt_relevance']=='supported' else ''
    row['retrieval_outcome']='GT ID match' if row['exact_id_outcome']=='match' else 'GT ID mismatch'
    ps=[p for p in path_index.get((row['dataset'],str(row['query_id'])),[]) if p.get('level')=='object' and 2 in p.get('selected_at_budgets',[])]
    row['explanation_level']='object_evidence' if ps else 'no_object_explanation'
    row['path_or_edge_text']='; '.join(' → '.join(p['node_ids']) for p in ps) or 'no path selected'
    for role,url_key,file_key in [('gt','gt_source_url','gt_image_file'),('retrieved','retrieved_source_url','retrieved_image_file')]:
        path=image_dir/f"{row['dataset']}_{row['query_id']}_{role}.jpg"
        if path.is_file():
            row[file_key]=str(path)
            download_records.append({'dataset':row['dataset'],'query_id':row['query_id'],'role':role,'status':'cached'})
            continue
        try:
            info=fetch_review_image(row[url_key],path,allow_unverified_tls=ALLOW_UNVERIFIED_TLS)
            row[file_key]=str(path)
            download_records.append({'dataset':row['dataset'],'query_id':row['query_id'],'role':role,**info})
        except Exception as exc:
            row[file_key]=''; download_records.append({'dataset':row['dataset'],'query_id':row['query_id'],'role':role,'source_url':row[url_key],'status':'failed','error':str(exc)})
            print('Image unavailable:',row['query_id'],role,str(exc)[:180])
    review_cases.append(row)
    if not ps: review_steps.append({'dataset':row['dataset'],'query_id':row['query_id'],'target_image_id':row['retrieved_image_id'],'path_id':'no_path'})
    for p in ps:
        for i,e in enumerate(p['edge_ids']):
            review_steps.append({'dataset':row['dataset'],'query_id':row['query_id'],'target_image_id':row['retrieved_image_id'],'path_id':p['path_id'],'step_index':i+1,'edge_id':e,'source_node':p['node_ids'][i],'target_node':p['node_ids'][i+1],'predicate_hint':p['predicates'][i],'edge_type':p['edge_types'][i],'edge_provenance':p['sources'][i],'model_score_drop':p['final_score_drop']})
workbook=analysis/'validity_case_review_10.xlsx'
if workbook.exists():
    existing=pd.read_excel(workbook,sheet_name='Case review',keep_default_na=False,dtype={'query_id':str})
    expected_ids={(r['dataset'],str(r['query_id'])) for r in review_cases}
    actual_ids={(r['dataset'],str(r['query_id'])) for r in existing.to_dict('records')}
    assert len(existing)==len(review_cases) and actual_ids==expected_ids, 'Existing workbook has a different case set'
    print('Keeping existing review workbook and its annotations:',workbook)
else:
    create_review_workbook(review_cases,review_steps,workbook,analysis)
pd.DataFrame(download_records).to_csv(analysis/'image_download_manifest.csv',index=False)
print('Downloaded/available images:',sum(r['status']!='failed' for r in download_records),'/',len(download_records))
annotations=pd.DataFrame(review_steps)
for col in ['reviewer','object_supported','visual_evidence','step_decision','support_source','evidence_reason']:
    annotations[col]=''
annotations_path=analysis/'validity_annotations.csv'
if not annotations_path.exists(): annotations.to_csv(annotations_path,index=False)
pd.DataFrame(review_cases).to_csv(analysis/'case_studies_10.csv',index=False)
print('Review workbook:',workbook,'step rows:',len(review_steps))
'''),
M('''## 4 · Chấm mức phù hợp của ảnh top-1

Một ảnh top-1 **khác ID GT** vẫn có thể hỗ trợ toàn query. Trong workbook `validity_case_review_10.xlsx`, tab **Case review**, các ô `retrieved_relevance` (`supported`/`contradicted`/`uncertain`), `visual_reason` và `reviewer` có thể lưu nhãn từng dòng. Người dùng đã xác nhận bằng mắt rằng cả năm mismatch của lượt `path-20261121-184857` hỗ trợ query; cell dưới dùng xác nhận này cho ô còn trống, nhưng nhãn đã điền trong workbook được ưu tiên. Chạy lại cell sau khi lưu Excel nếu có sửa nhãn. Tỷ lệ thu được chỉ mô tả **10 case được chọn cân bằng theo ID**, không phải semantic@1 của toàn bộ 34 query.
'''),
C('''RELEVANCE_WORKBOOK=analysis/'validity_case_review_10.xlsx'  # dùng workbook đã điền; có thể đặt đường dẫn khác
reviewed=pd.read_excel(RELEVANCE_WORKBOOK,sheet_name='Case review',keep_default_na=False,dtype={'query_id':str})
reviewed['query_id']=reviewed.query_id.astype(str)
expected=chosen[['dataset','query_id','exact_id_outcome']].copy()
expected['query_id']=expected.query_id.astype(str)
fields=['dataset','query_id','retrieved_relevance','visual_reason','reviewer']
assert set(fields) <= set(reviewed.columns)
semantic=expected.merge(reviewed[fields],on=['dataset','query_id'],how='left',validate='one_to_one')
assert len(semantic)==len(chosen) and semantic.retrieved_relevance.notna().all()
semantic['relevance_source']='case_review_workbook'
match_without_label=(semantic.exact_id_outcome=='match') & semantic.retrieved_relevance.eq('')
semantic.loc[match_without_label,'retrieved_relevance']='supported'  # same verified GT image
semantic.loc[match_without_label,'relevance_source']='verified_gt_same_image'
USER_CONFIRMED_MISMATCH_RUN='path-20261121-184857'
if RESULTS_DIR.name==USER_CONFIRMED_MISMATCH_RUN:
    confirmed=(semantic.exact_id_outcome=='mismatch') & semantic.retrieved_relevance.eq('')
    semantic.loc[confirmed,'retrieved_relevance']='supported'
    semantic.loc[confirmed,'relevance_source']='user_visual_confirmation'
    supported_mismatch=(semantic.exact_id_outcome=='mismatch') & semantic.retrieved_relevance.eq('supported')
    auto_provenance=supported_mismatch & (semantic.visual_reason.eq('') | semantic.reviewer.eq(''))
    semantic.loc[auto_provenance,'relevance_source']='user_visual_confirmation'
    semantic.loc[supported_mismatch & semantic.visual_reason.eq(''),'visual_reason']='User visually confirmed retrieved image supports the full query in the case workbook.'
    semantic.loc[supported_mismatch & semantic.reviewer.eq(''),'reviewer']='user_visual_confirmation'
missing=semantic[(semantic.exact_id_outcome=='mismatch') & (
    ~semantic.retrieved_relevance.isin(['supported','contradicted','uncertain']) |
    semantic.visual_reason.eq('') | semantic.reviewer.eq(''))]
if len(missing):
    print('Cần chấm ảnh top-1 khác ID GT cho các query:')
    display(missing[['dataset','query_id','retrieved_relevance','visual_reason','reviewer']])
else:
    assert semantic.loc[semantic.exact_id_outcome=='match','retrieved_relevance'].eq('supported').all()
    semantic.to_csv(analysis/'semantic_case_review.csv',index=False)
    counts=semantic.retrieved_relevance.value_counts().reindex(['supported','contradicted','uncertain'],fill_value=0)
    text=f"""# Top-1 relevance on stratified case studies

GT-ID match: {(semantic.exact_id_outcome=='match').sum()}/{len(semantic)}.
GT-ID mismatch: {(semantic.exact_id_outcome=='mismatch').sum()}/{len(semantic)}.
Top-1 supported: {counts['supported']}/{len(semantic)}; contradicted: {counts['contradicted']}; uncertain: {counts['uncertain']}.

These ten cases were selected to balance exact-ID match/mismatch. Blank mismatch labels for run path-20261121-184857 were filled from the user's visual confirmation; explicit workbook labels take precedence. This is not a semantic accuracy estimate for all 34 queries. Exact-ID ranking metrics remain valid for the dataset benchmark.
"""
    (analysis/'semantic_case_review.md').write_text(text,encoding='utf8')
    display(semantic)
    print(text)
'''),
M('''## 5 · Bằng chứng đối tượng (một workbook, tùy chọn)

Nếu muốn báo object/visual support của cạnh `contains`, điền `object_supported`, `visual_evidence`, `step_decision`, `evidence_reason` và `reviewer` trong tab **Step review**. Cell dưới chỉ tính trên nhãn đã điền; không tính đồng thuận giữa người chấm và không yêu cầu bản hòa giải.
'''),
C('''STEP_REVIEW_WORKBOOK=analysis/'validity_case_review_10.xlsx'
step_review=pd.read_excel(STEP_REVIEW_WORKBOOK,sheet_name='Step review',keep_default_na=False,dtype={'query_id':str})
real=step_review[step_review.edge_id.astype(str).str.strip().ne('')].copy()
labels=['supported','contradicted','uncertain']
judgements=['object_supported','visual_evidence','step_decision']
ready=(len(real)>0 and all(real[field].isin(labels).all() for field in judgements)
       and real.reviewer.astype(str).str.strip().ne('').all()
       and real.evidence_reason.astype(str).str.strip().ne('').all())
if not ready:
    print('Object/visual support chưa có đủ nhãn trong Step review; báo cáo fidelity và semantic relevance vẫn hợp lệ.')
else:
    real['object_step_supported']=real[['object_supported','step_decision']].eq('supported').all(axis=1)
    real['visual_step_supported']=real[['visual_evidence','step_decision']].eq('supported').all(axis=1)
    real['step_known']=real.step_decision.ne('uncertain')
    paths_review=real.groupby(['dataset','query_id','path_id']).agg(
        object_supported=('object_step_supported','all'),
        visual_supported=('visual_step_supported','all'),
        known=('step_known','all')).reset_index()
    real.to_csv(analysis/'object_step_review.csv',index=False)
    paths_review.to_csv(analysis/'object_path_review.csv',index=False)
    summary=f"""# Object evidence on selected case studies

Reviewed steps: {len(real)}; object step support: {real.object_step_supported.mean():.3f}; visual step support: {real.visual_step_supported.mean():.3f}.
Reviewed paths: {len(paths_review)}; object path support: {paths_review.object_supported.mean():.3f}; visual path support: {paths_review.visual_supported.mean():.3f}.
These labels concern object evidence only; a contains edge does not verify the full caption.
"""
    (analysis/'object_evidence_review.md').write_text(summary,encoding='utf8')
    print(summary)
'''),
])
