"""Build reproducible, output-free XAI path notebooks."""
import json
from pathlib import Path

ROOT = Path(__file__).parent

def M(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def C(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
def save(name,cells):
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (ROOT/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')

save('01_xai_audit_and_query_review.ipynb',[
M('''# A–B · Audit baseline và kiểm chứng query

**Mục tiêu:** khóa kết quả 24 caption cũ và chuẩn bị 40 truy vấn mới trước khi xem path XAI. Chạy CPU local hoặc Colab. `source_group` là nhãn từ bundle, chưa phải nhãn direct/relational đã kiểm chứng. Ảnh âm là ứng viên cùng đối tượng, chưa phải ground truth. Không ghi đè snapshot cũ.'''),
C('''from pathlib import Path
import hashlib, sys
import pandas as pd
ROOT = Path('/content/drive/MyDrive') if Path('/content/drive/MyDrive/graphml-project3').is_dir() else next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'graphml-project3').is_dir())
DUONG = ROOT/'graphml-project3/duong'
sys.path.insert(0,str(DUONG))
from xai.review_workbook import baseline_review_pack, create_review_workbook
from xai.query_registry import validated_registry_rows, import_reviewed_queries
SNAP = DUONG/'runs/qa_vg_psg_xai/xai-20261121-130842'
REG = DUONG/'review/queries_relational_candidates.csv'
assert SNAP.is_dir() and REG.is_file()
print('Frozen registry SHA256:',hashlib.sha256(REG.read_bytes()).hexdigest())
'''),
M('''## 1 · Audit snapshot cũ

Kiểm tra 216 dòng, 12 query/bộ × 3 seed × 3 mức k. Đây là leave-one-edge-out; path validity và evidence được báo riêng. Mẫu số sparsity cũ chỉ là vùng sàng lọc.'''),
C('''old=pd.read_csv(SNAP/'analysis/xai_results_corrected.csv')
assert len(old)==216 and old.groupby('dataset').caption_id.nunique().to_dict()=={'vg':12,'psg':12}
display(old[old.requested_k==3].groupby('dataset').agg(n=('caption_id','size'),query_count=('caption_id','nunique'),screen_coverage=('screen_coverage','mean'),final_drop=('final_score_drop','mean'),random_drop=('random_final_score_drop_mean','mean'),evidence_known=('evidence_known_rate','mean')))
'''),
M('''## 2 · Cổng query/ảnh

Bộ ứng viên có 10 `direct` nguồn và 10 `indirect` nguồn mỗi dataset. Hai người xem ảnh dương và ảnh âm khó, điền claim subject/predicate/object, `verified_group`, relevance, đối tượng chung/quan hệ khác và căn cứ ảnh. Duyệt CSV chỉ sau khi đối chiếu Excel; `validated_registry_rows` chặn mọi query chưa qua cổng. Embedding CLIP và concepts được lấy theo caption ID của chính bundle.'''),
C('''registry=pd.read_csv(REG,keep_default_na=False)
assert len(registry)==40
display(registry.groupby(['dataset','source_group']).size())
display(registry[['dataset','query_id','query','gt_image_id','hard_negative_candidate_id','gt_source_url','hard_negative_source_url','review_status']].head(10))
try:
    print('Approved:',len(validated_registry_rows(REG)))
except ValueError as exc:
    print('Visual verification pending:',exc)
'''),
M('''## 3 · Excel kiểm chứng

Workbook baseline nhúng 10 cặp ảnh GT/top-1 đã tải và các cạnh mô hình chọn. Workbook query mới chứa URL và nhúng ảnh khi tải được. Hai reviewer chấm độc lập; đưa bản đã hòa giải vào `REVIEWED_WORKBOOK`, nhập thành CSV mới và giữ registry nguồn bất biến.'''),
C('''baseline_review_pack(SNAP,DUONG/'review/baseline_case_review_10.xlsx')
import urllib.request
DOWNLOAD_QUERY_IMAGES=Path('/content').is_dir()
image_dir=DUONG/'review/query_images'; image_dir.mkdir(exist_ok=True)
cases=[]
for r in registry.to_dict('records'):
    case={'dataset':r['dataset'],'query_id':r['query_id'],'query':r['query'],'source_group':r['source_group'],'gt_image_id':r['gt_image_id'],'retrieved_image_id':r['hard_negative_candidate_id'],'retrieval_outcome':'pending visual review','explanation_level':'pre-XAI candidate','path_or_edge_text':'annotate claim before XAI','gt_source_url':r['gt_source_url'],'retrieved_source_url':r['hard_negative_source_url'],'review_status':'pending'}
    if DOWNLOAD_QUERY_IMAGES:
        for role,url_key,file_key in [('gt','gt_source_url','gt_image_file'),('negative','retrieved_source_url','retrieved_image_file')]:
            path=image_dir/f"{r['dataset']}_{r['query_id']}_{role}.jpg"
            try:
                if not path.is_file(): urllib.request.urlretrieve(case[url_key],path)
                case[file_key]=str(path) if path.stat().st_size>0 else ''
            except Exception as exc: print('Image unavailable:',r['query_id'],role,str(exc)[:80])
    cases.append(case)
create_review_workbook(cases,[],DUONG/'review/query_candidates_40.xlsx',DUONG)
print(DUONG/'review/baseline_case_review_10.xlsx',DUONG/'review/query_candidates_40.xlsx')
REVIEWED_WORKBOOK=None  # Path('.../adjudicated_query_review.xlsx')
if REVIEWED_WORKBOOK:
    verified=import_reviewed_queries(REG,REVIEWED_WORKBOOK,DUONG/'review/queries_relational_verified.csv')
    print('Verified query count:',len(validated_registry_rows(verified)))
''')])

save('02_xai_path_perturbation_colab.ipynb',[
M('''# C–D · Path search và perturbation trên QA scorer thật

**Câu hỏi:** xóa đường có làm đổi điểm/hạng của ảnh đích trên cùng top-100 không? Chạy Colab GPU. `label_h` chỉ nhận `edge_index`, do đó kết quả chỉ chứng minh **relational connectivity**, không chứng minh mô hình hiểu tên predicate. Đường 2–3 cạnh đi có hướng từ concept query qua nhãn ảnh rồi `contains`; đường một cạnh `contains` là object evidence. Mọi case, kể cả không có đường, ở trong mẫu số coverage.'''),
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

Mặc định chạy lại đúng 24 caption baseline. Chỉ chọn `verified40` sau khi ảnh dương/âm và nhóm query đã duyệt. Hai mức screen là phân tích độ nhạy được đặt trước. Budget là **số cạnh duy nhất**, random path khớp loại cạnh/độ dài/budget; thiếu đối chứng ghi N/A. Hash nguồn huấn luyện, bundle và checkpoint được xác minh; hash XAI mới khác baseline là có chủ đích.'''),
C('''QUERY_MODE='baseline24'  # 'verified40' only after visual review
MODEL_SEEDS=(0,1,2); PATH_SCREEN_LIMITS=(40,120); EDGE_BUDGETS=(2,3,5)
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
if QUERY_MODE=='baseline24':
    selected=json.loads((RUNS/'xai-20261121-130842/selected_cases.json').read_text())
else:
    selected=validated_registry_rows(DUONG/'review/queries_relational_verified.csv')
assert len(selected)==(24 if QUERY_MODE=='baseline24' else 40)
for dataset,spec in MODEL_SPECS.items():
    seed0_dir=RUNS/spec.experiment/Path(records[dataset,0]['run_dir']).name
    ranks0=json.loads((seed0_dir/'ranks_test_full.json').read_text())['ranks']
    subset=[r for r in selected if r['dataset']==dataset]
    success=sum(int(ranks0[int(r['candidate_row'])])==1 for r in subset)
    failure=sum(int(ranks0[int(r['candidate_row'])])>1 for r in subset)
    assert success>=5 and failure>=5,(dataset,success,failure)
    print(dataset,'seed-0 success/failure:',success,failure)
OUT=RUNS/('path-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'protocol.json').write_text(json.dumps({'query_mode':QUERY_MODE,'manifest':str(MANIFEST),'seeds':MODEL_SEEDS,'path_screen_limits':PATH_SCREEN_LIMITS,'edge_budgets':EDGE_BUDGETS,'random_repeats':RANDOM_REPEATS,'random_seed':RANDOM_SEED,'selected_query_ids':{d:[int(r['caption_id'] if QUERY_MODE=='baseline24' else r['query_id']) for r in selected if r['dataset']==d] for d in MODEL_SPECS}},indent=2))
print('Output:',OUT)
'''),
M('''## 2 · Score gốc, vùng đủ điều kiện và candidate path

`E_eligible` gồm mọi `contains` của ảnh đích và các cạnh nhãn–nhãn trong incoming receptive field thật (số tầng GNN). Screen chỉ dùng topology và giới hạn path đã khóa; cạnh ngoài screen vẫn hiện diện trong scorer. Với từng seed/query/screen, kiểm tra QA score và hạng GT so với checkpoint.'''),
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
            group=str(cands['group'][j]) if QUERY_MODE=='baseline24' else item['verified_group']
            for screen_limit in PATH_SCREEN_LIMITS:
                case=HuyQAPathCaseModel(model,graph,qd,bundle,protocol,cands,ev,j,rec['alpha'],DEVICE,max_paths=screen_limit,image_tensor=image_tensor)
                reference=qa_scores(model,qd,image_tensor,np.asarray(protocol.cap[cap_id:cap_id+1]),np.asarray([cap_id]),case.rows_np[None],DEVICE)[0]
                np.testing.assert_allclose(case.score_removed()['qa'],reference,rtol=1e-4,atol=1e-5)
                ids=tuple(map(str,case.ids)); before=case.scorer.score(query,ids,case.intervention_graph)
                if gt_id in ids: assert rank_of(before,gt_id,ids)==int(saved_ranks[j])
                else: assert int(saved_ranks[j])>len(ids)
                rr,pp=evaluate_case(case,query,gt_id,dataset=dataset,model_seed=seed,query_id=str(cap_id),group=group,budgets=EDGE_BUDGETS,random_repeats=RANDOM_REPEATS,random_seed=RANDOM_SEED+seed*100000+j*10+screen_limit,hard_negative_image_id=(item['hard_negative_candidate_id'] if QUERY_MODE=='verified40' else None))
                for r in rr: r['screen_path_limit']=screen_limit
                for p in pp: p.update(dataset=dataset,model_seed=seed,screen_path_limit=screen_limit,query=query,ground_truth_image_id=gt_id)
                result_rows.extend(rr); path_rows.extend(pp)
                def source_url(im): return im.get('url') or ('https://images.cocodataset.org/'+im['coco_file'] if im.get('coco_file') else im.get('coco_url',''))
                case_rows.append({'dataset':dataset,'query_id':cap_id,'query':query,'source_group':group,'model_seed':seed,'screen_path_limit':screen_limit,'gt_image_id':gt_id,'retrieved_image_id':case.top_id,'retrieval_outcome':'success' if case.top_id==int(gt_id) else 'failure','gt_rank':int(saved_ranks[j]),'target_rank':1,'gt_source_url':source_url(protocol.images[int(gt_id)]),'retrieved_source_url':source_url(protocol.images[case.top_id])})
        print('Completed:',dataset,seed,len(chosen),'queries')
'''),
M('''## 3 · Lưu artifact bất biến

`paths.jsonl`: mọi path đã sàng lọc, contribution cả path, lý do chọn và `no_path`. `xai_results.csv`: mỗi can thiệp trên cùng candidate pool, score graph/QA/fuse, hạng ảnh đích và GT riêng, mẫu số eligible/screened theo loại. `case_manifest.csv` giữ URL ảnh cho workbook. Không ghi đè baseline.'''),
C('''assert result_rows and path_rows and case_rows
with (OUT/'xai_results.csv').open('w',newline='',encoding='utf8') as f:
    keys=list(dict.fromkeys(k for row in result_rows for k in row)); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(result_rows)
with (OUT/'case_manifest.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(case_rows[0])); w.writeheader(); w.writerows(case_rows)
with (OUT/'paths.jsonl').open('w',encoding='utf8') as f:
    for row in path_rows: f.write(json.dumps(row,ensure_ascii=False)+'\\n')
(OUT/'complete.json').write_text(json.dumps({'n_result_rows':len(result_rows),'n_path_rows':len(path_rows),'n_cases':len(case_rows)},indent=2))
print('Saved:',OUT,len(result_rows),len(path_rows),len(case_rows))
''')])

save('03_xai_path_analysis_and_validity.ipynb',[
M('''# E · Phân tích fidelity, coverage và validity độc lập

**Câu hỏi:** đường có là lý do hợp lý cho truy vấn–ảnh không? Notebook CPU đọc một lượt `path-*` hoàn tất. Bootstrap theo **query**, giữ seed trong cụm; coverage tính trên mọi case đăng ký, kể cả `no_relational_explanation`. Fidelity chỉ so với random path khớp cấu trúc khi có đối chứng độc lập. Logic và căn cứ ảnh chờ hai người chấm; score drop không tự cho nhãn validity.'''),
C('''from pathlib import Path
import json,sys,urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path('/content/drive/MyDrive') if Path('/content/drive/MyDrive/graphml-project3').is_dir() else next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'graphml-project3').is_dir())
DUONG=ROOT/'graphml-project3/duong'; RUNS=DUONG/'runs/qa_vg_psg_xai'
sys.path.insert(0,str(DUONG))
from xai.review_workbook import create_review_workbook
RESULTS_DIR=None  # set an explicit completed path-* directory for the report
if RESULTS_DIR is None:
    complete=sorted(p.parent for p in RUNS.glob('path-*/complete.json'))
    assert complete,'Run notebook 02 first.'
    RESULTS_DIR=complete[-1]
RESULTS_DIR=Path(RESULTS_DIR)
results=pd.read_csv(RESULTS_DIR/'xai_results.csv')
cases=pd.read_csv(RESULTS_DIR/'case_manifest.csv')
paths=[json.loads(line) for line in (RESULTS_DIR/'paths.jsonl').open(encoding='utf8')]
meta=json.loads((RESULTS_DIR/'complete.json').read_text())
assert len(results)==meta['n_result_rows'] and len(paths)==meta['n_path_rows']
analysis=RESULTS_DIR/'analysis'; analysis.mkdir(exist_ok=True)
print(RESULTS_DIR,len(results),len(cases),len(paths))
'''),
M('''## 1 · Denominator, path yield và random matching

Mỗi `dataset × screen limit × budget` giữ cùng số case. Sparsity = 1 − số cạnh explanation duy nhất / `E_eligible`; `E_screened / E_eligible` được báo riêng. Nếu `E_eligible_relation = 0`, relation fraction là N/A. Random thiếu cấu trúc khớp được báo thiếu đối chứng.'''),
C('''main=results[results.intervention=='relational_path'].copy()
assert not main.duplicated(['dataset','query_id','model_seed','screen_path_limit','budget']).any()
assert (main.n_screened<=main.n_eligible).all() and (main.n_removed_unique<=main.budget).all()
main['matched_advantage']=main.final_score_drop-main.random_final_drop_mean
coverage=main.groupby(['dataset','screen_path_limit','budget']).agg(n_cases=('query_id','size'),n_queries=('query_id','nunique'),positive_path_rate=('has_positive_relational_path','mean'),selected_path_rate=('n_paths',lambda x:(x>0).mean()),no_explanation_rate=('status',lambda x:(x!='ok').mean()),structural_no_path_rate=('status',lambda x:(x=='no_relational_path').mean()),mean_eligible=('n_eligible','mean'),mean_screen_coverage=('screen_coverage','mean'),mean_sparsity=('sparsity','mean'),matched_cases=('random_n',lambda x:(x>0).sum()),unique_random_sets=('random_unique_n','sum'),mean_fidelity_advantage=('matched_advantage','mean')).reset_index()
display(coverage)
display(main.groupby(['dataset','screen_path_limit','budget','group_source']).status.value_counts())
'''),
M('''## 2 · CI 95% theo query và hình fidelity–sparsity–coverage

Lấy lại query ID, không lấy lại từng dòng seed. CI fidelity chỉ tính trên query có random match và đi kèm `n_matched_queries`. Yield/coverage dùng toàn bộ query. Không so trực tiếp thang fuse VG và PSG.'''),
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
for ax,title in zip(axes,['Matched final drop advantage','Sparsity / eligible edges','Relational yield / all cases']):
    ax.set_title(title); ax.set_xlabel('Unique-edge budget'); ax.grid(alpha=.3)
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(analysis/'fidelity_sparsity_coverage.png',dpi=160)
display(fig); plt.close(fig)
'''),
C('''report=f"""# Path XAI report — {RESULTS_DIR.name}

Protocol: {json.loads((RESULTS_DIR/'protocol.json').read_text())['query_mode']}; fixed top-100 candidates, frozen QA checkpoints and alpha.
The current scorer consumes edge_index, not predicate identity; relational paths describe connectivity only.

## Coverage and fidelity by dataset, screen and unique-edge budget

```
{coverage.to_string(index=False)}
```

Matched fidelity is conditional on an independent random path with the same edge-type/length signature and unique-edge budget. The `matched_cases` column gives its denominator. `mean_sparsity` is conditional on an explanation existing; no-path cases remain in yield/coverage denominators. VG and PSG use different graph/model configurations, so their fuse-score magnitudes are not directly compared.

## Query-cluster bootstrap 95% CI

```
{ci.to_string(index=False)}
```

Logical and visual validity remain pending independent review. Structural validity confirms graph continuity only. `no_relational_explanation` is a result, not an exclusion.
"""
(analysis/'report.md').write_text(report,encoding='utf8')
print('Report:',analysis/'report.md')
'''),
M('''## 3 · 10 case study và Excel chấm ảnh

Chọn 5 success/5 failure theo seed 0 và screen lớn nhất, gồm cả case không có đường. Tải GT và top-1 từ URL nguồn, nhúng vào workbook khi tải được; giữ URL nếu mạng lỗi. Các cột đánh giá và rubric để trống. Hai người chấm riêng từng bước: đối tượng, predicate/chiều, thông tin mới, căn cứ ảnh, toàn path. Mỗi bản là một reviewer độc lập.'''),
C('''limit=int(cases.screen_path_limit.max()); pool=cases[(cases.model_seed==0)&(cases.screen_path_limit==limit)]
def choose_outcome(outcome):
    part=pool[pool.retrieval_outcome==outcome].sort_values(['dataset','query_id'])
    preferred=pd.concat([x.head(2) for _,x in part.groupby('dataset')]) if len(part) else part
    remaining=part.drop(preferred.index).sort_values(['query_id','dataset'])
    return pd.concat([preferred,remaining.head(max(0,5-len(preferred)))]).head(5)
chosen=pd.concat([choose_outcome('success'),choose_outcome('failure')])
assert len(chosen)==10,'Need five success and five failure.'
image_dir=analysis/'case_images'; image_dir.mkdir(exist_ok=True)
path_index={}
for p in paths:
    if p.get('model_seed')==0 and p.get('screen_path_limit')==limit: path_index.setdefault((p['dataset'],str(p['query_id'])),[]).append(p)
review_cases=[]; review_steps=[]
for row in chosen.to_dict('records'):
    ps=[p for p in path_index.get((row['dataset'],str(row['query_id'])),[]) if 3 in p.get('selected_at_budgets',[])]
    row['explanation_level']='relational_connectivity' if ps else 'no_relational_explanation'
    row['path_or_edge_text']='; '.join(' → '.join(p['node_ids'])+' ['+','.join(p['predicates'])+']' for p in ps) or 'no path selected'
    for role,url_key,file_key in [('gt','gt_source_url','gt_image_file'),('retrieved','retrieved_source_url','retrieved_image_file')]:
        path=image_dir/f"{row['dataset']}_{row['query_id']}_{role}.jpg"
        try:
            if not path.is_file(): urllib.request.urlretrieve(row[url_key],path)
            row[file_key]=str(path) if path.stat().st_size>0 else ''
        except Exception as exc:
            row[file_key]=''; print('Image unavailable:',row['query_id'],str(exc)[:80])
    review_cases.append(row)
    if not ps: review_steps.append({'dataset':row['dataset'],'query_id':row['query_id'],'target_image_id':row['retrieved_image_id'],'path_id':'no_path'})
    for p in ps:
        for i,e in enumerate(p['edge_ids']):
            review_steps.append({'dataset':row['dataset'],'query_id':row['query_id'],'target_image_id':row['retrieved_image_id'],'path_id':p['path_id'],'step_index':i+1,'edge_id':e,'source_node':p['node_ids'][i],'target_node':p['node_ids'][i+1],'predicate_hint':p['predicates'][i],'edge_type':p['edge_types'][i],'edge_provenance':p['sources'][i],'model_score_drop':p['final_score_drop']})
workbook=create_review_workbook(review_cases,review_steps,analysis/'validity_case_review_10.xlsx',analysis)
annotations=pd.DataFrame(review_steps)
for col in ['reviewer','object_supported','predicate_direction_valid','new_information','visual_evidence','step_decision','support_source','evidence_reason','whole_path_decision','adjudicated_decision']:
    annotations[col]=''
annotations.to_csv(analysis/'validity_annotations.csv',index=False)
pd.DataFrame(review_cases).to_csv(analysis/'case_studies_10.csv',index=False)
print('Review workbook:',workbook,'step rows:',len(review_steps))
'''),
M('''## 4 · Hai reviewer và hòa giải

Sau khi chấm độc lập, chỉ định hai workbook. Tính tỷ lệ đồng thuận và Cohen κ trên `step_decision` gồm cả `uncertain`. Đưa bất đồng vào bảng hòa giải; chỉ báo logical/visual supported sau khi `adjudicated_decision` đầy đủ. Không dùng graph schema hoặc provenance làm nhãn ảnh.'''),
C('''REVIEWER_A=None  # Path('.../reviewer_a.xlsx')
REVIEWER_B=None  # Path('.../reviewer_b.xlsx')
if REVIEWER_A and REVIEWER_B:
    keys=['dataset','query_id','target_image_id','path_id','step_index','edge_id']
    a=pd.read_excel(REVIEWER_A,sheet_name='Step review',keep_default_na=False)
    b=pd.read_excel(REVIEWER_B,sheet_name='Step review',keep_default_na=False)
    ab=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one')
    assert len(ab)==len(a)==len(b)
    assert (ab.step_decision_a!='').all() and (ab.step_decision_b!='').all()
    labels=['supported','contradicted','uncertain']
    assert ab.step_decision_a.isin(labels).all() and ab.step_decision_b.isin(labels).all()
    po=(ab.step_decision_a==ab.step_decision_b).mean()
    pe=sum((ab.step_decision_a==label).mean()*(ab.step_decision_b==label).mean() for label in labels)
    kappa=(po-pe)/(1-pe) if pe<1 else float('nan')
    print('Agreement:',po,'Cohen kappa:',kappa,'n:',len(ab))
    ab.to_csv(analysis/'validity_annotations_two_reviewers.csv',index=False)
else:
    print('Validity pending independent human annotation; no logical/visual support rate reported.')
ADJUDICATED_WORKBOOK=None  # Path('.../adjudicated_validity.xlsx')
if ADJUDICATED_WORKBOOK:
    final=pd.read_excel(ADJUDICATED_WORKBOOK,sheet_name='Step review',keep_default_na=False)
    assert len(final)==len(review_steps)
    assert final.adjudicated_decision.isin(['supported','contradicted','uncertain']).all()
    final.to_csv(analysis/'validity_annotations.csv',index=False)
    real=final[final.path_id!='no_path']
    for field in ['object_supported','predicate_direction_valid','new_information','visual_evidence','whole_path_decision']:
        assert real[field].isin(['supported','contradicted','uncertain']).all(),field
    real=real.copy()
    real['logical_supported']=real[['predicate_direction_valid','new_information','adjudicated_decision']].eq('supported').all(axis=1)
    real['visual_supported']=real[['object_supported','visual_evidence','adjudicated_decision']].eq('supported').all(axis=1)
    real['step_known']=real.adjudicated_decision.ne('uncertain')
    path_scores=real.groupby(['dataset','query_id','path_id']).agg(
        logical_steps=('logical_supported','all'),visual_steps=('visual_supported','all'),
        known_steps=('step_known','all'),whole_supported=('whole_path_decision',lambda x:x.eq('supported').all()),
        whole_known=('whole_path_decision',lambda x:x.ne('uncertain').all())).reset_index()
    path_scores['supported']=path_scores.logical_steps & path_scores.visual_steps & path_scores.whole_supported
    path_scores['known']=path_scores.known_steps & path_scores.whole_known
    path_scores.to_csv(analysis/'validity_path_scores.csv',index=False)
    print('Known step rate:',real.step_known.mean(),'Logical step support:',real.logical_supported.mean(),
          'Visual step support:',real.visual_supported.mean(),'Known path rate:',path_scores.known.mean(),
          'Supported path rate:',path_scores.supported.mean(),
          'No-path cases:',(final.path_id=='no_path').sum())
    with (analysis/'report.md').open('a',encoding='utf8') as f:
        f.write(f"""

## Independently adjudicated validity (10 case studies)

Step known: {real.step_known.mean():.3f}; logical step support: {real.logical_supported.mean():.3f}; visual step support: {real.visual_supported.mean():.3f}.
Path known: {path_scores.known.mean():.3f}; whole-path support: {path_scores.supported.mean():.3f}; no-path cases: {(final.path_id=='no_path').sum()}.
These rates use every selected step/path in the 10-case workbook; uncertain counts in the denominator.
""")
''')])

print('Built three notebooks.')
