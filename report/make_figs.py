import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import subprocess
from matplotlib import font_manager
# Font giống thân bài: Times New Roman nếu có, không thì TeX Gyre Termes (đi kèm TeX Live — nạp thẳng file vì không nằm trong
# thư mục font hệ thống). Nhúng TrueType (Type 42) thay vì Type 3 cho PDF hợp chuẩn IEEE.
for f in subprocess.run(["fc-list", ":family=TeX Gyre Termes", "file"], capture_output=True, text=True).stdout.split():
    font_manager.fontManager.addfont(f.rstrip(":"))
plt.rcParams.update({"pdf.fonttype": 42})
from matplotlib.ticker import FuncFormatter
vn = lambda x, d: f"{x:.{d}f}".replace(".", ",")          # dấu phẩy thập phân như thân bài (64,76)
VN_AXIS = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
_serif = [f for f in ("Times New Roman", "TeX Gyre Termes") if f in {x.name for x in font_manager.fontManager.ttflist}]
plt.rcParams.update({"font.family":["serif","DejaVu Serif"],"font.serif":_serif + ["DejaVu Serif"],"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
O="figs/"
C={"clip":"#9aa0a6","ad":"#5b8def","g":"#1a8f5b","sh":"#d9534f"}
# Fig1: MRR toàn pool, VG và PSG
fig,ax=plt.subplots(1,2,figsize=(7,2.5),sharey=False)
vg=[("CLIP",60.40,0,"clip"),("+Adapter",63.14,0.02,"ad"),("+Đồ thị\n(contains+CN)\nSAGE",64.89,0.15,"g"),("+Đồ thị\nxáo cạnh",62.67,0.06,"sh")]
ps=[("CLIP",59.59,0,"clip"),("+Adapter",62.67,0.03,"ad"),("+Đồ thị\n(sg_rel)",64.17,0.09,"g"),("+Đồ thị\nxáo cạnh",62.17,0.13,"sh")]
for a,d,t,lo in ((ax[0],vg,"VG∩COCO (test, 2138 ảnh)",58),(ax[1],ps,"PSG∩COCO (test, 2186 ảnh)",58)):
    for i,(n,v,s,c) in enumerate(d):
        a.bar(i,v,yerr=s or None,color=C[c],width=.65,capsize=2)
        a.text(i,v+.15,vn(v,2),ha="center",fontsize=8)
    a.set_xticks(range(4));a.set_xticklabels([x[0] for x in d],fontsize=7.5);a.set_ylim(lo,66.5);a.set_title(t,fontsize=9);a.yaxis.set_major_formatter(VN_AXIS)
ax[0].set_ylabel("MRR (%)")
plt.tight_layout();plt.savefig(O+"fig_main.pdf");plt.close()
# Fig2: direct vs indirect, toàn pool seed 0 (vs_qagraph_paired.md)
rows=[("CLIP",60.79,60.36,"clip"),("+Khớp tên\n(không quan hệ)",68.29,59.97,"clip"),("+AdaMF-MAT",66.39,59.89,"clip"),("+Adapter\n(bỏ cạnh)",63.80,63.06,"ad"),("+Đồ thị\nxáo cạnh",64.03,62.50,"sh"),("+Đồ thị\nSAGE",71.10,63.78,"g")]
fig,a=plt.subplots(figsize=(7,2.6));w=.38
for i,(n,d,ind,c) in enumerate(rows):
    a.bar(i-w/2,d,w,color=C[c]);a.bar(i+w/2,ind,w,color=C[c],alpha=.45,hatch="//",edgecolor="white")
    a.text(i-w/2,d+.3,vn(d,1),ha="center",fontsize=7);a.text(i+w/2,ind+.3,vn(ind,1),ha="center",fontsize=7)
a.set_xticks(range(len(rows)));a.set_xticklabels([r[0] for r in rows],fontsize=7.5);a.set_ylim(55,74);a.set_ylabel("MRR (%)");a.yaxis.set_major_formatter(VN_AXIS)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#555",label="direct"),Patch(color="#555",alpha=.45,hatch="//",label="indirect")],fontsize=8,frameon=False,loc="upper left")
plt.tight_layout();plt.savefig(O+"fig_group.pdf");plt.close()
# Fig3: ΔMRR so với adapter, CI95 (VG seed0 vs AdaMF; dùng bảng Huy)
