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
# Hình MRR toàn pool, VG và PSG. Vẽ đúng bề rộng một cột IEEE (3,5 in) để cỡ chữ in ra bằng cỡ khai báo (≥ 7 pt).
fig,ax=plt.subplots(1,2,figsize=(3.5,2.3),sharey=True)
vg=[("CLIP",60.40,0,"clip"),("Adapter",63.14,0.02,"ad"),("Đồ thị",64.89,0.15,"g"),("Xáo\ncạnh",62.67,0.06,"sh")]
ps=[("CLIP",59.59,0,"clip"),("Adapter",62.67,0.03,"ad"),("Đồ thị",64.41,0.12,"g"),("Xáo\ncạnh",62.17,0.13,"sh")]
for a,d,t in ((ax[0],vg,"VG∩COCO"),(ax[1],ps,"PSG∩COCO")):
    for i,(n,v,s,c) in enumerate(d):
        a.bar(i,v,yerr=s or None,color=C[c],width=.7,capsize=1.5)
        a.text(i,v+.15,vn(v,2),ha="center",fontsize=7)
    a.set_xticks(range(4));a.set_xticklabels([x[0] for x in d],fontsize=7);a.set_ylim(58,66.5);a.set_title(t,fontsize=8)
    a.tick_params(axis="y",labelsize=7);a.yaxis.set_major_formatter(VN_AXIS)
ax[0].set_ylabel("MRR (%)",fontsize=7.5)
plt.tight_layout(pad=0.3,w_pad=0.6);plt.savefig(O+"fig_main.pdf");plt.close()
# Hình direct vs indirect, VG toàn pool seed 0 (vs_qagraph_paired.md), cùng bề rộng một cột
rows=[("CLIP",60.79,60.36,"clip"),("Khớp\ntên",68.29,59.97,"clip"),("AdaMF\n-MAT",66.39,59.89,"clip"),("Adapter",63.80,63.06,"ad"),("Xáo\ncạnh",64.03,62.50,"sh"),("Đồ thị",71.10,63.78,"g")]
fig,a=plt.subplots(figsize=(3.5,2.4));w=.42
for i,(n,d,ind,c) in enumerate(rows):
    a.bar(i-w/2,d,w,color=C[c]);a.bar(i+w/2,ind,w,color=C[c],alpha=.45,hatch="//",edgecolor="white")
    a.text(i-w/2,d+.3,vn(d,1),ha="center",fontsize=7,rotation=90,va="bottom");a.text(i+w/2,ind+.3,vn(ind,1),ha="center",fontsize=7,rotation=90,va="bottom")
a.set_xticks(range(len(rows)));a.set_xticklabels([r[0] for r in rows],fontsize=7);a.set_ylim(55,76);a.set_ylabel("MRR (%)",fontsize=7.5)
a.tick_params(axis="y",labelsize=7);a.yaxis.set_major_formatter(VN_AXIS)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#555",label="direct"),Patch(color="#555",alpha=.45,hatch="//",label="indirect")],fontsize=7,frameon=False,loc="upper left",ncol=2)
plt.tight_layout(pad=0.3);plt.savefig(O+"fig_group.pdf");plt.close()
# Fig3: ΔMRR so với adapter, CI95 (VG seed0 vs AdaMF; dùng bảng Huy)
