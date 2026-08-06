import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

BASE="."; OUT=os.path.join("..","paper","figures","fig2_r2dist.png")
mpl.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"font.family":"serif",
    "font.serif":["Times New Roman","DejaVu Serif"],"font.size":9,
    "axes.labelsize":9,"xtick.labelsize":8.5,"ytick.labelsize":8,"legend.fontsize":7.5})

CONFIGS=[("VisNet FT","VisNetFineTuneAndPredictionsResults/persite_results.csv"),
         ("RMEP FT","RMEPFineTuneAndPredictionsResults/persite_results.csv"),
         ("VisNet-Sim FT","VisNetSimilarityPersiteFineTuneResults/persite_results.csv")]
TIERS=[(0.89,np.inf,"Excellent ($R^2>0.89$)","#2ca02c"),
       (0.5,0.89,"Good (0.5–0.89)","#98df8a"),
       (0.0,0.5,"Fair (0–0.5)","#f4d03f"),
       (-np.inf,0.0,"Poor ($R^2<0$)","#d62728")]

def r2(path):
    df=pd.read_csv(os.path.join(BASE,path)); df=df[df["site_id"]!="OVERALL"]
    return df["R2"].dropna().values

data=[r2(p) for _,p in CONFIGS]; x=np.arange(len(CONFIGS))
fig,ax=plt.subplots(figsize=(5.0,4.3),constrained_layout=True); bottoms=np.zeros(len(CONFIGS))
for lo,hi,lab,c in TIERS:
    cnt=np.array([int(np.sum(v>lo) if hi==np.inf else np.sum((v>lo)&(v<=hi))) for v in data])
    ax.bar(x,cnt,bottom=bottoms,color=c,label=lab,edgecolor="white",linewidth=0.6,zorder=3)
    for xi,(b,n) in enumerate(zip(bottoms,cnt)):
        if n>0: ax.text(xi,b+n/2,str(n),ha="center",va="center",fontsize=8,fontweight="bold")
    bottoms+=cnt
ax.set_ylabel("Number of sites"); ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={len(v)})" for (l,_),v in zip(CONFIGS,data)])
ax.set_ylim(0,max(bottoms)*1.06)
h,l=ax.get_legend_handles_labels()
ax.legend(h[::-1],l[::-1],loc="lower center",bbox_to_anchor=(0.5,1.02),ncol=2,
          frameon=True,columnspacing=1.2,handlelength=1.4)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
fig.savefig(OUT,dpi=300,bbox_inches="tight"); print("saved",OUT)
for (l,_),v in zip(CONFIGS,data): print(f"{l:16s} sites={len(v)}")