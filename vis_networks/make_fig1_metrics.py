import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "." 
OUT  = os.path.join("..", "paper", "figures", "fig1_metrics.png")
mpl.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"font.family":"serif",
    "font.serif":["Times New Roman","DejaVu Serif"],"font.size":9,
    "axes.labelsize":9,"xtick.labelsize":7.5,"ytick.labelsize":8,
    "legend.fontsize":8,"axes.linewidth":0.8,"grid.linewidth":0.5,"grid.alpha":0.35})

CONFIGS = [
 ("VisNet\nGlobal","VisNetSiteEvaluationResults/site_specific_validation_epoch0.csv","global"),
 ("RMEP\nGlobal","RMEPSiteEvaluationResults/site_specific_validation_epoch0.csv","global"),
 ("VisNet\nScratch","VisNetPersiteAndPredictionsResults/persite_results.csv","scratch"),
 ("RMEP\nScratch","RMEPPerSiteAndPredictionsResults/persite_results.csv","scratch"),
 ("VisNet\nFine-tune","VisNetFineTuneAndPredictionsResults/persite_results.csv","ft"),
 ("RMEP\nFine-tune","RMEPFineTuneAndPredictionsResults/persite_results.csv","ft"),
 ("VisNet-Sim\nFT","VisNetSimilarityPersiteFineTuneResults/persite_results.csv","sim"),
]
COL = {"global":"#0072B2","scratch":"#E69F00","ft":"#009E73","sim":"#CC79A7"}

def overall(path):
    df = pd.read_csv(os.path.join(BASE, path))
    r = df[df["site_id"]=="OVERALL"].iloc[0]
    return (float(r["R2"]), float(r["accuracy"])*100,
            float(r["accuracy_within_2"])*100, float(r["MAE_visibility"]))

labels=[c[0] for c in CONFIGS]; colors=[COL[c[2]] for c in CONFIGS]
R2=[];ACC=[];ACC2=[];MAE=[]
for _,p,_ in CONFIGS:
    r,a,a2,m = overall(p); R2.append(r);ACC.append(a);ACC2.append(a2);MAE.append(m)

panels=[("$R^2$",R2,True),("Acc (%)",ACC,True),("$\\pm$2 Acc (%)",ACC2,True),("MAE (mi)",MAE,False)]
x=np.arange(len(CONFIGS))
fig,axes=plt.subplots(2,2,figsize=(6.3,4.5),constrained_layout=True)
for ax,(name,vals,hb) in zip(axes.flat,panels):
    bars=ax.bar(x,vals,width=0.62,color=colors,edgecolor="white",linewidth=0.5,zorder=3)
    best=int(np.argmax(vals) if hb else np.argmin(vals))
    bars[best].set_edgecolor("#111111");bars[best].set_linewidth(1.4)
    ax.axhline(0,color="black",linewidth=0.5)
    ax.set_ylabel(name);ax.set_xticks(x);ax.set_xticklabels(labels)
    ax.yaxis.grid(True,zorder=0);ax.set_axisbelow(True)
fig.legend(handles=[Patch(facecolor=COL["global"],label="Global"),
                    Patch(facecolor=COL["scratch"],label="Per-site scratch"),
                    Patch(facecolor=COL["ft"],label="Per-site fine-tune"),
                    Patch(facecolor=COL["sim"],label="VisNet-Similarity FT")],
           loc="lower center",ncol=4,bbox_to_anchor=(0.5,-0.04),frameon=True)
os.makedirs(os.path.dirname(OUT),exist_ok=True)
fig.savefig(OUT,dpi=300,bbox_inches="tight")
print("saved",OUT)
for (l,_,_),r,a,a2,m in zip(CONFIGS,R2,ACC,ACC2,MAE):
    print(f"{l.replace(chr(10),' '):20s} R2={r:+.3f} Acc={a:5.1f} pm2={a2:5.1f} MAE={m:.3f}")