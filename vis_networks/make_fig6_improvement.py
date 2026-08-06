import os, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

BASE="."; OUT=os.path.join("..","paper","figures","fig6_improvement.png")
GLOBAL="VisNetSiteEvaluationResults/site_specific_validation_epoch0.csv"
FT="VisNetFineTuneAndPredictionsResults/persite_results.csv"
mpl.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"font.family":"serif",
    "font.serif":["Times New Roman","DejaVu Serif"],"font.size":8.5,
    "axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":7.5})

g=pd.read_csv(os.path.join(BASE,GLOBAL)); g=g[g["site_id"]!="OVERALL"]
g=g[["site_id","MAE_visibility","RMSE_visibility"]].rename(columns={"MAE_visibility":"g_MAE","RMSE_visibility":"g_RMSE"})
f=pd.read_csv(os.path.join(BASE,FT)); f=f[f["site_id"]!="OVERALL"]
f=f[["site_id","n_train","MAE_visibility","RMSE_visibility"]].rename(columns={"MAE_visibility":"f_MAE","RMSE_visibility":"f_RMSE"})
d=pd.merge(f,g,on="site_id",how="inner")
d["dMAE"]=d["f_MAE"]-d["g_MAE"]; d["dRMSE"]=d["f_RMSE"]-d["g_RMSE"]
print(f"merged sites={len(d)}  n_train range {int(d.n_train.min())}-{int(d.n_train.max())}")

fig,ax=plt.subplots(2,2,figsize=(6.3,5.0),constrained_layout=True)
a=ax[0,0]; a.scatter(d.n_train,d.dMAE,c=d.dMAE,cmap="RdYlGn_r",s=26,edgecolor="white",linewidth=0.3)
a.axhline(0,ls="--",color="k",lw=0.8); a.set_xlabel("Training samples per site"); a.set_ylabel(r"$\Delta$MAE (FT $-$ global, mi)")
a=ax[0,1]; a.scatter(d.n_train,d.dRMSE,c=d.dRMSE,cmap="RdYlGn_r",s=26,edgecolor="white",linewidth=0.3)
a.axhline(0,ls="--",color="k",lw=0.8); a.set_xlabel("Training samples per site"); a.set_ylabel(r"$\Delta$RMSE (FT $-$ global, mi)")
a=ax[1,0]; sc=a.scatter(d.g_MAE,d.f_MAE,c=d.n_train,cmap="viridis",s=28,edgecolor="white",linewidth=0.3)
lo=min(d.g_MAE.min(),d.f_MAE.min()); hi=max(d.g_MAE.max(),d.f_MAE.max())
a.plot([lo,hi],[lo,hi],"k--",lw=0.8); a.set_xlabel("Global MAE (mi)"); a.set_ylabel("Fine-tune MAE (mi)")
fig.colorbar(sc,ax=a,label="n_train")
a=ax[1,1]; edges=[0,50,100,200,400,np.inf]; labs=["<50","50–100","100–200","200–400",">400"]
d["bucket"]=pd.cut(d.n_train,bins=edges,labels=labs,right=False)
groups=[d[d.bucket==l]["dRMSE"].values for l in labs]; keep=[i for i,gp in enumerate(groups) if len(gp)]
bp=a.boxplot([groups[i] for i in keep],patch_artist=True,medianprops=dict(color="black",linewidth=1.5))
for patch in bp["boxes"]: patch.set_facecolor("#98df8a"); patch.set_alpha(0.7)
a.axhline(0,ls="--",color="k",lw=0.8); a.set_xticklabels([labs[i] for i in keep])
a.set_xlabel("Training samples per site"); a.set_ylabel(r"$\Delta$RMSE (FT $-$ global, mi)")
os.makedirs(os.path.dirname(OUT),exist_ok=True)
fig.savefig(OUT,dpi=300,bbox_inches="tight"); print("saved",OUT)