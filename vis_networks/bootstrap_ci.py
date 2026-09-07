import csv, numpy as np
np.random.seed(42)
def load(f):
    rows=list(csv.DictReader(open(f"{f}/predictions.csv")))
    return (np.array([float(r["true_vis"]) for r in rows]),
            np.array([float(r["pred_vis"]) for r in rows]))
C={"VisNet Scratch":"VisNetPersiteAndPredictionsResults","RMEP Scratch":"RMEPPerSiteAndPredictionsResults",
   "VisNet FT":"VisNetFineTuneAndPredictionsResults","RMEP FT":"RMEPFineTuneAndPredictionsResults",
   "Sim FT":"VisNetSimilarityPersiteFineTuneResults"}
D={k:load(v) for k,v in C.items()}; yt=D["RMEP FT"][0]; N=len(yt); B=10000
idx=np.random.randint(0,N,size=(B,N))
def ci(a): return np.percentile(a,2.5), np.percentile(a,97.5)
boot={}
for k,(t,p) in D.items():
    tb,pb=t[idx],p[idx]
    r2=1-np.sum((tb-pb)**2,1)/np.sum((tb-tb.mean(1,keepdims=True))**2,1)
    mae=np.mean(np.abs(tb-pb),1); rmse=np.sqrt(np.mean((tb-pb)**2,1)); boot[k]=(r2,mae,rmse)
    R2=1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)
    print(f"{k:14s} R2={R2:.3f} {ci(r2)}  MAE={np.mean(np.abs(t-p)):.3f} {ci(mae)}")
for k in ["VisNet FT","Sim FT","RMEP Scratch"]:
    dmae=boot[k][1]-boot["RMEP FT"][1]
    print(f"RMEP vs {k}: dMAE={dmae.mean():+.3f} {ci(dmae)} p={2*min(np.mean(dmae<=0),np.mean(dmae>=0)):.4f}")