$VIS  = "runs\VisNetGlobal\best_epoch19.pt"
$RMEP = "runs\RMEPGlobal\best_epoch37.pt"
$SIM  = "runs\VisNetSimilarity\best_epoch4.pt"
$REFS = "D:\Research - Lasya\NewWebcams\references_by_view"
$OUT  = "lda_results"

function Step($name, [string[]]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & python3 @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name" -ForegroundColor Red; exit 1 }
}

# 1. Optional check, NOT for the thesis: repeats the earlier 22 Sep analysis (same 397
#    images, same image-level splits). The "image-level" line should read about 64.8%
#    (VisNet) and 46.8% (RMEP). Delete the two Step lines below to skip it.
Step "Check vs 22 Sep - VisNet" @("make_lda_sites.py", "--arch", "visnet", "--checkpoint", $VIS, "--images", "persite_test", "--label", "VisNet", "--out", "$OUT\check_sites_visnet")
Step "Check vs 22 Sep - RMEP"   @("make_lda_sites.py", "--arch", "rmep", "--checkpoint", $RMEP, "--images", "persite_test", "--label", "RMEP", "--out", "$OUT\check_sites_rmep")

Step "Sites - VisNet"             @("make_lda_sites.py", "--arch", "visnet", "--checkpoint", $VIS, "--label", "VisNet", "--out", "$OUT\sites_visnet")
Step "Sites - VisNet (untrained)" @("make_lda_sites.py", "--arch", "visnet", "--checkpoint", $VIS, "--untrained", "--label", "VisNet (untrained)", "--out", "$OUT\sites_visnet_untrained")
Step "Sites - RMEP"               @("make_lda_sites.py", "--arch", "rmep", "--checkpoint", $RMEP, "--label", "RMEP", "--out", "$OUT\sites_rmep")
Step "Sites - RMEP (untrained)"   @("make_lda_sites.py", "--arch", "rmep", "--checkpoint", $RMEP, "--untrained", "--label", "RMEP (untrained)", "--out", "$OUT\sites_rmep_untrained")

Step "One site - RMEP"   @("make_lda_onesite.py", "--arch", "rmep", "--checkpoint", $RMEP, "--out", "$OUT\onesite_rmep")
Step "One site - VisNet" @("make_lda_onesite.py", "--arch", "visnet", "--checkpoint", $VIS, "--out", "$OUT\onesite_visnet")

# 4. Optional: VisNet-Similarity (last, so a problem here cannot block the runs above).
#    The sites run also probes each half of its embedding: p_curr (current image) and
#    p_diff (difference from the camera's clear-day reference).
Step "Sites - Similarity"    @("make_lda_sites.py", "--arch", "similarity", "--checkpoint", $SIM, "--backbone", $VIS, "--ref_dir", $REFS, "--label", "VisNet-Similarity", "--out", "$OUT\sites_similarity")
Step "One site - Similarity" @("make_lda_onesite.py", "--arch", "similarity", "--checkpoint", $SIM, "--backbone", $VIS, "--ref_dir", $REFS, "--out", "$OUT\onesite_similarity")

Write-Host "`nAll LDA runs finished. Results are in $OUT\" -ForegroundColor Green
