import os

import torch
from torch.utils.data import DataLoader

ARCHS = ("visnet", "rmep", "similarity")
DIM = (280, 280)
N_CLASSES = 10


def default_splits(arch):
    # The split lists tracked in the repo. Similarity uses its own copy, which
    # drops the two images whose camera views have no reference.
    return os.path.join("splits", "similarity") if arch == "similarity" else "splits"


def read_split(splits_path, names):
    samples = []
    for name in names:
        with open(os.path.join(splits_path, name)) as f:
            samples += [line.rstrip() for line in f if line.strip()]
    return samples


def site_of(path):
    return os.path.basename(path).split("_")[0]


def event_of(path):
    # A capture event is one site at one harvest time. Harvest folders are shared
    # across sites, so the site has to be part of the key.
    return site_of(path) + "|" + os.path.basename(os.path.dirname(path))


def build_dataset(arch, dset_path, samples, ref_dir=None):
    """The dataset and evaluation transform each architecture was trained with."""
    from dsets.webcams import cls_10full, cls_10full_dual
    from models.visnet.transform import get_transformer
    if arch == "visnet":
        return cls_10full.DsetClass(dset_path, list(samples), DIM, 3, get_transformer(DIM), crop=True)
    if arch == "rmep":
        return cls_10full.DsetClass(dset_path, list(samples), DIM, 3, None, crop=True)
    if ref_dir is None:
        raise SystemExit("ERROR: --arch similarity needs --ref_dir")
    return cls_10full_dual.DsetClass(dset_path, list(samples), DIM, 3, get_transformer(DIM),
                                     crop=True, ref_dir=ref_dir)


def build_model(arch, input_size, checkpoint, backbone=None, untrained=False, seed=42):
    """Load a trained checkpoint, or (untrained=True) keep seeded random weights
    but take the checkpoint's input-normalization buffers, so the trained and
    untrained networks see identical inputs and differ only in their weights."""
    from models.visnet import visnet, visnet_similarity
    from models.rmep import rmep
    if not os.path.isfile(checkpoint):
        raise SystemExit(f"ERROR: checkpoint not found: {checkpoint}")
    if arch == "similarity":
        if untrained:
            raise SystemExit("ERROR: --untrained does not apply to similarity (its backbone is pretrained by design)")
        if backbone is None:
            raise SystemExit("ERROR: --arch similarity needs --backbone (the VisNet checkpoint)")
        model = visnet_similarity.Model(N_CLASSES, input_size, pretrained_path=backbone)
    else:
        torch.manual_seed(seed)
        model = (visnet if arch == "visnet" else rmep).Model(N_CLASSES, input_size)

    sd = torch.load(checkpoint, weights_only=True, map_location="cpu")
    try:
        if untrained:
            with torch.no_grad():
                model(torch.zeros(1, *input_size))        # materialise the lazy layers, still random
            report = model.load_state_dict({k: sd[k] for k in ("mean", "std")}, strict=False)
        else:
            report = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:                             # tensor shapes from another architecture
        raise SystemExit(f"ERROR: {checkpoint} does not fit --arch {arch} (wrong --arch?)\n{e}")
    if untrained:
        print(f"  UNTRAINED control: random weights (seed {seed}), mean/std buffers from the checkpoint")
    else:
        if report.missing_keys:
            raise SystemExit(f"ERROR: {checkpoint} is missing {len(report.missing_keys)} tensors "
                             f"for --arch {arch}: {report.missing_keys[:6]} ... wrong --arch?")
        extra = [k for k in report.unexpected_keys if k not in ("example_input", "example_output")]
        if extra:
            raise SystemExit(f"ERROR: {checkpoint} has tensors --arch {arch} does not use: "
                             f"{extra[:6]} ... wrong --arch?")
        print(f"  checkpoint loaded: {len(sd)} tensors, none missing")
    return model


def embedding_module(arch, model):
    """The layer whose INPUT is the embedding the classifier sees."""
    if arch == "similarity":
        return model.cls_head[0]                          # cat([p_curr, p_diff]), 256-dim
    last = None
    for m in model.modules():                             # VisNet: 4096 -> 10; RMEP: 256 -> 10
        if isinstance(m, torch.nn.Linear):                # LazyLinear is a Linear subclass
            last = m
    return last


def embed(arch, model, dataset, batch_size=16):
    """Embeddings, visibility (class value, mi) and paths for every sample, in order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0 if os.name == "nt" else 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    grabbed, labels, paths = [], [], []
    hook = embedding_module(arch, model).register_forward_pre_hook(
        lambda _m, inp: grabbed.append(inp[0].detach().float().cpu()))
    with torch.no_grad():
        for data, lab, pth in loader:
            model(data.to(device))
            labels.append(torch.argmax(lab, 1))
            paths += list(pth)
    hook.remove()
    X = torch.cat(grabbed).numpy()
    vis = torch.cat(labels).numpy().astype(float) + 1.0   # class index -> 1..10 mi
    return X, vis, paths


def pooled_train_share(paths):
    """How many of these images the pooled models were trained on."""
    trained = set(read_split("splits", ["train.csv"]))
    return sum(p in trained for p in paths)
