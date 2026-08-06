import torch
import torcheval.metrics.functional as temf
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def confusionMat(outputs, i_targets, num_classes, class_names, normalize='true'):
    confusion_tensor = temf.multiclass_confusion_matrix(outputs, i_targets, num_classes, normalize=normalize)
    confusion_df = pd.DataFrame(confusion_tensor, index=class_names, columns=class_names)

    sn.set_theme(style='white', font_scale=0.6)
    plot = sn.heatmap(confusion_df, annot=True, vmin=0.0, vmax=1.0, annot_kws={'size': 5}, square=True)
    plot.set_xlabel('Predicted Label', fontsize=10)
    plot.set_ylabel('True Label', fontsize=10)
    plot.set_title('Confusion Matrix', fontsize=12)
    plot.tick_params(labelrotation=0)

    plot = plt.gcf()
    plot.set_dpi(300)
    plt.close()

    return plot

def reliabilityFigure(outputs, i_targets, num_bins, graph_type='line'):
    outputs = outputs.to(torch.float)

    sm_outputs = torch.softmax(outputs, 1)

    predictions = torch.argmax(sm_outputs, 1)
    confs = torch.max(sm_outputs, 1)[0]
    correct = (predictions == i_targets).to(torch.float32)

    bin_indices = (confs*num_bins).to(torch.int64).clamp(max=num_bins-1)
    bin_counts = torch.bincount(bin_indices, minlength=num_bins)
    bin_confs = torch.tensor([-1.]*num_bins).index_reduce(0, bin_indices, confs, 'mean', include_self=False)
    bin_accs = torch.tensor([-1.]*num_bins).index_reduce(0, bin_indices, correct, 'mean', include_self=False)
    expected_accs = torch.Tensor([(i+.5)/num_bins for i in range(num_bins)])    

    sn.set_theme(style='white', font_scale=1)
    fig, (plot1, plot2) = plt.subplots(2, 1, figsize=(7.2, 6.4))
    fig.supxlabel('Confidence')
    for p in (plot1,plot2):
        p.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        p.set_xlim(0,1)
        p.tick_params(bottom=True, left=True, top=False, right=False)
        p.grid(True, color="#888888", linestyle=':')

    if graph_type == 'line':
        #Line graph version
        plot1.set_title('Reliability Diagram')
        plot1.set_ylabel('Accuracy')
        plot1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plot1.plot([0,1], [0,1], linestyle='--', linewidth=2, color="#000000", label='Target')
        plot1.plot(bin_confs[bin_confs>-1], bin_accs[bin_accs>-1], linewidth=2, color="#3030ff", marker='o', label='Actual')
        plot1.legend()

    elif graph_type == 'bar':
        # Bar graph version
        gap_bars = torch.stack((bucket_accs, expected_accs)).transpose(0,1).sort(1)[0]
        for gb in gap_bars: gb[:] = 0 if gb[0] < 0 else gb[:]
        for gb in gap_bars: gb[1] = gb[1]-gb[0]
        gap_bars = gap_bars.transpose(0,1)

        bucket_accs = torch.clamp_min(bucket_accs, 0)
        bar_width = 1/num_bins

        plot1.set_title('Reliability Diagram')
        plot1.set_ylabel('Accuracy')
        plot1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plot1.bar(expected_accs, bucket_accs, width=bar_width, color="#3030ff", edgecolor="#000000", alpha=1.0, label='Actual')
        plot1.bar(expected_accs, gap_bars[1], bottom=gap_bars[0], width=bar_width, color="#ff3030", edgecolor="#ff3030", hatch='/', alpha=0.3, label='Gap')
        plot1.bar(expected_accs, gap_bars[1], bottom=gap_bars[0], width=bar_width, color="none", edgecolor="#ff3030", hatch='/', alpha=0.5)
        plot1.plot([0, 1], [0,1], linestyle='--', linewidth=2, color="#000000", label='Target')
        plot1.legend()

    #Line graph version
    plot1.set_title('Reliability Diagram')
    plot1.set_ylabel('Accuracy')
    plot1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    bin_counts_rel = bin_counts/bin_counts.sum()
    bin_edges = [i/num_bins for i in range(num_bins+1)]
    plot2.set_title('Confidence Histogram')
    plot2.set_ylabel('Proportion')
    plot2.stairs(bin_counts_rel, bin_edges, fill=True, color="#3030ff", alpha=0.3)
    plot2.stairs(bin_counts_rel, bin_edges, fill=False, linewidth=2, color="#3030ff", alpha=1.0)

    fig.gca().set_aspect('auto', adjustable='box')
    fig.tight_layout()
    fig.set_dpi(100)
    plt.close()

    return fig