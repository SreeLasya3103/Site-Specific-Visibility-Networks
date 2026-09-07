import csv

outputs_path = r"D:\Research - Lasya\Visibility-Networks\vis_networks\runs\Apr07_16-38-24_ENG-GM05S89C\outputs.csv"

class_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

correct = 0
within_2 = 0
total = 0
ss_res = 0.0
sum_true = 0.0
abs_errors = []

with open(outputs_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
true_values = []

for row in rows:
    pred_idx = int(row['predicted'])
    true_idx = int(row['truth'])
    pred_vis = class_values[pred_idx]
    true_vis = class_values[true_idx]
    
    if pred_idx == true_idx:
        correct += 1
    if abs(pred_idx - true_idx) <= 2:
        within_2 += 1
    
    abs_errors.append(abs(pred_vis - true_vis))
    ss_res += (true_vis - pred_vis) ** 2
    true_values.append(true_vis)

mean_true = sum(true_values) / len(true_values)
ss_tot = sum((t - mean_true) ** 2 for t in true_values)

acc = correct / total
acc2 = within_2 / total
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
mae = sum(abs_errors) / len(abs_errors)
rmse = (ss_res / total) ** 0.5

print(f"N:          {total}")
print(f"Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
print(f"±2 Acc:     {acc2:.4f} ({acc2*100:.2f}%)")
print(f"R²:         {r2:.4f}")
print(f"MAE:        {mae:.4f} miles")
print(f"RMSE:       {rmse:.4f} miles")
