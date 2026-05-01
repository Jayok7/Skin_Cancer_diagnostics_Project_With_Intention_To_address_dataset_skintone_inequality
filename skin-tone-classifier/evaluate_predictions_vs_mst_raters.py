#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MST5_NAMES = [
    "Very Dark (MST 9-10)", "Dark (MST 7-8)", "Medium (MST 5-6)",
    "Light (MST 3-4)", "Very Light (MST 1-2)",
]
CLASSES_5WAY = MST5_NAMES

def map_mst10_to_mst5_name(mst_val):
    if pd.isna(mst_val):
        return None
    try:
        mst_val = int(mst_val)
    except:
        return None
        
    if mst_val in [1, 2]: return "Very Light (MST 1-2)"
    if mst_val in [3, 4]: return "Light (MST 3-4)"
    if mst_val in [5, 6]: return "Medium (MST 5-6)"
    if mst_val in [7, 8]: return "Dark (MST 7-8)"
    if mst_val in [9, 10]: return "Very Dark (MST 9-10)"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, default="outputs/FairFace-Model-3.2/skin_tone_predictions.csv")
    parser.add_argument("--colorimeter", type=str, default="datasets/MSKCC-MST/Analysis of colorimeter vs. image-extracted ITA in non-lesional sites..csv")
    parser.add_argument("--inter-rater", type=str, default="datasets/MSKCC-MST/Analysis of inter-rater agreement for in-person Pantone and MST on both lesional and non-lesional sites.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/FairFace-Model-3.2/eval_vs_raters")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading predictions...")
    preds_df = pd.read_csv(args.preds)
    preds_df['isic_id'] = preds_df['file'].apply(lambda x: x.replace(".jpg", "").replace(".png", ""))
    
    print("Loading MSKCC MST Datasets...")
    colorimeter_df = pd.read_csv(args.colorimeter)
    rater_df = pd.read_csv(args.inter_rater)
    
    # Merge colorimeter to get tag_id for each isic_id
    colorimeter_map = colorimeter_df[['isic_id', 'tag_id']].drop_duplicates()
    
    # Merge inter-rater to get mst_r1 and mst_r2 for each tag_id
    rater_map = rater_df[['tag_id', 'mst_r1', 'mst_r2']].drop_duplicates()
    
    # Create final mapping dataframe
    gt_df = pd.merge(colorimeter_map, rater_map, on='tag_id', how='inner')
    
    # Merge with predictions
    eval_df = pd.merge(preds_df, gt_df, on='isic_id', how='inner')
    
    # Map MST 1-10 to MST 5-Way names
    eval_df['r1_mst5'] = eval_df['mst_r1'].apply(map_mst10_to_mst5_name)
    eval_df['r2_mst5'] = eval_df['mst_r2'].apply(map_mst10_to_mst5_name)
    
    print(f"\nFound {len(eval_df)} images with both predictions and Rater Ground Truth.")
    if len(eval_df) == 0:
        print("No matches found. Exiting.")
        return

    # 1. Error Analytics & Classification Reports
    for rater, col in [("Rater 1", "r1_mst5"), ("Rater 2", "r2_mst5")]:
        valid_df = eval_df.dropna(subset=[col, 'mst_name'])
        if len(valid_df) == 0: continue
        
        def within_one(true_val, pred_val):
            try:
                return abs(CLASSES_5WAY.index(true_val) - CLASSES_5WAY.index(pred_val)) <= 1
            except ValueError:
                return False
                
        acc = accuracy_score(valid_df[col], valid_df['mst_name'])
        relaxed_acc = valid_df.apply(lambda x: within_one(x[col], x['mst_name']), axis=1).mean()
        
        print(f"\n{'-'*50}\nEVALUATION AGAINST MSKCC {rater.upper()}\n{'-'*50}")
        print(f"Overall Accuracy: {acc:.2%}")
        print(f"Relaxed Accuracy (±1 Class): {relaxed_acc:.2%}")
        print("\nClassification Report:")
        print(classification_report(valid_df[col], valid_df['mst_name'], labels=CLASSES_5WAY, zero_division=0))
        
        # Breakdown by Method
        print(f"\nAccuracy Breakdown by Method ({rater}):")
        for method in valid_df['method'].unique():
            method_df = valid_df[valid_df['method'] == method]
            if len(method_df) > 0:
                m_acc = accuracy_score(method_df[col], method_df['mst_name'])
                print(f"  {method}: {m_acc:.2%} ({len(method_df)} samples)")
        
        # Breakdown by Confidence
        print(f"\nAccuracy Breakdown by Confidence ({rater}):")
        for conf in valid_df['confidence'].unique():
            conf_df = valid_df[valid_df['confidence'] == conf]
            if len(conf_df) > 0:
                c_acc = accuracy_score(conf_df[col], conf_df['mst_name'])
                print(f"  {conf} confidence: {c_acc:.2%} ({len(conf_df)} samples)")

        # Save Confusion Matrix
        cm = confusion_matrix(valid_df[col], valid_df['mst_name'], labels=CLASSES_5WAY)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES_5WAY, yticklabels=CLASSES_5WAY)
        plt.title(f'FairFace CNN vs MSKCC {rater} (MST-5)')
        plt.ylabel(f'True Label ({rater})')
        plt.xlabel('Predicted Label (FairFace)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"cm_{rater.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    # 2. Confidence Analytics
    print(f"\n{'-'*50}\nCONFIDENCE ANALYTICS\n{'-'*50}")
    print(f"Mean Confidence Value: {eval_df['confidence_val'].mean():.3f}")
    
    # We will compute if prediction is correct by Rater 1 OR Rater 2 (agreement with at least one)
    eval_df['is_correct_either'] = (eval_df['mst_name'] == eval_df['r1_mst5']) | (eval_df['mst_name'] == eval_df['r2_mst5'])
    conf_correct = eval_df[eval_df['is_correct_either']]['confidence_val'].mean()
    conf_incorrect = eval_df[~eval_df['is_correct_either']]['confidence_val'].mean()
    print(f"Mean Confidence (Correct by at least 1 Rater): {conf_correct:.3f}")
    print(f"Mean Confidence (Incorrect by both Raters):   {conf_incorrect:.3f}")

    # 3. Bar Chart: Side-by-Side MST Counts
    print(f"\nGenerating Side-by-Side MST Count Bar Chart...")
    
    pred_counts = eval_df['mst_name'].value_counts().reindex(CLASSES_5WAY, fill_value=0)
    r1_counts = eval_df['r1_mst5'].value_counts().reindex(CLASSES_5WAY, fill_value=0)
    r2_counts = eval_df['r2_mst5'].value_counts().reindex(CLASSES_5WAY, fill_value=0)
    
    # For baseline, we can also plot the FST-based mapping that was generated initially (true_label)
    fst_counts = eval_df['true_label'].value_counts().reindex(CLASSES_5WAY, fill_value=0)
    
    x = np.arange(len(CLASSES_5WAY))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width*1.5, fst_counts.values, width, label='MSKCC FST Mapping', color='gray')
    ax.bar(x - width/2, r1_counts.values, width, label='MSKCC Rater 1', color='midnightblue')
    ax.bar(x + width/2, r2_counts.values, width, label='MSKCC Rater 2', color='cornflowerblue')
    ax.bar(x + width*1.5, pred_counts.values, width, label='FairFace CNN Preds', color='darkorange')
    
    ax.set_ylabel('Count')
    ax.set_title('MST Category Counts: FairFace vs MSKCC Raters')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES_5WAY, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "mst_category_comparison_bar_chart.png"), dpi=300)
    plt.close()
    
    # Save the consolidated DataFrame
    eval_df.to_csv(os.path.join(args.out_dir, "evaluation_data_vs_raters.csv"), index=False)
    print(f"\nAll evaluation outputs saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()
