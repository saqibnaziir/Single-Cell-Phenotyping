"""
Evaluation Script — Label-Free Single-Cell Phenotyping
=======================================================
"Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"
Saqib Nazir, Ardhendu Behera — Edge Hill University, UK
ICPR 2026 | arXiv:2605.14717

Comprehensive evaluation including:
- Classification metrics (accuracy, confusion matrix, ROC curves)
- Regression metrics (MSE, Pearson correlation per protein marker)
- Qualitative visualizations (predictions, Grad-CAM, uncertainty)

Usage:
    python evaluate.py --model_path checkpoints/run1/best_model.pth \\
                       --data_path /path/to/BSCCMNIST \\
                       --output_dir evaluation_results
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    mutual_info_score, silhouette_score, r2_score, mean_squared_error,
    average_precision_score
)
from scipy import stats
from scipy.stats import f_oneway, kruskal, spearmanr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import gaussian_kde  # For ridge plots

from model import create_model
from data_loading import (
    load_bsccm_dataset,
    create_train_val_test_splits,
    create_data_loaders
)


class MultiTaskEvaluator:
    """Comprehensive evaluator for multi-task model"""
    
    def __init__(self, model, test_loader, device, num_classes=3, num_proteins=10):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.num_proteins = num_proteins
        
        # Class names
        self.class_names = ['Lymphocyte', 'Granulocyte', 'Monocyte']
        
        # Results storage
        self.results = {}
    
    def evaluate(self, mc_samples=1):
        """Complete evaluation on test set"""
        self.model.eval()
        
        all_cls_preds = []
        all_cls_labels = []
        all_cls_probs = []
        all_prot_preds = []
        all_prot_labels = []
        all_images = []
        all_uncertainties = []
        
        print("Evaluating on test set...")
        with torch.no_grad():
            for images, labels, proteins in tqdm(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                proteins = proteins.to(self.device)
                
                # Standard prediction
                cls_logits, prot_preds = self.model(images)
                cls_probs = F.softmax(cls_logits, dim=1)
                
                # Store results
                _, predicted = cls_logits.max(1)
                all_cls_preds.extend(predicted.cpu().numpy())
                all_cls_labels.extend(labels.cpu().numpy())
                all_cls_probs.extend(cls_probs.cpu().numpy())
                all_prot_preds.extend(prot_preds.cpu().numpy())
                all_prot_labels.extend(proteins.cpu().numpy())
                all_images.extend(images.cpu().numpy())
        
        # Convert to numpy
        all_cls_preds = np.array(all_cls_preds)
        all_cls_labels = np.array(all_cls_labels)
        all_cls_probs = np.array(all_cls_probs)
        all_prot_preds = np.array(all_prot_preds)
        all_prot_labels = np.array(all_prot_labels)
        
        # Store results
        self.results = {
            'cls_preds': all_cls_preds,
            'cls_labels': all_cls_labels,
            'cls_probs': all_cls_probs,
            'prot_preds': all_prot_preds,
            'prot_labels': all_prot_labels,
            'images': np.array(all_images)
        }
        
        return self.results
    
    def compute_classification_metrics(self):
        """Compute and print classification metrics"""
        cls_preds = self.results['cls_preds']
        cls_labels = self.results['cls_labels']
        cls_probs = self.results['cls_probs']
        
        # Overall accuracy
        accuracy = accuracy_score(cls_labels, cls_preds)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            cls_labels, cls_preds, average=None, zero_division=0
        )
        
        # Macro-averaged
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            cls_labels, cls_preds, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(cls_labels, cls_preds)
        
        # ROC-AUC
        auc_scores = []
        for i in range(self.num_classes):
            binary_labels = (cls_labels == i).astype(int)
            try:
                auc = roc_auc_score(binary_labels, cls_probs[:, i])
                auc_scores.append(auc)
            except:
                auc_scores.append(0.0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'auc_scores': auc_scores,
            'mean_auc': np.mean(auc_scores)
        }
        
        # Print results
        print("\n" + "="*80)
        print("CLASSIFICATION METRICS")
        print("="*80)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
        
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Support':<10}")
        print("-"*80)
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {auc_scores[i]:<12.4f} {support[i]:<10}")
        
        print("\nConfusion Matrix:")
        print("                Predicted")
        print(f"              {self.class_names[0]:<10} {self.class_names[1]:<10} {self.class_names[2]:<10}")
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {cm[i][0]:<10} {cm[i][1]:<10} {cm[i][2]:<10}")
        
        return metrics
    
    def compute_regression_metrics(self, protein_names):
        """Compute and print regression metrics"""
        prot_preds = self.results['prot_preds']
        prot_labels = self.results['prot_labels']
        
        # Overall MSE
        overall_mse = np.mean((prot_preds - prot_labels) ** 2)
        overall_mae = np.mean(np.abs(prot_preds - prot_labels))
        
        # Per-protein metrics
        per_protein_mse = np.mean((prot_preds - prot_labels) ** 2, axis=0)
        per_protein_corr = [
            np.corrcoef(prot_preds[:, i], prot_labels[:, i])[0, 1]
            for i in range(self.num_proteins)
        ]
        
        # Handle NaN correlations
        per_protein_corr = np.nan_to_num(per_protein_corr, nan=0.0)
        
        metrics = {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'per_protein_mse': per_protein_mse,
            'per_protein_corr': per_protein_corr
        }
        
        # Print results
        print("\n" + "="*80)
        print("REGRESSION METRICS")
        print("="*80)
        print(f"\nOverall MSE: {overall_mse:.4f}")
        print(f"Overall MAE: {overall_mae:.4f}")
        
        print("\nPer-Protein Metrics:")
        print(f"{'Protein':<35} {'MSE':<10} {'Correlation':<12}")
        print("-"*80)
        for i, name in enumerate(protein_names[:self.num_proteins]):
            print(f"{name:<35} {per_protein_mse[i]:<10.4f} {per_protein_corr[i]:<12.4f}")
        
        return metrics
    
    def compute_comprehensive_protein_metrics(self, protein_names):
        """
        Compute comprehensive quantitative protein analysis metrics.
        Returns detailed statistics for publication-quality results.
        """
        prot_preds = self.results['prot_preds']
        prot_labels = self.results['prot_labels']
        cls_labels = self.results['cls_labels']
        
        print("\n" + "="*80)
        print("COMPREHENSIVE PROTEIN ANALYSIS")
        print("="*80)
        
        metrics = {}
        
        # ==================== Per-Protein Regression Metrics ====================
        print("\n1. Computing advanced regression metrics per protein...")
        per_protein_metrics = []
        
        for i in range(self.num_proteins):
            pred = prot_preds[:, i]
            true = prot_labels[:, i]
            
            # Basic metrics
            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred - true))
            
            # Correlation metrics
            pearson_r, pearson_p = stats.pearsonr(pred, true)
            spearman_r, spearman_p = spearmanr(pred, true)
            
            # R² (coefficient of determination)
            r2 = r2_score(true, pred)
            
            # Concordance Correlation Coefficient (CCC)
            # Measures agreement between predictions and truth
            mean_true = np.mean(true)
            mean_pred = np.mean(pred)
            var_true = np.var(true)
            var_pred = np.var(pred)
            cov = np.cov(true, pred)[0, 1]
            ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
            
            # Mean Absolute Percentage Error (MAPE) - careful with zeros
            non_zero_mask = np.abs(true) > 1e-6
            if np.sum(non_zero_mask) > 0:
                mape = np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / true[non_zero_mask])) * 100
            else:
                mape = np.nan
            
            # Coefficient of Variation (CV) for predictions and true values
            cv_true = np.std(true) / (np.abs(np.mean(true)) + 1e-8)
            cv_pred = np.std(pred) / (np.abs(np.mean(pred)) + 1e-8)
            
            per_protein_metrics.append({
                'Protein': protein_names[i],
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'R²': r2,
                'CCC': ccc,
                'MAPE': mape,
                'CV_true': cv_true,
                'CV_pred': cv_pred
            })
        
        metrics['per_protein_regression'] = pd.DataFrame(per_protein_metrics)
        
        # Print summary
        print("\nPer-Protein Regression Metrics:")
        print(f"{'Protein':<35} {'RMSE':<8} {'MAE':<8} {'Pearson':<8} {'Spearman':<8} {'R²':<8} {'CCC':<8}")
        print("-"*95)
        for m in per_protein_metrics:
            print(f"{m['Protein']:<35} {m['RMSE']:<8.4f} {m['MAE']:<8.4f} "
                  f"{m['Pearson_r']:<8.4f} {m['Spearman_r']:<8.4f} {m['R²']:<8.4f} {m['CCC']:<8.4f}")
        
        # ==================== Differential Expression Analysis ====================
        print("\n2. Computing differential expression between cell types...")
        diff_expr_results = []
        
        for i in range(self.num_proteins):
            # Group predictions by cell type
            groups = [prot_preds[cls_labels == ct, i] for ct in range(self.num_classes)]
            
            # ANOVA (parametric test)
            f_stat, anova_p = f_oneway(*groups)
            
            # Kruskal-Wallis (non-parametric alternative)
            h_stat, kruskal_p = kruskal(*groups)
            
            # Mean expression per cell type
            means = [np.mean(g) for g in groups]
            stds = [np.std(g) for g in groups]
            
            # Effect size (largest Cohen's d between any two groups)
            max_cohens_d = 0
            for j in range(self.num_classes):
                for k in range(j+1, self.num_classes):
                    pooled_std = np.sqrt((stds[j]**2 + stds[k]**2) / 2)
                    if pooled_std > 1e-8:
                        cohens_d = abs(means[j] - means[k]) / pooled_std
                        max_cohens_d = max(max_cohens_d, cohens_d)
            
            # Fold change (max vs min)
            max_mean = max(means)
            min_mean = min(means)
            if abs(min_mean) > 1e-6:
                fold_change = max_mean / min_mean if min_mean != 0 else np.inf
            else:
                fold_change = np.nan
            
            diff_expr_results.append({
                'Protein': protein_names[i],
                'ANOVA_F': f_stat,
                'ANOVA_p': anova_p,
                'KruskalWallis_H': h_stat,
                'KruskalWallis_p': kruskal_p,
                'Significant': 'Yes' if anova_p < 0.05 else 'No',
                'Max_Cohens_d': max_cohens_d,
                'Fold_Change': fold_change,
                f'Mean_{self.class_names[0]}': means[0],
                f'Mean_{self.class_names[1]}': means[1],
                f'Mean_{self.class_names[2]}': means[2],
                f'Std_{self.class_names[0]}': stds[0],
                f'Std_{self.class_names[1]}': stds[1],
                f'Std_{self.class_names[2]}': stds[2]
            })
        
        metrics['differential_expression'] = pd.DataFrame(diff_expr_results)
        
        # Print summary
        print("\nDifferential Expression Analysis:")
        print(f"{'Protein':<35} {'ANOVA_p':<10} {'Sig?':<6} {'Cohens_d':<10} {'Fold_Change':<12}")
        print("-"*80)
        for r in diff_expr_results:
            print(f"{r['Protein']:<35} {r['ANOVA_p']:<10.4e} {r['Significant']:<6} "
                  f"{r['Max_Cohens_d']:<10.4f} {r['Fold_Change']:<12.4f}")
        
        # ==================== Cell Type Discrimination Power ====================
        print("\n3. Computing cell type discrimination metrics...")
        discrimination_results = []
        
        for i in range(self.num_proteins):
            pred = prot_preds[:, i]
            
            # Mutual information with cell type
            # Discretize continuous predictions into bins for MI calculation
            n_bins = 10
            bin_edges = np.linspace(pred.min(), pred.max(), n_bins + 1)
            pred_binned = np.digitize(pred, bins=bin_edges[:-1])  # Exclude last edge to avoid out-of-range
            mi = mutual_info_score(cls_labels, pred_binned)
            
            # One-vs-rest ROC-AUC for each cell type
            auc_scores = []
            auprc_scores = []
            for ct in range(self.num_classes):
                binary_labels = (cls_labels == ct).astype(int)
                try:
                    # Normalize predictions to [0, 1] for ROC
                    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                    auc = roc_auc_score(binary_labels, pred_norm)
                    auprc = average_precision_score(binary_labels, pred_norm)
                    auc_scores.append(auc)
                    auprc_scores.append(auprc)
                except:
                    auc_scores.append(0.5)
                    auprc_scores.append(0.0)
            
            # Mean AUC across cell types
            mean_auc = np.mean(auc_scores)
            mean_auprc = np.mean(auprc_scores)
            
            # Silhouette score (how well this protein separates cell types)
            # Reshape for silhouette (needs 2D input)
            pred_2d = pred.reshape(-1, 1)
            try:
                sil_score = silhouette_score(pred_2d, cls_labels)
            except:
                sil_score = 0.0
            
            # Percentage of cells with high expression (>1 std above mean)
            high_expr_threshold = 1.0  # Z-score threshold
            pct_high = np.sum(pred > high_expr_threshold) / len(pred) * 100
            
            discrimination_results.append({
                'Protein': protein_names[i],
                'Mutual_Info': mi,
                'Mean_ROC_AUC': mean_auc,
                'Mean_AUPRC': mean_auprc,
                'Silhouette': sil_score,
                f'AUC_{self.class_names[0]}': auc_scores[0],
                f'AUC_{self.class_names[1]}': auc_scores[1],
                f'AUC_{self.class_names[2]}': auc_scores[2],
                'Pct_High_Expr': pct_high
            })
        
        metrics['discrimination'] = pd.DataFrame(discrimination_results)
        
        # Print summary
        print("\nCell Type Discrimination Power:")
        print(f"{'Protein':<35} {'MI':<8} {'Mean_AUC':<10} {'Silhouette':<12} {'%High':<8}")
        print("-"*80)
        for r in discrimination_results:
            print(f"{r['Protein']:<35} {r['Mutual_Info']:<8.4f} {r['Mean_ROC_AUC']:<10.4f} "
                  f"{r['Silhouette']:<12.4f} {r['Pct_High_Expr']:<8.2f}")
        
        # ==================== Marker Enrichment Analysis ====================
        print("\n4. Computing marker enrichment per cell type...")
        enrichment_results = []
        
        for i in range(self.num_proteins):
            row = {'Protein': protein_names[i]}
            
            for ct in range(self.num_classes):
                # Mean expression in this cell type vs others
                in_type = prot_preds[cls_labels == ct, i]
                out_type = prot_preds[cls_labels != ct, i]
                
                mean_in = np.mean(in_type)
                mean_out = np.mean(out_type)
                
                # Enrichment score (log fold change)
                if abs(mean_out) > 1e-6:
                    enrichment = mean_in - mean_out
                else:
                    enrichment = mean_in
                
                # T-test for significance
                t_stat, t_p = stats.ttest_ind(in_type, out_type)
                
                # Percentage expressing (>0.5 threshold)
                pct_in = np.sum(in_type > 0.5) / len(in_type) * 100 if len(in_type) > 0 else 0
                pct_out = np.sum(out_type > 0.5) / len(out_type) * 100 if len(out_type) > 0 else 0
                
                row[f'{self.class_names[ct]}_Enrichment'] = enrichment
                row[f'{self.class_names[ct]}_p_value'] = t_p
                row[f'{self.class_names[ct]}_Pct_Expr'] = pct_in
            
            enrichment_results.append(row)
        
        metrics['enrichment'] = pd.DataFrame(enrichment_results)
        
        # Print summary
        print("\nMarker Enrichment by Cell Type (Enrichment Score):")
        print(f"{'Protein':<35} {self.class_names[0]:<12} {self.class_names[1]:<12} {self.class_names[2]:<12}")
        print("-"*80)
        for r in enrichment_results:
            print(f"{r['Protein']:<35} "
                  f"{r[f'{self.class_names[0]}_Enrichment']:<12.4f} "
                  f"{r[f'{self.class_names[1]}_Enrichment']:<12.4f} "
                  f"{r[f'{self.class_names[2]}_Enrichment']:<12.4f}")
        
        # ==================== Overall Summary Statistics ====================
        print("\n5. Computing overall summary statistics...")
        
        # Aggregate metrics
        summary = {
            'Mean_RMSE': metrics['per_protein_regression']['RMSE'].mean(),
            'Mean_MAE': metrics['per_protein_regression']['MAE'].mean(),
            'Mean_Pearson': metrics['per_protein_regression']['Pearson_r'].mean(),
            'Mean_Spearman': metrics['per_protein_regression']['Spearman_r'].mean(),
            'Mean_R²': metrics['per_protein_regression']['R²'].mean(),
            'Mean_CCC': metrics['per_protein_regression']['CCC'].mean(),
            'Median_RMSE': metrics['per_protein_regression']['RMSE'].median(),
            'Std_RMSE': metrics['per_protein_regression']['RMSE'].std(),
            'Best_Protein_Pearson': metrics['per_protein_regression'].loc[
                metrics['per_protein_regression']['Pearson_r'].idxmax(), 'Protein'
            ],
            'Worst_Protein_Pearson': metrics['per_protein_regression'].loc[
                metrics['per_protein_regression']['Pearson_r'].idxmin(), 'Protein'
            ],
            'Num_Significant_DE': np.sum(metrics['differential_expression']['ANOVA_p'] < 0.05),
            'Mean_Mutual_Info': metrics['discrimination']['Mutual_Info'].mean(),
            'Mean_Discrimination_AUC': metrics['discrimination']['Mean_ROC_AUC'].mean(),
            'Best_Discriminator': metrics['discrimination'].loc[
                metrics['discrimination']['Mean_ROC_AUC'].idxmax(), 'Protein'
            ]
        }
        
        metrics['summary'] = summary
        
        print("\nOverall Summary Statistics:")
        print("-"*80)
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"{key:<35}: {value:.4f}")
            else:
                print(f"{key:<35}: {value}")
        
        return metrics
    
    def save_protein_metrics_to_csv(self, metrics, output_dir):
        """Save all protein metrics to CSV files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each metrics table
        metrics['per_protein_regression'].to_csv(
            output_dir / 'protein_regression_metrics.csv', index=False
        )
        metrics['differential_expression'].to_csv(
            output_dir / 'protein_differential_expression.csv', index=False
        )
        metrics['discrimination'].to_csv(
            output_dir / 'protein_discrimination_metrics.csv', index=False
        )
        metrics['enrichment'].to_csv(
            output_dir / 'protein_enrichment_by_celltype.csv', index=False
        )
        
        # Save summary as a single-row CSV
        summary_df = pd.DataFrame([metrics['summary']])
        summary_df.to_csv(output_dir / 'protein_analysis_summary.csv', index=False)
        
        print(f"\n✅ All protein metrics saved to CSV files in {output_dir}")
        
        # Also save a combined report
        with open(output_dir / 'protein_analysis_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE PROTEIN ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. REGRESSION METRICS\n")
            f.write("-"*80 + "\n")
            f.write(metrics['per_protein_regression'].to_string())
            f.write("\n\n")
            
            f.write("2. DIFFERENTIAL EXPRESSION\n")
            f.write("-"*80 + "\n")
            f.write(metrics['differential_expression'].to_string())
            f.write("\n\n")
            
            f.write("3. DISCRIMINATION METRICS\n")
            f.write("-"*80 + "\n")
            f.write(metrics['discrimination'].to_string())
            f.write("\n\n")
            
            f.write("4. ENRICHMENT BY CELL TYPE\n")
            f.write("-"*80 + "\n")
            f.write(metrics['enrichment'].to_string())
            f.write("\n\n")
            
            f.write("5. SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n")
            for key, value in metrics['summary'].items():
                if isinstance(value, (int, float)):
                    f.write(f"{key:<35}: {value:.4f}\n")
                else:
                    f.write(f"{key:<35}: {value}\n")
        
        print(f"✅ Comprehensive report saved to {output_dir / 'protein_analysis_report.txt'}")
    
    def plot_protein_metrics_summary(self, metrics, protein_names, save_path='protein_metrics_summary.jpg'):
        """
        Create a comprehensive visual summary of all protein metrics.
        Multiple panels showing different aspects of protein analysis.
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Regression Performance (RMSE vs Correlation)
        ax1 = fig.add_subplot(gs[0, 0])
        df_reg = metrics['per_protein_regression']
        scatter = ax1.scatter(df_reg['RMSE'], df_reg['Pearson_r'], 
                             s=100, alpha=0.6, c=df_reg['R²'], cmap='viridis')
        for i, txt in enumerate(df_reg['Protein']):
            ax1.annotate(txt.split('_')[0] if '_' in txt else txt[:8], 
                        (df_reg['RMSE'].iloc[i], df_reg['Pearson_r'].iloc[i]),
                        fontsize=7, ha='right')
        ax1.set_xlabel('RMSE', fontweight='bold')
        ax1.set_ylabel('Pearson Correlation', fontweight='bold')
        ax1.set_title('Regression Performance', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='R²')
        
        # 2. Correlation Comparison (Pearson vs Spearman)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df_reg['Pearson_r'], df_reg['Spearman_r'], s=100, alpha=0.6)
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax2.set_xlabel('Pearson Correlation', fontweight='bold')
        ax2.set_ylabel('Spearman Correlation', fontweight='bold')
        ax2.set_title('Linear vs Rank Correlation', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Concordance Correlation Coefficient
        ax3 = fig.add_subplot(gs[0, 2])
        protein_abbrev = [name.split('_')[0] if '_' in name else name[:12] 
                         for name in df_reg['Protein']]
        colors_ccc = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' 
                      for x in df_reg['CCC']]
        ax3.barh(range(len(df_reg)), df_reg['CCC'], color=colors_ccc, alpha=0.7)
        ax3.set_yticks(range(len(df_reg)))
        ax3.set_yticklabels(protein_abbrev, fontsize=9)
        ax3.set_xlabel('CCC (Agreement)', fontweight='bold')
        ax3.set_title('Concordance Correlation\n(>0.7=Good)', fontweight='bold')
        ax3.axvline(x=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax3.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Differential Expression (Effect Sizes)
        ax4 = fig.add_subplot(gs[1, 0])
        df_de = metrics['differential_expression']
        colors_sig = ['green' if p < 0.05 else 'gray' for p in df_de['ANOVA_p']]
        ax4.barh(range(len(df_de)), df_de['Max_Cohens_d'], color=colors_sig, alpha=0.7)
        ax4.set_yticks(range(len(df_de)))
        ax4.set_yticklabels(protein_abbrev, fontsize=9)
        ax4.set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
        ax4.set_title('Differential Expression\n(Green=Significant p<0.05)', fontweight='bold')
        ax4.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Medium')
        ax4.axvline(x=0.8, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Large')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Discrimination Power (AUC)
        ax5 = fig.add_subplot(gs[1, 1])
        df_disc = metrics['discrimination']
        ax5.barh(range(len(df_disc)), df_disc['Mean_ROC_AUC'], 
                color='steelblue', alpha=0.7)
        ax5.set_yticks(range(len(df_disc)))
        ax5.set_yticklabels(protein_abbrev, fontsize=9)
        ax5.set_xlabel('Mean ROC-AUC', fontweight='bold')
        ax5.set_title('Cell Type Discrimination\n(AUC across cell types)', fontweight='bold')
        ax5.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random')
        ax5.axvline(x=0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.set_xlim([0.4, 1.0])
        
        # 6. Mutual Information
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.barh(range(len(df_disc)), df_disc['Mutual_Info'], 
                color='coral', alpha=0.7)
        ax6.set_yticks(range(len(df_disc)))
        ax6.set_yticklabels(protein_abbrev, fontsize=9)
        ax6.set_xlabel('Mutual Information', fontweight='bold')
        ax6.set_title('Information Content\n(Higher=More informative)', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 7. Enrichment Heatmap
        ax7 = fig.add_subplot(gs[2, :2])
        df_enrich = metrics['enrichment']
        enrich_matrix = []
        for _, row in df_enrich.iterrows():
            enrich_matrix.append([
                row[f'{self.class_names[0]}_Enrichment'],
                row[f'{self.class_names[1]}_Enrichment'],
                row[f'{self.class_names[2]}_Enrichment']
            ])
        enrich_matrix = np.array(enrich_matrix)
        
        im = ax7.imshow(enrich_matrix.T, aspect='auto', cmap='RdBu_r', 
                       vmin=-2, vmax=2, interpolation='nearest')
        ax7.set_yticks(range(self.num_classes))
        ax7.set_yticklabels(self.class_names, fontweight='bold')
        ax7.set_xticks(range(len(df_enrich)))
        ax7.set_xticklabels(protein_abbrev, rotation=45, ha='right')
        ax7.set_xlabel('Protein Markers', fontweight='bold')
        ax7.set_ylabel('Cell Type', fontweight='bold')
        ax7.set_title('Marker Enrichment by Cell Type\n(Red=Enriched, Blue=Depleted)', 
                     fontweight='bold')
        plt.colorbar(im, ax=ax7, label='Enrichment Score')
        
        # 8. Summary Statistics Table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        summary = metrics['summary']
        summary_text = [
            "OVERALL SUMMARY",
            "-" * 30,
            f"Mean RMSE: {summary['Mean_RMSE']:.4f}",
            f"Mean MAE: {summary['Mean_MAE']:.4f}",
            f"Mean Pearson: {summary['Mean_Pearson']:.4f}",
            f"Mean R²: {summary['Mean_R²']:.4f}",
            f"Mean CCC: {summary['Mean_CCC']:.4f}",
            "",
            f"Best Protein:",
            f"  {summary['Best_Protein_Pearson'][:20]}",
            f"Worst Protein:",
            f"  {summary['Worst_Protein_Pearson'][:20]}",
            "",
            f"Significant DE: {summary['Num_Significant_DE']}/{self.num_proteins}",
            f"Mean MI: {summary['Mean_Mutual_Info']:.4f}",
            f"Mean Discrim AUC: {summary['Mean_Discrimination_AUC']:.4f}",
            "",
            f"Best Discriminator:",
            f"  {summary['Best_Discriminator'][:20]}"
        ]
        ax8.text(0.1, 0.95, '\n'.join(summary_text), 
                transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Comprehensive Protein Analysis Summary', 
                    fontweight='bold', fontsize=16)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein metrics summary plot saved to {save_path}")
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.jpg'):
        """Plot confusion matrix"""
        cm = self.results.get('classification_metrics', {}).get('confusion_matrix')
        if cm is None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.results['cls_labels'], self.results['cls_preds'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, save_path='roc_curves.jpg'):
        """Plot ROC curves for each class"""
        cls_labels = self.results['cls_labels']
        cls_probs = self.results['cls_probs']
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            binary_labels = (cls_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, cls_probs[:, i])
            auc = roc_auc_score(binary_labels, cls_probs[:, i])
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves (One-vs-Rest)', fontweight='bold', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curves saved to {save_path}")
    
    def plot_protein_predictions(self, protein_names, num_proteins=6, save_path='protein_predictions.jpg'):
        """Plot true vs predicted protein expression"""
        prot_preds = self.results['prot_preds']
        prot_labels = self.results['prot_labels']
        
        n_cols = 3
        n_rows = (num_proteins + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx in range(min(num_proteins, len(protein_names))):
            ax = axes[idx]
            name = protein_names[idx]
            
            # Scatter plot
            ax.scatter(prot_labels[:, idx], prot_preds[:, idx], 
                      alpha=0.3, s=10)
            
            # Perfect prediction line
            min_val = min(prot_labels[:, idx].min(), prot_preds[:, idx].min())
            max_val = max(prot_labels[:, idx].max(), prot_preds[:, idx].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Correlation
            corr = np.corrcoef(prot_labels[:, idx], prot_preds[:, idx])[0, 1]
            ax.set_title(f'{name}\nCorr={corr:.3f}', fontweight='bold')
            ax.set_xlabel('True Expression')
            ax.set_ylabel('Predicted Expression')
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for idx in range(num_proteins, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein predictions saved to {save_path}")
    
    def plot_sample_predictions(self, num_samples=16, save_path='sample_predictions.jpg'):
        """Plot sample predictions with true and predicted labels"""
        images = self.results['images']
        cls_preds = self.results['cls_preds']
        cls_labels = self.results['cls_labels']
        cls_probs = self.results['cls_probs']
        
        # Randomly select samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                img_idx = indices[idx]
                img = images[img_idx, 0]  # First channel
                
                ax.imshow(img, cmap='gray')
                
                # Labels
                true_label = self.class_names[cls_labels[img_idx]]
                pred_label = self.class_names[cls_preds[img_idx]]
                confidence = cls_probs[img_idx, cls_preds[img_idx]]
                
                # Color based on correctness
                color = 'green' if true_label == pred_label else 'red'
                
                ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                           color=color, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Sample predictions saved to {save_path}")
    
    def analyze_errors(self):
        """Analyze prediction errors"""
        cls_preds = self.results['cls_preds']
        cls_labels = self.results['cls_labels']
        
        # Find errors
        error_mask = cls_preds != cls_labels
        error_indices = np.where(error_mask)[0]
        
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)
        print(f"\nTotal errors: {len(error_indices)} / {len(cls_labels)} "
              f"({100*len(error_indices)/len(cls_labels):.2f}%)")
        
        # Error confusion (what was predicted incorrectly)
        error_confusion = np.zeros((self.num_classes, self.num_classes))
        for idx in error_indices:
            error_confusion[cls_labels[idx], cls_preds[idx]] += 1
        
        print("\nError Confusion Matrix:")
        print("                Predicted (Errors Only)")
        print(f"              {self.class_names[0]:<10} {self.class_names[1]:<10} {self.class_names[2]:<10}")
        for i, name in enumerate(self.class_names):
            print(f"{name:<15} {int(error_confusion[i,0]):<10} {int(error_confusion[i,1]):<10} {int(error_confusion[i,2]):<10}")
        
        return error_indices, error_confusion
    
    def plot_protein_heatmap(self, protein_names, num_cells=500, save_path='protein_heatmap.jpg'):
        """
        Plot heatmap of protein expression across cells, organized by cell type.
        This is the standard way to visualize protein expression in single-cell biology.
        
        What it shows: Each row is a cell, each column is a protein marker.
        Colors represent expression levels (normalized Z-scores).
        Cells are grouped by predicted cell type.
        
        Interpretation:
        - Red/positive values: High expression of that protein
        - Blue/negative values: Low/negative expression
        - White: Near-zero expression (baseline)
        """
        prot_preds = self.results['prot_preds']
        cls_preds = self.results['cls_preds']
        
        # Sample cells if too many
        if len(prot_preds) > num_cells:
            indices = np.random.choice(len(prot_preds), num_cells, replace=False)
            prot_preds = prot_preds[indices]
            cls_preds = cls_preds[indices]
        
        # Sort by cell type, then by first protein expression
        sort_indices = np.lexsort((prot_preds[:, 0], cls_preds))
        prot_preds_sorted = prot_preds[sort_indices]
        cls_preds_sorted = cls_preds[sort_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 16))
        
        # Plot heatmap
        im = ax.imshow(prot_preds_sorted, aspect='auto', cmap='RdBu_r', 
                      vmin=-2, vmax=2, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Normalized Expression (Z-score)', pad=0.02)
        
        # Add cell type annotations
        cell_type_bounds = []
        current_type = cls_preds_sorted[0]
        start_idx = 0
        for i, cell_type in enumerate(cls_preds_sorted):
            if cell_type != current_type:
                cell_type_bounds.append((start_idx, i, current_type))
                start_idx = i
                current_type = cell_type
        cell_type_bounds.append((start_idx, len(cls_preds_sorted), current_type))
        
        # Add dividing lines and labels
        y_pos = 0
        for start, end, cell_type in cell_type_bounds:
            mid = (start + end) / 2
            ax.axhline(y=start, color='black', linewidth=2, linestyle='-')
            ax.text(-0.5, mid, self.class_names[cell_type], 
                   ha='right', va='center', fontweight='bold', fontsize=10,
                   transform=ax.get_yaxis_transform())
        
        # Labels
        ax.set_xlabel('Protein Markers', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cells (sorted by type)', fontweight='bold', fontsize=12)
        ax.set_title('Protein Expression Heatmap\n(Normalized Z-scores)', 
                    fontweight='bold', fontsize=14, pad=20)
        
        # Set x-ticks to protein names (abbreviate if too long)
        protein_labels = [name.split('_')[0] if '_' in name else name[:15] 
                         for name in protein_names[:self.num_proteins]]
        ax.set_xticks(range(len(protein_labels)))
        ax.set_xticklabels(protein_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein heatmap saved to {save_path}")
    
    def plot_protein_violin(self, protein_names, save_path='protein_violin_plots.jpg'):
        """
        Plot violin plots showing protein expression distribution per cell type.
        Standard visualization in single-cell biology (like Scanpy, Seurat).
        
        What it shows: Distribution of each protein's expression across different cell types.
        The width of the violin at each value represents the density of cells with that expression level.
        
        Interpretation:
        - Wider violins: More cells express at that level
        - Median line: Typical expression level for that cell type
        - Different shapes between cell types: Distinct expression patterns
        """
        prot_preds = self.results['prot_preds']
        cls_labels = self.results['cls_labels']
        
        n_proteins = min(self.num_proteins, len(protein_names))
        n_cols = 2
        n_rows = (n_proteins + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx in range(n_proteins):
            ax = axes[idx]
            protein_name = protein_names[idx].split('_')[0] if '_' in protein_names[idx] else protein_names[idx][:20]
            
            # Prepare data for violin plot
            data_for_plot = []
            labels_for_plot = []
            for cell_type in range(self.num_classes):
                mask = cls_labels == cell_type
                data_for_plot.append(prot_preds[mask, idx])
                labels_for_plot.append(self.class_names[cell_type])
            
            # Create violin plot
            parts = ax.violinplot(data_for_plot, positions=range(self.num_classes), 
                                  showmeans=True, showmedians=True)
            
            # Customize colors
            colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
            for pc, color in zip(parts['bodies'], colors[:self.num_classes]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_linewidth(1.5)
            
            ax.set_xticks(range(self.num_classes))
            ax.set_xticklabels(labels_for_plot, fontweight='bold')
            ax.set_ylabel('Normalized Expression (Z-score)', fontweight='bold')
            ax.set_title(f'{protein_name}', fontweight='bold', fontsize=12)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove unused subplots
        for idx in range(n_proteins, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Protein Expression Distribution by Cell Type\n(Violin Plots)', 
                    fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein violin plots saved to {save_path}")
    
    def plot_protein_ridge(self, protein_names, save_path='protein_ridge_plots.jpg'):
        """
        Plot ridge plots (overlapping density distributions) for protein expression.
        Alternative to violin plots, shows density curves stacked vertically.
        
        What it shows: Probability density of protein expression for each cell type.
        Each "ridge" represents one cell type, showing how expression is distributed.
        
        Interpretation:
        - Peaks: Most common expression levels
        - Width: Variability in expression
        - Separation: How distinct cell types are based on this protein
        """
        prot_preds = self.results['prot_preds']
        cls_labels = self.results['cls_labels']
        
        n_proteins = min(self.num_proteins, len(protein_names))
        n_cols = 2
        n_rows = (n_proteins + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
        
        for idx in range(n_proteins):
            ax = axes[idx]
            protein_name = protein_names[idx].split('_')[0] if '_' in protein_names[idx] else protein_names[idx][:20]
            
            # Compute density for each cell type
            x_min = prot_preds[:, idx].min() - 1
            x_max = prot_preds[:, idx].max() + 1
            x = np.linspace(x_min, x_max, 200)
            
            y_offset = 0
            for cell_type in range(self.num_classes):
                mask = cls_labels == cell_type
                data = prot_preds[mask, idx]
                
                if len(data) > 1:
                    try:
                        kde = gaussian_kde(data)
                        density = kde(x)
                        density = density / density.max()  # Normalize to [0, 1]
                        
                        # Plot filled curve
                        ax.fill_between(x, y_offset, y_offset + density, 
                                       alpha=0.6, color=colors[cell_type],
                                       label=self.class_names[cell_type])
                        
                        # Plot line
                        ax.plot(x, y_offset + density, color='black', linewidth=1.5)
                        
                        # Add mean line
                        mean_val = np.mean(data)
                        mean_dens = kde(mean_val) / kde(x).max()
                        ax.plot([mean_val, mean_val], [y_offset, y_offset + mean_dens],
                               color='black', linewidth=2, linestyle='--')
                        
                        y_offset += 1.2
                    except:
                        pass
            
            ax.set_xlabel('Normalized Expression (Z-score)', fontweight='bold')
            ax.set_ylabel('Density (scaled)', fontweight='bold')
            ax.set_title(f'{protein_name}', fontweight='bold', fontsize=12)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            if idx == 0:
                ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')
        
        # Remove unused subplots
        for idx in range(n_proteins, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Protein Expression Density by Cell Type\n(Ridge Plots)', 
                    fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein ridge plots saved to {save_path}")
    
    def plot_protein_feature_plots(self, protein_names, num_cells_per_type=200, 
                                   save_path='protein_feature_plots.jpg'):
        """
        Plot feature plots style visualization - each protein shown as separate panels.
        Similar to Scanpy's sc.pl.umap with multiple features.
        
        What it shows: Each cell plotted with color intensity representing protein expression.
        Cells are positioned by their type, with color showing expression level.
        
        Interpretation:
        - Red cells: High expression
        - Blue cells: Low/negative expression
        - White: Baseline expression
        - Clustering by color: Similar expression patterns
        """
        prot_preds = self.results['prot_preds']
        cls_labels = self.results['cls_labels']
        
        # Sample cells per type
        sampled_indices = []
        for cell_type in range(self.num_classes):
            type_indices = np.where(cls_labels == cell_type)[0]
            if len(type_indices) > num_cells_per_type:
                sampled = np.random.choice(type_indices, num_cells_per_type, replace=False)
            else:
                sampled = type_indices
            sampled_indices.extend(sampled)
        
        prot_preds_sampled = prot_preds[sampled_indices]
        cls_labels_sampled = cls_labels[sampled_indices]
        
        # Create 2D embedding (simple PCA-like or just use protein 1 vs protein 2)
        # For simplicity, use first two proteins as coordinates
        if self.num_proteins >= 2:
            x_coords = prot_preds_sampled[:, 0]
            y_coords = prot_preds_sampled[:, 1]
        else:
            # Fallback to random if only one protein
            x_coords = np.random.randn(len(prot_preds_sampled))
            y_coords = prot_preds_sampled[:, 0]
        
        n_proteins = min(self.num_proteins, len(protein_names))
        n_cols = 2
        n_rows = (n_proteins + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx in range(n_proteins):
            ax = axes[idx]
            protein_name = protein_names[idx].split('_')[0] if '_' in protein_names[idx] else protein_names[idx][:20]
            
            # Scatter plot colored by expression
            scatter = ax.scatter(x_coords, y_coords, c=prot_preds_sampled[:, idx],
                               cmap='RdBu_r', s=20, alpha=0.6, vmin=-2, vmax=2,
                               edgecolors='black', linewidths=0.3)
            
            ax.set_xlabel('Component 1', fontweight='bold')
            ax.set_ylabel('Component 2', fontweight='bold')
            ax.set_title(f'{protein_name}', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Expression', pad=0.02)
        
        # Remove unused subplots
        for idx in range(n_proteins, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Plots: Protein Expression on Cell Space', 
                    fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Protein feature plots saved to {save_path}")
    
    def plot_cell_protein_overlay(self, protein_names, num_samples=16, 
                                  save_path='cell_protein_overlay.jpg'):
        """
        Plot cell images with protein expression overlaid as colored heatmaps.
        
        What it shows: Original cell image with a color overlay representing protein expression.
        Each panel shows one cell with all proteins combined (weighted average) or individual proteins.
        
        Interpretation:
        - Color intensity: Level of protein expression on that cell
        - Overlay on morphology: How protein expression relates to cell structure
        """
        images = self.results['images']
        prot_preds = self.results['prot_preds']
        cls_preds = self.results['cls_preds']
        cls_labels = self.results['cls_labels']
        
        # Select diverse samples (one per cell type, then random)
        selected_indices = []
        for cell_type in range(self.num_classes):
            type_indices = np.where(cls_labels == cell_type)[0]
            if len(type_indices) > 0:
                selected_indices.append(np.random.choice(type_indices))
        
        # Fill remaining slots randomly
        remaining = num_samples - len(selected_indices)
        if remaining > 0:
            other_indices = np.setdiff1d(np.arange(len(images)), selected_indices)
            if len(other_indices) >= remaining:
                selected_indices.extend(np.random.choice(other_indices, remaining, replace=False))
        
        n_cols = 4
        n_rows = (num_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5*n_rows))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(selected_indices):
                img_idx = selected_indices[idx]
                img = images[img_idx, 0]  # First channel
                proteins = prot_preds[img_idx]
                
                # Normalize protein expression to [0, 1] for overlay
                # Use average of all proteins as "expression score"
                expr_score = np.mean(proteins) + 1  # Shift to positive
                expr_score = np.clip(expr_score / 2, 0, 1)  # Normalize to [0, 1]
                
                # Display base image
                ax.imshow(img, cmap='gray', alpha=0.7)
                
                # Create colored overlay (red for high, blue for low)
                overlay = np.zeros((*img.shape, 3))
                if expr_score > 0.5:
                    # Red for positive expression
                    overlay[:, :, 0] = (expr_score - 0.5) * 2  # Red channel
                else:
                    # Blue for negative expression
                    overlay[:, :, 2] = (0.5 - expr_score) * 2  # Blue channel
                
                ax.imshow(overlay, alpha=0.4, interpolation='bilinear')
                
                # Labels
                true_label = self.class_names[cls_labels[img_idx]]
                pred_label = self.class_names[cls_preds[img_idx]]
                color = 'green' if true_label == pred_label else 'red'
                
                # Protein expression summary
                top_protein_idx = np.argmax(np.abs(proteins))
                top_protein_name = protein_names[top_protein_idx].split('_')[0] if '_' in protein_names[top_protein_idx] else protein_names[top_protein_idx][:10]
                top_protein_val = proteins[top_protein_idx]
                
                ax.set_title(f'{true_label} → {pred_label}\n{top_protein_name}: {top_protein_val:.2f}',
                           color=color, fontweight='bold', fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle('Cell Images with Protein Expression Overlay\n(Red=High, Blue=Low)', 
                    fontweight='bold', fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Cell protein overlay saved to {save_path}")
    
    def plot_cell_protein_profiles(self, protein_names, num_samples=12, 
                                   save_path='cell_protein_profiles.jpg'):
        """
        Plot combined view: cell images alongside their protein expression bar charts.
        Most comprehensive view showing both morphology and expression profile.
        
        What it shows: For each cell, shows the actual image next to a bar chart of all protein markers.
        
        Interpretation:
        - Image: Cell morphology and structure
        - Bar chart: Quantitative protein expression profile (normalized Z-scores)
        - Positive bars: Proteins expressed above average
        - Negative bars: Proteins expressed below average
        """
        images = self.results['images']
        prot_preds = self.results['prot_preds']
        cls_preds = self.results['cls_preds']
        cls_labels = self.results['cls_labels']
        
        # Select diverse samples
        selected_indices = []
        for cell_type in range(self.num_classes):
            type_indices = np.where(cls_labels == cell_type)[0]
            if len(type_indices) > 0:
                selected_indices.append(np.random.choice(type_indices))
        
        remaining = num_samples - len(selected_indices)
        if remaining > 0:
            other_indices = np.setdiff1d(np.arange(len(images)), selected_indices)
            if len(other_indices) >= remaining:
                selected_indices.extend(np.random.choice(other_indices, remaining, replace=False))
        
        n_cols = 2  # Image and bar chart side by side
        n_rows = num_samples
        
        fig = plt.figure(figsize=(16, 3*n_rows))
        
        for idx, img_idx in enumerate(selected_indices):
            img = images[img_idx, 0]
            proteins = prot_preds[img_idx]
            true_label = self.class_names[cls_labels[img_idx]]
            pred_label = self.class_names[cls_preds[img_idx]]
            
            # Cell image
            ax1 = plt.subplot(n_rows, n_cols, idx*2 + 1)
            ax1.imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax1.set_title(f'Cell {img_idx}: {true_label} → {pred_label}', 
                         color=color, fontweight='bold')
            ax1.axis('off')
            
            # Protein expression bar chart
            ax2 = plt.subplot(n_rows, n_cols, idx*2 + 2)
            protein_abbrev = [name.split('_')[0] if '_' in name else name[:15] 
                             for name in protein_names[:self.num_proteins]]
            colors_bar = ['red' if p > 0 else 'blue' for p in proteins]
            bars = ax2.barh(range(len(proteins)), proteins, color=colors_bar, alpha=0.7)
            ax2.set_yticks(range(len(proteins)))
            ax2.set_yticklabels(protein_abbrev, fontsize=9)
            ax2.set_xlabel('Normalized Expression (Z-score)', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax2.set_title('Protein Expression Profile', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Cell Morphology and Protein Expression Profiles', 
                    fontweight='bold', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Cell protein profiles saved to {save_path}")


# ==================== MAIN ==================== #

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Model')
    
    # Paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to BSCCMNIST dataset directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    # Model config
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_proteins', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--use_cross_task_attention', action='store_true', default=False,
                       help='Use cross-task attention (deprecated, use --use_task_gating)')
    parser.add_argument('--use_task_gating', action='store_true', default=True,
                       help='Use efficient task gating (default: True)')
    
    # Data
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATION CONFIGURATION")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key:25s}: {value}")
    print("="*80 + "\n")
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Create model with appropriate parameters based on which model is available
    if USING_OPTIMIZED_MODEL:
        # Optimized model uses use_task_gating
        use_gating = getattr(args, 'use_task_gating', True)
        model = create_model(
            num_classes=args.num_classes,
            num_proteins=args.num_proteins,
            img_size=args.img_size,
            in_channels=args.in_channels,
            use_task_gating=use_gating
        )
    else:
        # Improved model uses use_cross_task_attention
        use_cross_task = getattr(args, 'use_cross_task_attention', args.use_cross_task_attention)
        model = create_model(
            num_classes=args.num_classes,
            num_proteins=args.num_proteins,
            img_size=args.img_size,
            in_channels=args.in_channels,
            use_cross_task_attention=use_cross_task
        )
    # Load model state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("✅ Model loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Error loading model state dict: {e}")
        print("\nTrying to load with strict=False (ignoring mismatched keys)...")
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Model loaded with some mismatched keys (this may affect performance)")
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            print("\nPossible causes:")
            print("1. Model architecture doesn't match checkpoint")
            print("2. Checkpoint was saved with different model (optimized vs improved)")
            print("3. Model parameters changed after training")
            raise
    
    # Load data
    print("\nLoading dataset...")
    from enhanced_data_loading import PROTEIN_MARKERS
    bsccm, indices, class_labels, sm_df, protein_markers = load_bsccm_dataset(
        args.data_path,
        protein_markers=PROTEIN_MARKERS
    )
    
    # Create splits (same as training)
    from sklearn.model_selection import train_test_split
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, class_labels, test_size=0.3, stratify=class_labels, random_state=args.seed
    )
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )
    
    splits = {
        'train': (train_indices, train_labels),
        'val': (val_indices, val_labels),
        'test': (test_indices, test_labels)
    }
    
    # Create loaders (need train for normalization stats)
    # Use multi_channel=True to match training configuration
    loaders, datasets = create_data_loaders(
        args.data_path,
        splits,
        sm_df,
        protein_markers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=False,
        multi_channel=True  # Use 4-channel DPC input
    )
    
    # Evaluate
    evaluator = MultiTaskEvaluator(
        model, loaders['test'], device,
        num_classes=args.num_classes,
        num_proteins=len(protein_markers)
    )
    
    evaluator.evaluate()
    
    # Compute metrics
    cls_metrics = evaluator.compute_classification_metrics()
    evaluator.results['classification_metrics'] = cls_metrics
    
    reg_metrics = evaluator.compute_regression_metrics(protein_markers)
    
    # Compute comprehensive protein metrics
    protein_metrics = evaluator.compute_comprehensive_protein_metrics(protein_markers)
    
    # Save all metrics to CSV
    evaluator.save_protein_metrics_to_csv(protein_metrics, output_dir)
    
    # Plot comprehensive metrics summary
    print("\nGenerating comprehensive protein metrics summary visualization...")
    evaluator.plot_protein_metrics_summary(
        protein_metrics, 
        protein_markers, 
        save_path=output_dir / 'protein_metrics_summary.jpg'
    )
    
    # Standard visualizations
    print("\n" + "="*80)
    print("GENERATING STANDARD VISUALIZATIONS")
    print("="*80)
    evaluator.plot_confusion_matrix(output_dir / 'confusion_matrix.jpg')
    evaluator.plot_roc_curves(output_dir / 'roc_curves.jpg')
    evaluator.plot_protein_predictions(
        protein_markers,
        num_proteins=min(9, len(protein_markers)),
        save_path=output_dir / 'protein_predictions.jpg'
    )
    evaluator.plot_sample_predictions(num_samples=16, save_path=output_dir / 'sample_predictions.jpg')
    
    # Protein expression visualizations (standard single-cell biology methods)
    print("\n" + "="*80)
    print("GENERATING PROTEIN EXPRESSION VISUALIZATIONS")
    print("="*80)
    print("\n1. Heatmap - Shows protein expression across all cells organized by cell type")
    evaluator.plot_protein_heatmap(
        protein_markers,
        num_cells=500,
        save_path=output_dir / 'protein_heatmap.jpg'
    )
    
    print("\n2. Violin Plots - Distribution of protein expression per cell type")
    evaluator.plot_protein_violin(
        protein_markers,
        save_path=output_dir / 'protein_violin_plots.jpg'
    )
    
    print("\n3. Ridge Plots - Density distributions of protein expression")
    evaluator.plot_protein_ridge(
        protein_markers,
        save_path=output_dir / 'protein_ridge_plots.jpg'
    )
    
    print("\n4. Feature Plots - Protein expression mapped to cell space")
    evaluator.plot_protein_feature_plots(
        protein_markers,
        num_cells_per_type=200,
        save_path=output_dir / 'protein_feature_plots.jpg'
    )
    
    print("\n5. Cell Images with Protein Overlay - Morphology + expression overlay")
    evaluator.plot_cell_protein_overlay(
        protein_markers,
        num_samples=16,
        save_path=output_dir / 'cell_protein_overlay.jpg'
    )
    
    print("\n6. Cell Protein Profiles - Combined morphology and expression bar charts")
    evaluator.plot_cell_protein_profiles(
        protein_markers,
        num_samples=12,
        save_path=output_dir / 'cell_protein_profiles.jpg'
    )
    
    # Error analysis
    error_indices, error_confusion = evaluator.analyze_errors()
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)
    
    print("\n" + "="*80)
    print("QUANTITATIVE PROTEIN METRICS GUIDE")
    print("="*80)
    print("""
COMPREHENSIVE QUANTITATIVE METRICS COMPUTED:

1. **REGRESSION METRICS** (per protein):
   - RMSE: Root Mean Squared Error (lower = better)
   - MAE: Mean Absolute Error (lower = better)
   - Pearson r: Linear correlation coefficient (-1 to 1, closer to ±1 = better)
   - Spearman r: Rank correlation (non-parametric, robust to outliers)
   - R²: Coefficient of determination (0-1, proportion of variance explained)
   - CCC: Concordance Correlation Coefficient (measures agreement, >0.7 = good)
   - MAPE: Mean Absolute Percentage Error
   - CV: Coefficient of Variation (variability)

2. **DIFFERENTIAL EXPRESSION ANALYSIS**:
   - ANOVA F-statistic & p-value: Tests if protein differs across cell types
   - Kruskal-Wallis H & p-value: Non-parametric alternative to ANOVA
   - Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)
   - Fold Change: Ratio of max to min expression across cell types
   - Mean & Std per cell type

3. **CELL TYPE DISCRIMINATION**:
   - Mutual Information: How much protein expression tells us about cell type
   - ROC-AUC per cell type: Discriminative power (0.5=random, 1.0=perfect)
   - AUPRC: Area Under Precision-Recall Curve
   - Silhouette Score: How well protein separates cell types (-1 to 1)
   - % High Expression: Percentage of cells with expression >1 std

4. **MARKER ENRICHMENT**:
   - Enrichment Score: Difference in mean expression (in-type vs out-type)
   - T-test p-value: Statistical significance of enrichment
   - % Expressing: Percentage of cells expressing above threshold per cell type

5. **SUMMARY STATISTICS**:
   - Aggregate metrics across all proteins
   - Best/worst performing proteins
   - Number of significantly differentially expressed proteins
   - Overall discrimination power

OUTPUT FILES:
- protein_regression_metrics.csv: All regression metrics per protein
- protein_differential_expression.csv: DE analysis results
- protein_discrimination_metrics.csv: Cell type discrimination power
- protein_enrichment_by_celltype.csv: Marker enrichment per cell type
- protein_analysis_summary.csv: Overall summary statistics
- protein_analysis_report.txt: Complete text report
- protein_metrics_summary.jpg: Visual summary of all metrics
    """)
    
    print("\n" + "="*80)
    print("PROTEIN EXPRESSION VISUALIZATION GUIDE")
    print("="*80)
    print("""
The following visualizations show protein expression in different ways commonly used in single-cell biology:

1. **Heatmap (protein_heatmap.jpg)**:
   - Rows = cells, Columns = protein markers
   - Colors show expression levels (red=high, blue=low, white=baseline)
   - Cells grouped by predicted cell type
   - Best for: Seeing overall patterns across all cells

2. **Violin Plots (protein_violin_plots.jpg)**:
   - Shows distribution of each protein's expression per cell type
   - Width = density of cells at that expression level
   - Median line shows typical expression
   - Best for: Comparing expression distributions between cell types

3. **Ridge Plots (protein_ridge_plots.jpg)**:
   - Overlapping density curves for each cell type
   - Shows probability distribution of expression
   - Best for: Visualizing expression variability and overlap between types

4. **Feature Plots (protein_feature_plots.jpg)**:
   - Each cell plotted in 2D space, colored by protein expression
   - Similar to Scanpy/Seurat feature plots
   - Best for: Seeing how protein expression clusters in cell space

5. **Cell Overlay (cell_protein_overlay.jpg)**:
   - Actual cell images with colored protein expression overlay
   - Red = high expression, Blue = low expression
   - Best for: Linking morphology to protein expression

6. **Cell Profiles (cell_protein_profiles.jpg)**:
   - Cell image + bar chart of all protein markers
   - Most comprehensive single-cell view
   - Best for: Detailed inspection of individual cells

**What normalized Z-scores mean:**
- Values are Z-score normalized (mean=0, std=1) from training data
- Positive values: Expression above average
- Negative values: Expression below average
- Zero: Average expression level
- Typical range: -2 to +2 (values outside this are very high/low)

**Biological Interpretation:**
- CD45: Pan-leukocyte marker (should be positive in all immune cells)
- CD16: Fcγ receptor (neutrophils, NK cells, activated monocytes)
- CD123/HLA-DR/CD14: Monocyte/dendritic cell markers
- CD3/CD19/CD56: T-cell/B-cell/NK-cell markers (lymphocytes)
- Expression patterns help identify cell type and activation state
    """)
    print("="*80)


if __name__ == "__main__":
    main()

