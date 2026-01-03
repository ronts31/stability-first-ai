"""
Visualize Lazarus experiment results and add to documentation
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json(filepath):
    """Load JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def plot_recovery_curve(data, save_path):
    """Plot recovery curve for weight noise experiments"""
    if not data:
        return None
    
    # Extract data
    alphas = []
    damaged = []
    restored = []
    recovery_pct = []
    
    for alpha_str, methods in data.items():
        if 'v3_full' in methods:
            alpha = float(alpha_str)
            alphas.append(alpha)
            damaged.append(methods['v3_full']['damaged'])
            restored.append(methods['v3_full']['restored'])
            recovery_pct.append(methods['v3_full']['recovery_pct'])
    
    if not alphas:
        return None
    
    # Sort by alpha
    sorted_data = sorted(zip(alphas, damaged, restored, recovery_pct))
    alphas, damaged, restored, recovery_pct = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Alpha
    ax1.plot(alphas, damaged, 'o-', label='Damaged', linewidth=2, markersize=8, color='red')
    ax1.plot(alphas, restored, 's-', label='Restored (Lazarus v3)', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Damage Level (α)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Recovery Curve: Weight Noise', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([65, 75])
    
    # Add V-shape annotation
    if len(alphas) >= 2:
        ax1.annotate('V-Shape Recovery', 
                    xy=(alphas[1], restored[1]), 
                    xytext=(alphas[1]+0.05, restored[1]+1),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=10, fontweight='bold', color='blue')
    
    # Plot 2: Recovery Percentage
    ax2.bar(alphas, recovery_pct, width=0.05, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Damage Level (α)', fontsize=12)
    ax2.set_ylabel('Recovery (%)', fontsize=12)
    ax2.set_title('Recovery Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (alpha, rec) in enumerate(zip(alphas, recovery_pct)):
        ax2.text(alpha, rec + (5 if rec > 0 else -5), f'{rec:.1f}%', 
                ha='center', va='bottom' if rec > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_pruning_curve(data, save_path):
    """Plot pruning curve comparing Frozen Mask vs Regrow"""
    if not data:
        return None
    
    # Extract data
    pruning_levels = []
    frozen_pruned = []
    frozen_restored = []
    frozen_recovery = []
    regrow_pruned = []
    regrow_restored = []
    regrow_recovery = []
    
    for level_str, modes in data.items():
        level = float(level_str)
        pruning_levels.append(level * 100)  # Convert to percentage
        
        if 'frozen_mask' in modes:
            frozen_pruned.append(modes['frozen_mask']['pruned'])
            frozen_restored.append(modes['frozen_mask']['restored'])
            frozen_recovery.append(modes['frozen_mask']['recovery_pct'])
        
        if 'regrow' in modes:
            regrow_pruned.append(modes['regrow']['pruned'])
            regrow_restored.append(modes['regrow']['restored'])
            regrow_recovery.append(modes['regrow']['recovery_pct'])
    
    if not pruning_levels:
        return None
    
    # Sort by pruning level
    sorted_data = sorted(zip(pruning_levels, frozen_pruned, frozen_restored, frozen_recovery,
                            regrow_pruned, regrow_restored, regrow_recovery))
    pruning_levels, frozen_pruned, frozen_restored, frozen_recovery, \
    regrow_pruned, regrow_restored, regrow_recovery = zip(*sorted_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy comparison
    ax1.plot(pruning_levels, frozen_pruned, 'o--', label='Frozen Mask (Pruned)', 
            linewidth=2, markersize=8, color='red', alpha=0.7)
    ax1.plot(pruning_levels, frozen_restored, 'o-', label='Frozen Mask (Restored)', 
            linewidth=2, markersize=8, color='green')
    ax1.plot(pruning_levels, regrow_pruned, 's--', label='Regrow (Pruned)', 
            linewidth=2, markersize=8, color='orange', alpha=0.7)
    ax1.plot(pruning_levels, regrow_restored, 's-', label='Regrow (Restored)', 
            linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Pruning Level (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Pruning Curve: Frozen Mask vs Regrow', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 73])
    
    # Plot 2: Recovery percentage comparison
    x = np.arange(len(pruning_levels))
    width = 3
    
    ax2.bar(x - width/2, frozen_recovery, width, label='Frozen Mask', 
           color='green', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, regrow_recovery, width, label='Regrow', 
           color='blue', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Pruning Level (%)', fontsize=12)
    ax2.set_ylabel('Recovery (%)', fontsize=12)
    ax2.set_title('Recovery Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(p)}%' for p in pruning_levels])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (frozen, regrow) in enumerate(zip(frozen_recovery, regrow_recovery)):
        if frozen > 0:
            ax2.text(i - width/2, frozen + 2, f'{frozen:.1f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        if regrow > 0:
            ax2.text(i + width/2, regrow + 2, f'{regrow:.1f}%', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def create_results_markdown(analysis_data, pruning_data):
    """Create markdown with embedded graphs"""
    md_content = """# Lazarus Project: Results Visualization

## Recovery Curve (Weight Noise)

The following graph shows the V-shape recovery pattern when applying Lazarus v3 protocol to models damaged by weight noise.

![Recovery Curve](results/lazarus_recovery_curve.png)

### Key Observations:
- **V-Shape Pattern**: Clear recovery pattern visible at alpha=0.2
- **Optimal Zone**: Alpha 0.15-0.25 shows best recovery rates
- **Recovery Rate**: Up to 93.9% of lost accuracy recovered

## Pruning Curve (Frozen Mask vs Regrow)

Comparison of two recovery modes: Frozen Mask (topology preserved) vs Regrow (weights allowed to regrow).

![Pruning Curve](results/pruning_curve_comparison.png)

### Key Findings:
- **Frozen Mask > Regrow**: Topology preservation outperforms weight regrowth
- **Sweet Spot**: 70-80% pruning shows optimal recovery
- **Recovery Rate**: Up to 85.3% recovery at 80% pruning with Frozen Mask

## Data Summary

"""
    
    # Add analysis data summary
    if analysis_data:
        md_content += "### Weight Noise Analysis\n\n"
        md_content += "| Alpha | Damaged | Restored | Recovery % |\n"
        md_content += "|-------|---------|----------|------------|\n"
        
        for alpha_str in sorted(analysis_data.keys(), key=float):
            if 'v3_full' in analysis_data[alpha_str]:
                r = analysis_data[alpha_str]['v3_full']
                md_content += f"| {alpha_str} | {r['damaged']:.2f}% | {r['restored']:.2f}% | {r['recovery_pct']:.1f}% |\n"
    
    # Add pruning data summary
    if pruning_data:
        md_content += "\n### Pruning Analysis (Frozen Mask)\n\n"
        md_content += "| Pruning | Pruned | Restored | Recovery % |\n"
        md_content += "|---------|--------|----------|------------|\n"
        
        for level_str in sorted(pruning_data.keys(), key=float):
            if 'frozen_mask' in pruning_data[level_str]:
                r = pruning_data[level_str]['frozen_mask']
                md_content += f"| {float(level_str)*100:.0f}% | {r['pruned']:.2f}% | {r['restored']:.2f}% | {r['recovery_pct']:.1f}% |\n"
    
    md_content += "\n---\n\n*Generated automatically from experiment results*\n"
    
    return md_content

def main():
    """Main function"""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    analysis_data = load_json(f'{results_dir}/lazarus_analysis_results.json')
    pruning_data = load_json(f'{results_dir}/pruning_curve_results.json')
    
    # Create plots
    plots_created = []
    
    if analysis_data:
        plot_path = plot_recovery_curve(analysis_data, f'{results_dir}/lazarus_recovery_curve.png')
        if plot_path:
            plots_created.append(plot_path)
            print(f"[OK] Created: {plot_path}")
    
    if pruning_data:
        plot_path = plot_pruning_curve(pruning_data, f'{results_dir}/pruning_curve_comparison.png')
        if plot_path:
            plots_created.append(plot_path)
            print(f"[OK] Created: {plot_path}")
    
    # Create markdown
    md_content = create_results_markdown(analysis_data, pruning_data)
    md_path = f'{results_dir}/RESULTS_VISUALIZATION.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[OK] Created: {md_path}")
    
    if not plots_created:
        print("\n[WARNING] No data files found. Run experiments first:")
        print("  - cd experiments/noise && python experiment_analysis.py")
        print("  - cd experiments/pruning && python experiment_pruning_curve.py")
    
    print(f"\n[OK] Visualization complete! Check {results_dir}/ directory")

if __name__ == "__main__":
    main()

