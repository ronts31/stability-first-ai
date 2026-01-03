"""
Create recovery curve plot from manifest data
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from LAZARUS_FINAL_MANIFESTO.md
alphas = [0.1, 0.2, 0.3]
damaged = [72.78, 71.06, 69.84]
restored = [72.98, 72.96, 72.57]
recovery_pct = [51.3, 90.0, 82.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Alpha
ax1.plot(alphas, damaged, 'o-', label='Damaged', linewidth=2, markersize=10, color='red')
ax1.plot(alphas, restored, 's-', label='Restored (Lazarus v3)', linewidth=2, markersize=10, color='green')
ax1.set_xlabel('Damage Level (α)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Recovery Curve: Weight Noise', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([68, 74])

# Add V-shape annotation
ax1.annotate('V-Shape Recovery\n(Optimal Zone)', 
            xy=(alphas[1], restored[1]), 
            xytext=(alphas[1]+0.05, restored[1]+0.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11, fontweight='bold', color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# Add value labels
for i, (alpha, d, r) in enumerate(zip(alphas, damaged, restored)):
    ax1.text(alpha, d - 0.3, f'{d:.2f}%', ha='center', va='top', fontsize=9, color='red')
    ax1.text(alpha, r + 0.3, f'{r:.2f}%', ha='center', va='bottom', fontsize=9, color='green')

# Plot 2: Recovery Percentage
bars = ax2.bar(alphas, recovery_pct, width=0.05, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Damage Level (α)', fontsize=12)
ax2.set_ylabel('Recovery (%)', fontsize=12)
ax2.set_title('Recovery Rate', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 100])

# Add value labels on bars
for i, (alpha, rec) in enumerate(zip(alphas, recovery_pct)):
    ax2.text(alpha, rec + 3, f'{rec:.1f}%', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight optimal zone
ax2.axvspan(0.15, 0.25, alpha=0.2, color='green', label='Optimal Zone')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('results/lazarus_recovery_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Created: results/lazarus_recovery_curve.png")

