"""
Analyze and visualize the distribution of enemies, obstacles, and gaps
across all patches in the dataset. Saves separate PNGs for each distribution.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from data.parser import LevelParser
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator

# Load data
patches = np.load('./output/processed/patches.npy')
with open('./output/processed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

parser = LevelParser()
evaluator = PatchDifficultyEvaluator(parser)

# Collect counts
enemy_counts = []
obstacle_counts = []
gap_counts = []

for patch in patches:
    result = evaluator.evaluate_patch(patch)
    enemy_counts.append(result['counts']['enemies'])
    obstacle_counts.append(result['counts']['obstacles'])
    gap_counts.append(result['counts']['gaps'])

# Enemy distribution
enemy_counter = Counter(enemy_counts)
max_enemies = max(enemy_counts) if enemy_counts else 0
x_enemies = list(range(max_enemies + 1))
y_enemies = [enemy_counter.get(i, 0) for i in x_enemies]

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(x_enemies, y_enemies, color='#e74c3c', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Number of Enemies', fontsize=12)
ax1.set_ylabel('Number of Patches', fontsize=12)
ax1.set_title(f'Enemy Distribution (Total: {len(patches)} patches)', fontsize=14, fontweight='bold')
ax1.set_xticks(x_enemies)
plt.tight_layout()
plt.savefig('output/visualizations/enemy_distribution.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: output/visualizations/enemy_distribution.png")
plt.close()

# Obstacle distribution
obstacle_counter = Counter(obstacle_counts)
max_obstacles = max(obstacle_counts) if obstacle_counts else 0
x_obstacles = list(range(max_obstacles + 1))
y_obstacles = [obstacle_counter.get(i, 0) for i in x_obstacles]

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(x_obstacles, y_obstacles, color='#3498db', edgecolor='black', alpha=0.8)
ax2.set_xlabel('Number of Obstacles', fontsize=12)
ax2.set_ylabel('Number of Patches', fontsize=12)
ax2.set_title('Obstacle Distribution (cannons, pipes, platforms)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_obstacles)
plt.tight_layout()
plt.savefig('output/visualizations/obstacle_distribution.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: output/visualizations/obstacle_distribution.png")
plt.close()

gap_counter = Counter(gap_counts)
max_gaps = max(gap_counts) if gap_counts else 0
x_gaps = list(range(max_gaps + 1))
y_gaps = [gap_counter.get(i, 0) for i in x_gaps]

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(x_gaps, y_gaps, color='#2ecc71', edgecolor='black', alpha=0.8)
ax3.set_xlabel('Number of Gaps', fontsize=12)
ax3.set_ylabel('Number of Patches', fontsize=12)
ax3.set_title('Gap Distribution (holes in ground)', fontsize=14, fontweight='bold')
ax3.set_xticks(x_gaps)
plt.tight_layout()
plt.savefig('output/visualizations/gap_distribution.png', dpi=150, bbox_inches='tight')
print(f"[OK] Saved: output/visualizations/gap_distribution.png")
plt.close()

# Print summary
print("\n" + "=" * 60)
print("DISTRIBUTION SUMMARY")
print("=" * 60)

print(f"\nEnemies:")
for count, num_patches in sorted(enemy_counter.items()):
    print(f"  {count} enemies: {num_patches} patches ({num_patches/len(patches)*100:.1f}%)")

print(f"\nObstacles:")
for count, num_patches in sorted(obstacle_counter.items()):
    print(f"  {count} obstacles: {num_patches} patches ({num_patches/len(patches)*100:.1f}%)")

print(f"\nGaps:")
for count, num_patches in sorted(gap_counter.items()):
    print(f"  {count} gaps: {num_patches} patches ({num_patches/len(patches)*100:.1f}%)")
