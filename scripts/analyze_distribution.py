"""
Analyze and visualize the distribution of enemies, obstacles, and gaps
across all patches in the dataset.
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

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribution of Level Challenges Across Patches', fontsize=14, fontweight='bold')

# Enemy distribution
enemy_counter = Counter(enemy_counts)
max_enemies = max(enemy_counts) if enemy_counts else 0
x_enemies = list(range(max_enemies + 1))
y_enemies = [enemy_counter.get(i, 0) for i in x_enemies]

axes[0].bar(x_enemies, y_enemies, color='#e74c3c', edgecolor='black', alpha=0.8)
axes[0].set_xlabel('Number of Enemies')
axes[0].set_ylabel('Number of Patches')
axes[0].set_title(f'Enemy Distribution\n(Total patches: {len(patches)})')
axes[0].set_xticks(x_enemies)

# Obstacle distribution
obstacle_counter = Counter(obstacle_counts)
max_obstacles = max(obstacle_counts) if obstacle_counts else 0
x_obstacles = list(range(max_obstacles + 1))
y_obstacles = [obstacle_counter.get(i, 0) for i in x_obstacles]

axes[1].bar(x_obstacles, y_obstacles, color='#3498db', edgecolor='black', alpha=0.8)
axes[1].set_xlabel('Number of Obstacles')
axes[1].set_ylabel('Number of Patches')
axes[1].set_title(f'Obstacle Distribution\n(cannons, pipes, platforms)')

# Gap distribution
gap_counter = Counter(gap_counts)
max_gaps = max(gap_counts) if gap_counts else 0
x_gaps = list(range(max_gaps + 1))
y_gaps = [gap_counter.get(i, 0) for i in x_gaps]

axes[2].bar(x_gaps, y_gaps, color='#2ecc71', edgecolor='black', alpha=0.8)
axes[2].set_xlabel('Number of Gaps')
axes[2].set_ylabel('Number of Patches')
axes[2].set_title('Gap Distribution\n(holes in ground)')

plt.tight_layout()
plt.savefig('output/visualizations/challenge_distribution.png', dpi=150, bbox_inches='tight')
print(f"[OK] Chart saved to output/visualizations/challenge_distribution.png")

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

# Show plot
plt.show()
