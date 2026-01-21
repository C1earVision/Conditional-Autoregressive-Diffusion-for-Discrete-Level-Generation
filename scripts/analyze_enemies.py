from data.parser import LevelParser
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import glob
import os

parser = LevelParser()
evaluator = PatchDifficultyEvaluator(parser)

# Load all patches from the processed data
patches = np.load('./output/processed/patches.npy')
all_enemy_counts = []

for patch in patches:
    enemies = evaluator._count_enemies(patch)
    all_enemy_counts.append(enemies)

# Count distribution
counter = Counter(all_enemy_counts)
enemy_nums = sorted(counter.keys())
counts = [counter[e] for e in enemy_nums]

print('Enemy Count Distribution:')
print('-' * 30)
for e, c in zip(enemy_nums, counts):
    pct = c / len(all_enemy_counts) * 100
    print(f'{e} enemies: {c:4d} patches ({pct:5.1f}%)')
print('-' * 30)
print(f'Total patches: {len(all_enemy_counts)}')

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(enemy_nums, counts, color='steelblue', edgecolor='black')
plt.xlabel('Number of Enemies', fontsize=12)
plt.ylabel('Number of Patches', fontsize=12)
plt.title('Distribution of Enemy Count per Patch', fontsize=14)
plt.xticks(enemy_nums)

# Add value labels on bars
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig('output/enemy_distribution.png', dpi=150)
print(f'\nChart saved to output/enemy_distribution.png')
