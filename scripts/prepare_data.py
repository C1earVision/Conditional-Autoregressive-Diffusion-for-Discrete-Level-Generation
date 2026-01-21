from data.parser import LevelParser
from data.extractor import PatchExtractor
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
import numpy as np
import pickle
import random

import glob

parser = LevelParser()
extractor = PatchExtractor()
patch_evaluator = PatchDifficultyEvaluator(parser)

rawData = []
for filepath in glob.glob("./dataset/*.txt"):
    with open(filepath, "r") as file:
        content = [line.strip("\n") for line in file.readlines()]
        rawData.append(content)

parsed_dataset = parser.parse_dataset(
    levels=rawData,
    level_names=[f"map_{i+1}" for i in range(1000)]
)


patches, metadata = extractor.extract_patches_from_dataset(
parsed_dataset
)


patch_evaluation_results = patch_evaluator.evaluate_patches_batch(patches, metadata)
for i, result in enumerate(patch_evaluation_results):
  metadata[i]['final_score'] = result['scores']['difficulty_score']

np.save('./output/processed/patches.npy', patches)
with open('./output/processed/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

scores = np.array([m['final_score'] for m in metadata])
print("\n" + "="*60)
print("DIFFICULTY DISTRIBUTION (Discrete Classes)")
print("="*60)
easy_count = np.sum(scores == 0.0)
medium_count = np.sum(scores == 0.5)
hard_count = np.sum(scores == 1.0)
print(f"Easy   (0.0): {easy_count:4d} ({easy_count/len(scores)*100:5.1f}%)")
print(f"Medium (0.5): {medium_count:4d} ({medium_count/len(scores)*100:5.1f}%)")
print(f"Hard   (1.0): {hard_count:4d} ({hard_count/len(scores)*100:5.1f}%)")

print("\n" + "="*60)
print("EXAMPLE PATCHES FROM RANDOM LEVEL")
print("="*60)

level_names = [m['level_name'] for m in metadata]
unique_levels = list(set(level_names))
random_level = random.choice(unique_levels)

level_indices = [i for i, m in enumerate(metadata) if m['level_name'] == random_level]
print(f"Level: {random_level} ({len(level_indices)} patches)\n")

for idx in level_indices[:5]: 
    patch = patches[idx]
    result = patch_evaluation_results[idx]
    
    print(f"--- Patch {idx} (Difficulty: {result['scores']['difficulty_score']:.3f}) ---")
    print(f"Enemies={result['counts']['enemies']}, Gaps={result['counts']['gaps']}, "
          f"Obstacles={result['counts']['obstacles']}")
    print(parser.decode_level(patch))
    print()