"""
Dataset Balancing Script
Modifies existing patches IN-PLACE to balance difficulty classes.
Does NOT increase total dataset size.
"""
from data.parser import LevelParser
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
import numpy as np
import pickle
import random

# Initialize
parser = LevelParser()
evaluator = PatchDifficultyEvaluator(parser)

# Tile indices
EMPTY = parser.tile_to_idx['-']
GROUND = parser.tile_to_idx['X']
ENEMY = parser.tile_to_idx['E']
BREAKABLE = parser.tile_to_idx['S']

def count_enemies(patch):
    return np.sum(patch == ENEMY)

def get_difficulty_class(enemies):
    if enemies >= 4:
        return 'hard'
    elif enemies >= 1:
        return 'medium'
    return 'easy'

def find_valid_enemy_positions(patch):
    """Find positions where enemies can be placed."""
    height, width = patch.shape
    valid_positions = []
    ground_tiles = {GROUND, BREAKABLE}
    
    for row in range(1, height):
        for col in range(2, width - 2):
            if patch[row - 1, col] == EMPTY and patch[row, col] in ground_tiles:
                if patch[row - 1, col] != ENEMY:
                    valid_positions.append((row - 1, col))
    
    return valid_positions

def add_enemies_to_patch(patch, num_to_add):
    """Add enemies to a patch, modifies in place and returns success."""
    valid_positions = find_valid_enemy_positions(patch)
    
    if len(valid_positions) < num_to_add:
        return False
    
    selected = random.sample(valid_positions, num_to_add)
    for row, col in selected:
        patch[row, col] = ENEMY
    
    return True

def balance_dataset():
    # Run prepare_data first to get fresh patches
    print("=" * 60)
    print("LOADING FRESH DATA")
    print("=" * 60)
    
    import subprocess
    subprocess.run(['python', '-m', 'scripts.prepare_data'], check=True)
    
    # Load data
    patches = np.load('./output/processed/patches.npy')
    with open('./output/processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    total = len(patches)
    target_per_class = total // 3  # ~479 per class
    
    print(f"\nTotal patches: {total}")
    print(f"Target per class: {target_per_class}")
    
    print("\n" + "=" * 60)
    print("ORIGINAL DISTRIBUTION")
    print("=" * 60)
    
    # Classify and index patches
    easy_indices = []
    medium_indices = []
    hard_indices = []
    
    for i, patch in enumerate(patches):
        enemies = count_enemies(patch)
        difficulty = get_difficulty_class(enemies)
        
        if difficulty == 'easy':
            easy_indices.append(i)
        elif difficulty == 'medium':
            medium_indices.append(i)
        else:
            hard_indices.append(i)
    
    print(f"Easy   (0   enemies): {len(easy_indices):4d} ({len(easy_indices)/total*100:5.1f}%)")
    print(f"Medium (1-2 enemies): {len(medium_indices):4d} ({len(medium_indices)/total*100:5.1f}%)")
    print(f"Hard   (3+  enemies): {len(hard_indices):4d} ({len(hard_indices)/total*100:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("BALANCING DATASET (IN-PLACE MODIFICATION)")
    print("=" * 60)
    
    # Shuffle to randomize which patches get modified
    random.shuffle(easy_indices)
    random.shuffle(medium_indices)
    
    # Calculate how many to convert
    hard_needed = target_per_class - len(hard_indices)
    medium_needed = target_per_class - len(medium_indices)
    
    print(f"\nNeed {hard_needed} more hard patches")
    print(f"Need {medium_needed} more medium patches")
    
    modified_count = 0
    
    # Convert some MEDIUM -> HARD (add enemies: 1-2 -> 3+)
    hard_from_medium = min(hard_needed, len(medium_indices))
    print(f"\nConverting up to {hard_from_medium} medium -> hard...")
    
    converted_to_hard = 0
    medium_to_convert = medium_indices[:hard_from_medium]
    
    for idx in medium_to_convert:
        patch = patches[idx]
        enemies = count_enemies(patch)
        target_enemies = random.randint(3, 5)  # Vary hard patches from 3-5 enemies
        to_add = max(1, target_enemies - enemies)
        
        if add_enemies_to_patch(patch, to_add):
            metadata[idx]['modified'] = True
            metadata[idx]['added_enemies'] = to_add
            converted_to_hard += 1
            modified_count += 1
    
    print(f"Converted {converted_to_hard} medium -> hard")
    hard_needed -= converted_to_hard
    
    # Convert some EASY -> HARD (add 3 enemies)
    if hard_needed > 0:
        print(f"\nConverting up to {hard_needed} easy -> hard...")
        easy_to_hard = 0
        
        for idx in easy_indices:
            if easy_to_hard >= hard_needed:
                break
            
            patch = patches[idx]
            enemies = count_enemies(patch)
            target_enemies = random.randint(3, 5)  # Vary hard patches from 3-5 enemies
            to_add = target_enemies - enemies
            
            if add_enemies_to_patch(patch, to_add):
                metadata[idx]['modified'] = True
                metadata[idx]['added_enemies'] = to_add
                easy_to_hard += 1
                modified_count += 1
                # Remove from easy_indices so we don't use it for medium
                easy_indices.remove(idx)
        
        print(f"Converted {easy_to_hard} easy -> hard")
    
    # Recalculate medium needed after hard conversions
    # Recount current distribution
    current_medium = 0
    for i in range(len(patches)):
        enemies = count_enemies(patches[i])
        if 1 <= enemies <= 2:
            current_medium += 1
    
    medium_needed = target_per_class - current_medium
    
    # Convert some EASY -> MEDIUM (add 1-2 enemies)
    if medium_needed > 0:
        print(f"\nConverting up to {medium_needed} easy -> medium...")
        easy_to_medium = 0
        
        for idx in easy_indices:
            if easy_to_medium >= medium_needed:
                break
            
            patch = patches[idx]
            enemies = count_enemies(patch)
            to_add = random.randint(1, 2)  # Add 1-2 enemies for medium
            
            if add_enemies_to_patch(patch, to_add):
                # Verify it's actually medium now (1-2 enemies)
                final_enemies = count_enemies(patch)
                if 1 <= final_enemies <= 2:
                    metadata[idx]['modified'] = True
                    metadata[idx]['added_enemies'] = to_add
                    easy_to_medium += 1
                    modified_count += 1
        
        print(f"Converted {easy_to_medium} easy -> medium")
    
    # Recalculate final distribution
    print("\n" + "=" * 60)
    print("FINAL DISTRIBUTION")
    print("=" * 60)
    
    easy_count = medium_count = hard_count = 0
    for patch in patches:
        enemies = count_enemies(patch)
        difficulty = get_difficulty_class(enemies)
        if difficulty == 'easy':
            easy_count += 1
        elif difficulty == 'medium':
            medium_count += 1
        else:
            hard_count += 1
    
    print(f"Easy   (0   enemies): {easy_count:4d} ({easy_count/total*100:5.1f}%)")
    print(f"Medium (1-2 enemies): {medium_count:4d} ({medium_count/total*100:5.1f}%)")
    print(f"Hard   (3+  enemies): {hard_count:4d} ({hard_count/total*100:5.1f}%)")
    print(f"\nTotal patches: {total} (unchanged)")
    print(f"Modified patches: {modified_count}")
    
    # Update difficulty scores in metadata
    for i, patch in enumerate(patches):
        enemies = count_enemies(patch)
        if enemies >= 3:
            metadata[i]['final_score'] = 1.0
        elif enemies >= 1:
            metadata[i]['final_score'] = 0.5
        else:
            metadata[i]['final_score'] = 0.0
    
    # Save
    np.save('./output/processed/patches.npy', patches)
    with open('./output/processed/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n[OK] Balanced dataset saved to output/processed/")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE MODIFIED PATCHES")
    print("=" * 60)
    
    modified_indices = [i for i, m in enumerate(metadata) if m.get('modified', False)]
    if modified_indices:
        samples = random.sample(modified_indices, min(3, len(modified_indices)))
        for idx in samples:
            patch = patches[idx]
            meta = metadata[idx]
            enemies = count_enemies(patch)
            print(f"\n--- Modified: added {meta.get('added_enemies', '?')} enemies, now {enemies} total ---")
            print(parser.decode_level(patch))

if __name__ == "__main__":
    balance_dataset()
