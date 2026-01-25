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
QUESTION = parser.tile_to_idx['?']
QUESTION_EMPTY = parser.tile_to_idx['Q']
PIPE_TOP_L = parser.tile_to_idx['<']
PIPE_TOP_R = parser.tile_to_idx['>']
PIPE_L = parser.tile_to_idx['[']
PIPE_R = parser.tile_to_idx[']']
CANNON_TOP = parser.tile_to_idx['B']

# All tiles that enemies can stand on
STANDABLE_TILES = {
    GROUND, BREAKABLE, QUESTION, QUESTION_EMPTY,
    PIPE_TOP_L, PIPE_TOP_R, PIPE_L, PIPE_R, CANNON_TOP
}

def count_enemies(patch):
    return np.sum(patch == ENEMY)

def get_difficulty_class(enemies):
    if enemies >= 4:
        return 'hard'
    elif enemies >= 2:
        return 'medium'
    return 'easy'

def find_connected_platform(patch, start_row, start_col, visited):
    """Find all tiles connected to a starting standable tile using flood-fill.
    
    Returns a set of (row, col) positions that form a connected platform.
    """
    height, width = patch.shape
    platform = set()
    stack = [(start_row, start_col)]
    
    while stack:
        row, col = stack.pop()
        if (row, col) in visited:
            continue
        if row < 0 or row >= height or col < 0 or col >= width:
            continue
        if patch[row, col] not in STANDABLE_TILES:
            continue
        
        visited.add((row, col))
        platform.add((row, col))
        
        # Check horizontal neighbors only (platforms are typically horizontal)
        stack.append((row, col + 1))
        stack.append((row, col - 1))
        # Also check same-row adjacent for diagonal platforms
        stack.append((row + 1, col))
        stack.append((row - 1, col))
    
    return platform

def identify_platforms(patch):
    """Identify all distinct platforms in the patch.
    
    Returns:
        List of platforms, where each platform is a set of (row, col) positions
    """
    height, width = patch.shape
    visited = set()
    platforms = []
    
    for row in range(height):
        for col in range(width):
            if patch[row, col] in STANDABLE_TILES and (row, col) not in visited:
                platform = find_connected_platform(patch, row, col, visited)
                if platform:
                    platforms.append(platform)
    
    return platforms

def get_enemy_positions_by_platform(patch):
    """Get valid enemy positions grouped by platform.
    
    Returns:
        Dict mapping platform_id to list of valid (enemy_row, enemy_col) positions
        Also returns count of existing enemies per platform
    """
    height, width = patch.shape
    platforms = identify_platforms(patch)
    
    # Create mapping from tile position to platform index
    tile_to_platform = {}
    for idx, platform in enumerate(platforms):
        for pos in platform:
            tile_to_platform[pos] = idx
    
    # Find valid enemy positions and group by platform
    positions_by_platform = {i: [] for i in range(len(platforms))}
    enemies_by_platform = {i: 0 for i in range(len(platforms))}
    
    for row in range(1, height):
        for col in range(2, width - 2):
            standing_tile = (row, col)
            if standing_tile in tile_to_platform:
                platform_id = tile_to_platform[standing_tile]
                enemy_pos = (row - 1, col)
                
                # Check if position is empty or has enemy
                if patch[enemy_pos[0], enemy_pos[1]] == EMPTY:
                    positions_by_platform[platform_id].append(enemy_pos)
                elif patch[enemy_pos[0], enemy_pos[1]] == ENEMY:
                    enemies_by_platform[platform_id] += 1
    
    return positions_by_platform, enemies_by_platform, platforms

def add_enemies_varied(patch, num_to_add, min_spacing=2, max_per_platform=2):
    """Add enemies spread across platforms, limiting each platform to max_per_platform enemies.
    
    Returns False if we can't place enough enemies without exceeding platform limits.
    """
    positions_by_platform, enemies_by_platform, platforms = get_enemy_positions_by_platform(patch)
    
    # Calculate available slots per platform
    available_by_platform = {}
    total_available = 0
    for platform_id in positions_by_platform:
        existing = enemies_by_platform[platform_id]
        slots = max(0, max_per_platform - existing)
        available_positions = min(len(positions_by_platform[platform_id]), slots)
        available_by_platform[platform_id] = available_positions
        total_available += available_positions
    
    if total_available < num_to_add:
        return False  # Skip this patch - can't meet target
    
    def is_far_enough(pos, selected_positions, min_dist):
        """Check if position is at least min_dist columns away from all selected."""
        row, col = pos
        for sel_row, sel_col in selected_positions:
            if abs(col - sel_col) < min_dist:
                return False
        return True
    
    # Collect all positions with their platform IDs
    all_candidates = []
    for platform_id, positions in positions_by_platform.items():
        for pos in positions:
            all_candidates.append((pos, platform_id))
    
    random.shuffle(all_candidates)
    
    selected = []
    used_by_platform = {i: 0 for i in range(len(platforms))}
    
    # First pass: with spacing constraint
    for pos, platform_id in all_candidates:
        if len(selected) >= num_to_add:
            break
        if used_by_platform[platform_id] >= available_by_platform[platform_id]:
            continue
        if is_far_enough(pos, selected, min_spacing):
            selected.append(pos)
            used_by_platform[platform_id] += 1
    
    # Second pass: relax spacing to 1
    if len(selected) < num_to_add:
        for pos, platform_id in all_candidates:
            if len(selected) >= num_to_add:
                break
            if used_by_platform[platform_id] >= available_by_platform[platform_id]:
                continue
            if pos not in selected and is_far_enough(pos, selected, 1):
                selected.append(pos)
                used_by_platform[platform_id] += 1
    
    # Third pass: no spacing constraint
    if len(selected) < num_to_add:
        for pos, platform_id in all_candidates:
            if len(selected) >= num_to_add:
                break
            if used_by_platform[platform_id] >= available_by_platform[platform_id]:
                continue
            if pos not in selected:
                selected.append(pos)
                used_by_platform[platform_id] += 1
    
    if len(selected) < num_to_add:
        return False  # Couldn't place enough enemies
    
    for row, col in selected:
        patch[row, col] = ENEMY
    
    return True

def add_elevated_platform(patch):
    """Add a random elevated platform to a flat ground patch.
    
    Returns True if a platform was added successfully.
    """
    height, width = patch.shape
    
    # Platform types: ground blocks, breakable blocks, or question blocks
    platform_types = [
        GROUND,      # Solid ground platform
        BREAKABLE,   # Breakable brick platform
        QUESTION,    # Question block platform
    ]
    
    # Choose random platform parameters
    platform_type = random.choice(platform_types)
    platform_width = random.randint(3, 6)  # 3-6 tiles wide
    platform_row = random.randint(5, 10)   # Row 5-10 (above ground, not too high)
    start_col = random.randint(2, width - platform_width - 2)  # Leave margins
    
    # Check if placement area is clear
    for col in range(start_col, start_col + platform_width):
        if patch[platform_row, col] != EMPTY:
            return False  # Space not clear
        # Also check row above to ensure headroom
        if platform_row > 0 and patch[platform_row - 1, col] != EMPTY:
            return False
    
    # Place the platform
    for col in range(start_col, start_col + platform_width):
        patch[platform_row, col] = platform_type
    
    return True

def add_multiple_platforms(patch, num_platforms=None):
    """Add 1-3 elevated platforms to a patch.
    
    Returns True if at least one platform was added.
    """
    if num_platforms is None:
        num_platforms = random.randint(1, 3)
    
    added = 0
    attempts = 0
    max_attempts = num_platforms * 5  # Try multiple times per platform
    
    while added < num_platforms and attempts < max_attempts:
        if add_elevated_platform(patch):
            added += 1
        attempts += 1
    
    return added > 0

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
    
    def has_interesting_features(patch):
        result = evaluator.evaluate_patch(patch)
        obstacles = result['counts']['obstacles']
        gaps = result['counts']['gaps']
        
        # Check for elevated platforms (non-ground tiles above the ground row)
        height, width = patch.shape
        has_platforms = False
        for row in range(height - 2):  # Exclude bottom 2 rows
            for col in range(width):
                if patch[row, col] == GROUND or patch[row, col] == BREAKABLE:
                    has_platforms = True
                    break
            if has_platforms:
                break
        
        # Must have at least one interesting feature
        return obstacles > 0 or gaps > 0 or has_platforms
    
    total = len(patches)
    target_per_class = total // 3  # ~479 per class
    
    print(f"\nTotal patches: {total}")
    print(f"Target per class: {target_per_class}")
    
    print("\n" + "=" * 60)
    print("ORIGINAL DISTRIBUTION")
    print("=" * 60)
    
    # Classify and index patches - separate by features
    easy_with_features = []
    easy_without_features = []
    medium_indices = []
    hard_indices = []
    
    for i, patch in enumerate(patches):
        enemies = count_enemies(patch)
        difficulty = get_difficulty_class(enemies)
        
        if difficulty == 'easy':
            if has_interesting_features(patch):
                easy_with_features.append(i)
            else:
                easy_without_features.append(i)
        elif difficulty == 'medium':
            medium_indices.append(i)
        else:
            hard_indices.append(i)
    
    easy_indices = easy_with_features + easy_without_features
    print(f"Easy   (0-1 enemies): {len(easy_indices):4d} ({len(easy_indices)/total*100:5.1f}%)")
    print(f"  - With features: {len(easy_with_features)}")
    print(f"  - Flat/empty: {len(easy_without_features)}")
    
    print(f"Easy   (0-1 enemies): {len(easy_indices):4d} ({len(easy_indices)/total*100:5.1f}%)")
    print(f"Medium (2-3 enemies): {len(medium_indices):4d} ({len(medium_indices)/total*100:5.1f}%)")
    print(f"Hard   (4+  enemies): {len(hard_indices):4d} ({len(hard_indices)/total*100:5.1f}%)")
    
    # =================================================================
    # ADD PLATFORMS TO PATCHES WITHOUT ELEVATED PLATFORMS (80% of them)
    # =================================================================
    print("\n" + "=" * 60)
    print("ADDING PLATFORMS TO FLAT PATCHES (ALL DIFFICULTIES)")
    print("=" * 60)
    
    def has_elevated_platforms(patch):
        """Check if patch has any elevated platforms (tiles above ground floor)."""
        height, width = patch.shape
        for row in range(height - 2):  # Exclude bottom 2 rows (ground floor)
            for col in range(width):
                if patch[row, col] in {GROUND, BREAKABLE, QUESTION, QUESTION_EMPTY}:
                    return True
        return False
    
    # Find ALL patches without elevated platforms (any difficulty)
    all_indices = list(range(total))
    flat_patches = [i for i in all_indices if not has_elevated_platforms(patches[i])]
    
    print(f"Found {len(flat_patches)} patches without elevated platforms")
    
    flat_patches_to_modify = int(len(flat_patches) * 0.8)
    random.shuffle(flat_patches)
    platforms_added = 0
    platform_modified_indices = []  # Track which patches got platforms
    
    for idx in flat_patches[:flat_patches_to_modify]:
        patch = patches[idx]
        if add_multiple_platforms(patch):
            metadata[idx]['added_platforms'] = True
            platforms_added += 1
            platform_modified_indices.append(idx)
            # Update feature tracking for easy patches
            if idx in easy_without_features:
                easy_with_features.append(idx)
    
    # Update easy_without_features list
    easy_without_features = [i for i in easy_without_features if i not in easy_with_features]
    print(f"Added platforms to {platforms_added}/{flat_patches_to_modify} flat patches")
    print(f"Easy patches now: {len(easy_with_features)} with features, {len(easy_without_features)} flat")
    
    # Show some patches that had platforms added
    if platform_modified_indices:
        print("\n--- Sample patches with added platforms ---")
        samples = random.sample(platform_modified_indices, min(5, len(platform_modified_indices)))
        for idx in samples:
            patch = patches[idx]
            enemies = count_enemies(patch)
            print()
            for row in range(patch.shape[0]):
                print(''.join(parser.idx_to_tile[int(t)] for t in patch[row]))
            print(f"(Patch {idx}: {enemies} enemies)")
        print("--- End platform samples ---\n")
    
    print("\n" + "=" * 60)
    print("BALANCING DATASET (IN-PLACE MODIFICATION)")
    print("=" * 60)
    print("Note: Only modifying patches WITH interesting features")
    
    # Shuffle - keep feature patches separate
    random.shuffle(easy_with_features)
    random.shuffle(easy_without_features)
    random.shuffle(medium_indices)
    
    # Use only patches with features for conversion
    easy_candidates = easy_with_features.copy()
    
    # Filter medium patches to only include those with interesting features
    medium_with_features = [i for i in medium_indices if has_interesting_features(patches[i])]
    medium_without_features = [i for i in medium_indices if i not in medium_with_features]
    print(f"\nMedium patches breakdown:")
    print(f"  - With features: {len(medium_with_features)}")
    print(f"  - Flat/boring: {len(medium_without_features)}")
    
    # Calculate how many to convert
    hard_needed = target_per_class - len(hard_indices)
    medium_needed = target_per_class - len(medium_indices)
    
    print(f"\nNeed {hard_needed} more hard patches")
    print(f"Need {medium_needed} more medium patches")
    
    modified_count = 0
    
    # Convert some MEDIUM -> HARD (only patches WITH features)
    hard_from_medium = min(hard_needed, len(medium_with_features))
    print(f"\nConverting up to {hard_from_medium} medium -> hard (only with features)...")
    
    converted_to_hard = 0
    medium_to_convert = medium_with_features[:hard_from_medium]
    
    for idx in medium_to_convert:
        patch = patches[idx]
        enemies = count_enemies(patch)
        target_enemies = random.randint(4, 5)  # Hard: 4-5 enemies
        to_add = max(1, target_enemies - enemies)
        
        if add_enemies_varied(patch, to_add):
            metadata[idx]['modified'] = True
            metadata[idx]['added_enemies'] = to_add
            converted_to_hard += 1
            modified_count += 1
    
    print(f"Converted {converted_to_hard} medium -> hard")
    
    # Recalculate medium needed after hard conversions
    # Recount current distribution
    current_medium = 0
    for i in range(len(patches)):
        enemies = count_enemies(patches[i])
        if 2 <= enemies <= 3:  # Medium: 2-3 enemies
            current_medium += 1
    
    medium_needed = target_per_class - current_medium
    
    # Convert some EASY -> MEDIUM (add 2-3 enemies) - ONLY patches with features
    if medium_needed > 0:
        print(f"\nConverting up to {medium_needed} easy -> medium...")
        easy_to_medium = 0
        
        for idx in easy_candidates[:]:
            if easy_to_medium >= medium_needed:
                break
            
            patch = patches[idx]
            enemies = count_enemies(patch)
            to_add = random.randint(2, 3) - enemies  # Target 2-3 enemies for medium
            to_add = max(1, to_add)
            
            if add_enemies_varied(patch, to_add):
                final_enemies = count_enemies(patch)
                if 2 <= final_enemies <= 3:  # Medium: 2-3 enemies
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
    
    print(f"Easy   (0-1 enemies): {easy_count:4d} ({easy_count/total*100:5.1f}%)")
    print(f"Medium (2-3 enemies): {medium_count:4d} ({medium_count/total*100:5.1f}%)")
    print(f"Hard   (4+  enemies): {hard_count:4d} ({hard_count/total*100:5.1f}%)")
    print(f"\nTotal patches: {total}")
    print(f"Modified patches: {modified_count}")
    
    # Update difficulty scores in metadata
    for i, patch in enumerate(patches):
        enemies = count_enemies(patch)
        if enemies >= 4:
            metadata[i]['final_score'] = 1.0
        elif enemies >= 2:
            metadata[i]['final_score'] = 0.5
        else:
            metadata[i]['final_score'] = 0.0
    
    # Save
    np.save('./output/processed/patches.npy', patches)
    with open('./output/processed/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n[OK] Balanced dataset saved to output/processed/")
    
    # Save visualization of augmented enemy distribution
    import matplotlib.pyplot as plt
    from collections import Counter
    
    enemy_counts = [count_enemies(p) for p in patches]
    enemy_counter = Counter(enemy_counts)
    max_enemies = max(enemy_counts) if enemy_counts else 0
    x_enemies = list(range(max_enemies + 1))
    y_enemies = [enemy_counter.get(i, 0) for i in x_enemies]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_enemies, y_enemies, color='#e74c3c', edgecolor='black', alpha=0.8)
    
    # Add class labels
    ax.axvspan(-0.5, 1.5, alpha=0.1, color='green', label='Easy (0-1)')
    ax.axvspan(1.5, 3.5, alpha=0.1, color='orange', label='Medium (2-3)')
    ax.axvspan(3.5, max_enemies + 0.5, alpha=0.1, color='red', label='Hard (4+)')
    
    ax.set_xlabel('Number of Enemies', fontsize=12)
    ax.set_ylabel('Number of Patches', fontsize=12)
    ax.set_title(f'Enemy Distribution After Augmentation (Total: {total} patches)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_enemies)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('output/visualizations/enemy_distribution_augmented.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/visualizations/enemy_distribution_augmented.png")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE MODIFIED PATCHES")
    print("=" * 60)
    
    modified_indices = [i for i, m in enumerate(metadata) if m.get('modified', False)]
    if modified_indices:
        samples = random.sample(modified_indices, min(5, len(modified_indices)))
        for idx in samples:
            patch = patches[idx]
            meta = metadata[idx]
            enemies = count_enemies(patch)
            print(f"\n--- Modified: added {meta.get('added_enemies', '?')} enemies, now {enemies} total ---")
            print(parser.decode_level(patch))

if __name__ == "__main__":
    balance_dataset()
