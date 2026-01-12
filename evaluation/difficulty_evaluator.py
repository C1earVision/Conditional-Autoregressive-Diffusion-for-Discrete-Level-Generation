import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict

class PatchDifficultyEvaluator:
    def __init__(self, parser):
        self.parser = parser
        self.EMPTY = parser.tile_to_idx['-']
        self.GROUND = parser.tile_to_idx['X']
        self.BREAKABLE = parser.tile_to_idx['S']
        self.COIN = parser.tile_to_idx['o']
        self.ENEMY = parser.tile_to_idx['E']
        self.PIPE_LEFT = parser.tile_to_idx['[']
        self.PIPE_RIGHT = parser.tile_to_idx[']']
        self.QUESTION = parser.tile_to_idx['Q']
        self.BRICK_LEFT = parser.tile_to_idx['<']
        self.BRICK_RIGHT = parser.tile_to_idx['>']
        self.CANNON_BOTTOM = parser.tile_to_idx['b']
        self.CANNON_TOP = parser.tile_to_idx['B']

        self.MAX_ENEMY_DENSITY = 0.15
        self.MAX_CANNON_DENSITY = 0.15
        self.MAX_PIPE_DENSITY = 0.15
        self.MAX_JUMP_DENSITY = 0.2
        self.MAX_PLATFORM_DENSITY = 0.1

    def _count_enemies(self, patch):
        return np.sum(patch == self.ENEMY)

    def _count_cannons(self, patch):
        height, width = patch.shape
        cannon_count = 0

        for col in range(width):
            has_cannon_bottom = np.any(patch[:, col] == self.CANNON_BOTTOM)
            if has_cannon_bottom:
                cannon_count += 1

        return cannon_count

    def _count_pipes(self, patch):
        height, width = patch.shape
        pipe_count = 0

        for col in range(width):
            has_pipe_left = np.any(patch[:, col] == self.PIPE_LEFT)
            if has_pipe_left:
                pipe_count += 1

        return pipe_count

    def _count_jumps(self, patch):
        height, width = patch.shape
        jumps = 0

        ground_floor = patch[height - 1]
        gap_length = 0

        for col in range(width):
            if ground_floor[col] == self.EMPTY:
                gap_length += 1
            else:
                if gap_length > 0:
                    jumps += 1
                    if gap_length > 4:
                        jumps += (gap_length - 2) * 0.5
                gap_length = 0

        if gap_length > 0:
            jumps += 1
            if gap_length > 2:
                jumps += (gap_length - 2) * 0.5

        return int(jumps)

    def _count_elevated_platforms(self, patch):
            height, width = patch.shape
            platforms = 0

            PLATFORM_TILES = {self.GROUND, self.BREAKABLE, self.QUESTION}

            for row in range(height - 1):
                in_platform = False
                platform_start = -1

                for col in range(width):
                    if patch[row, col] in PLATFORM_TILES:
                        if not in_platform:
                            in_platform = True
                            platform_start = col
                    else:
                        if in_platform:
                            is_floating = False
                            for c in range(platform_start, col):
                                for r in range(row + 1, height):
                                    if patch[r, c] == self.EMPTY:
                                        is_floating = True
                                        break
                                if is_floating:
                                    break

                            if is_floating:
                                platforms += 1

                            in_platform = False

                if in_platform:
                    is_floating = False
                    for c in range(platform_start, width):
                        for r in range(row + 1, height):
                            if patch[r, c] == self.EMPTY:
                                is_floating = True
                                break
                        if is_floating:
                            break

                    if is_floating:
                        platforms += 1

            return platforms

    def _count_obstacles(self, patch):
        """Count obstacle groups (pipes, blocks, cannons as simple count)"""
        height, width = patch.shape
        
        OBSTACLE_TILES = {
            self.PIPE_LEFT, self.PIPE_RIGHT,
            self.CANNON_BOTTOM, self.CANNON_TOP,
            self.QUESTION, self.BREAKABLE
        }
        
        visited = set()
        obstacle_count = 0
        
        for row in range(height):
            for col in range(width):
                if patch[row, col] in OBSTACLE_TILES and (row, col) not in visited:
                    # Found new obstacle group, count it as 1
                    obstacle_count += 1
                    
                    # Mark all connected tiles as visited (flood fill)
                    self._mark_connected(patch, row, col, OBSTACLE_TILES, visited)
        
        return obstacle_count

    def _mark_connected(self, patch, row, col, obstacle_tiles, visited):
        """Mark all horizontally/vertically connected obstacle tiles"""
        if (row, col) in visited:
            return
        if row < 0 or row >= patch.shape[0] or col < 0 or col >= patch.shape[1]:
            return
        if patch[row, col] not in obstacle_tiles:
            return
        
        visited.add((row, col))
        
        # Check 4 adjacent tiles
        self._mark_connected(patch, row+1, col, obstacle_tiles, visited)
        self._mark_connected(patch, row-1, col, obstacle_tiles, visited)
        self._mark_connected(patch, row, col+1, obstacle_tiles, visited)
        self._mark_connected(patch, row, col-1, obstacle_tiles, visited)


    def evaluate_patch(self, patch, metadata=None):
        height, width = patch.shape
        total_tiles = height * width

        enemies = self._count_enemies(patch)
        gaps = self._count_jumps(patch)  
        obstacles = self._count_obstacles(patch)  
        
        cannons = self._count_cannons(patch)
        pipes = self._count_pipes(patch)
        jumps = gaps
        platforms = self._count_elevated_platforms(patch)

        raw_score = (
            enemies * 3.0 +
            gaps * 5.0 +
            obstacles * 1.5
        )
        
        MAX_RAW_SCORE = 15.0  
        diff_score = min(raw_score / MAX_RAW_SCORE, 1.0)


        result = {
            "metadata": metadata,
            "counts": {
                "enemies": enemies,
                "gaps": gaps,
                "obstacles": obstacles,
                "cannons": cannons,
                "pipes": pipes,
                "jumps": jumps,
                "platforms": platforms,
                "total_tiles": total_tiles
            },
            "scores": {
                "difficulty_score": round(diff_score, 3),
            },
        }

        return result

    def evaluate_patches_batch(self, patches, metadata_list=None):
        if metadata_list is None:
            metadata_list = [None] * len(patches)

        results = []
        for i in range(len(patches)):
            result = self.evaluate_patch(patches[i], metadata_list[i])
            results.append(result)

        return results