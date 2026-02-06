import torch
import numpy as np
import yaml
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from models.autoencoder import Autoencoder
from models.diffusion import DiffusionUNet
from generation.sampler import Sampler
from models.latent_normalizer import LatentNormalizer
from models.noise_scheduler import NoiseSchedule
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
from evaluation.astar_agent import test_playability, test_playability_batch
from data.parser import LevelParser
from config.model_config import (
    AutoencoderConfig,
    DiffusionConfig,
    NoiseScheduleConfig,
    NormalizerConfig
)

try:
    from mario_gpt import MarioLM, SampleOutput
    BASELINE_AVAILABLE = True
except ImportError:
    print("Warning: mario-gpt not installed. Baseline Model comparison will be skipped.")
    BASELINE_AVAILABLE = False


def load_diffusion_model(device: str) -> tuple:
    ae_config = AutoencoderConfig()
    diff_config = DiffusionConfig()
    schedule_config = NoiseScheduleConfig()
    normalizer_config = NormalizerConfig()
    
    with open('config/generation_config.yaml', 'r') as f:
        gen_config = yaml.safe_load(f)
    
    autoencoder = Autoencoder(
        num_tile_types=ae_config.num_tile_types,
        embedding_dim=ae_config.embedding_dim,
        latent_dim=ae_config.latent_dim,
        patch_height=ae_config.patch_height,
        patch_width=ae_config.patch_width
    )
    ae_checkpoint = torch.load(gen_config['models']['autoencoder_path'], map_location=device)
    autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    autoencoder.to(device).eval()
    
    unet = DiffusionUNet(
        latent_dim=diff_config.latent_dim,
        time_emb_dim=diff_config.time_emb_dim,
        context_emb_dim=diff_config.context_emb_dim,
        hidden_dims=diff_config.hidden_dims,
        num_res_blocks=diff_config.num_res_blocks,
        cond_dropout=diff_config.cond_dropout,
    ).to(device)
    diff_checkpoint = torch.load(gen_config['models']['diffusion_path'], map_location=device)
    unet.load_state_dict(diff_checkpoint['unet_state_dict'], strict=False)
    unet.eval()
    
    normalizer = LatentNormalizer(target_norm=normalizer_config.norm)
    normalizer.load(gen_config['models']['normalizer_path'])
    
    schedule = NoiseSchedule(
        num_timesteps=schedule_config.num_timesteps,
        schedule_type=schedule_config.schedule_type,
        device=device,
    )
    
    sampler = Sampler(unet, schedule, device=device)
    
    return autoencoder, sampler, normalizer


def generate_diffusion_samples(
    autoencoder,
    sampler, 
    normalizer,
    difficulty_targets: List[float],
    samples_per_target: int,
    temperature: float = 1.0,
    guidance_scale: float = 5.0,
    device: str = 'cuda'
) -> Dict[float, List[np.ndarray]]:
    results = {}
    
    for target in difficulty_targets:
        print(f"\nGenerating {samples_per_target} samples at difficulty {target:.1f}...")
        samples = []
        
        for _ in tqdm(range(samples_per_target), desc=f"Diffusion (target={target})"):
            latent = sampler.sample_single_patch(
                normalizer=normalizer,
                previous_latent=None,
                target_difficulty=target,
                previous_difficulties=None,
                temperature=temperature,
                guidance_scale=guidance_scale,
                show_progress=False
            )
            
            latent_denorm = normalizer.denormalize(latent.unsqueeze(0))
            
            with torch.no_grad():
                latent_denorm = latent_denorm.to(device)
                decoded_logits = autoencoder.decoder(latent_denorm)
                
                if decoded_logits.dim() == 4:
                    patch = torch.argmax(decoded_logits, dim=1)
                else:
                    patch = torch.argmax(decoded_logits, dim=-1)
                
                patch = patch.cpu().numpy()[0]
            
            samples.append(patch)
        
        results[target] = samples
    
    return results


def generate_baseline_samples(
    difficulty_targets: List[float],
    samples_per_target: int,
    level_height: int = 14,
    level_width: int = 16
) -> Dict[float, List[np.ndarray]]:
    if not BASELINE_AVAILABLE:
        print("Baseline Model not available. Skipping.")
        return {}
    
    mario_lm = MarioLM()
    parser = LevelParser()
    
    prompt_map = {
        0.0: "no enemies",      
        0.5: "some enemies",    
        1.0: "many enemies"     
    }
    
    results = {}
    
    for target in difficulty_targets:
        prompt = prompt_map.get(target, "some enemies")
        print(f"\nGenerating {samples_per_target} samples with prompt: '{prompt}'...")
        samples = []
        
        for _ in tqdm(range(samples_per_target), desc=f"Baseline ({prompt})"):
            try:
                generated = mario_lm.sample(
                    prompts=[prompt],
                    num_steps=16,
                    temperature=1.0,
                    use_tqdm=False
                )
                
                if isinstance(generated, list) and len(generated) > 0:
                    sample_output = generated[0]
                    if hasattr(sample_output, 'level'):
                        level_lines = sample_output.level
                    else:
                        level_lines = ['-' * level_width] * level_height
                elif hasattr(generated, 'level'):
                    level_lines = generated.level
                else:
                    level_lines = ['-' * level_width] * level_height
                
                if isinstance(level_lines, str):
                    level_lines = level_lines.strip().split('\n')
                
                char_map = {
                    'x': 'X', 's': 'S', 'e': 'E', 'q': 'Q', '?': '?',
                    'o': 'o', '#': 'X', 'g': 'E', 'k': 'E',
                }
                
                def convert_char(c):
                    if c in char_map:
                        return char_map[c]
                    elif c.upper() in 'XSQE-<>[]oBb?o':
                        return c.upper() if c.upper() in 'XSQE' else c
                    elif c in '-':
                        return '-'
                    else:
                        return '-'
                
                level_lines = level_lines[:level_height]
                converted_lines = []
                for line in level_lines:
                    converted = ''.join(convert_char(c) for c in str(line)[:level_width])
                    converted = converted.ljust(level_width, '-')
                    converted_lines.append(converted)
                level_lines = converted_lines
                
                while len(level_lines) < level_height:
                    level_lines.append('-' * level_width)
                
                patch = parser.parse_level_list(level_lines)
                samples.append(patch)
                
            except Exception as e:
                print(f"Warning: Baseline generation failed: {e}")
                samples.append(np.full((level_height, level_width), 2, dtype=np.int32))
        
        results[target] = samples
    
    return results


def evaluate_controllability(
    samples: Dict[float, List[np.ndarray]],
    difficulty_evaluator: PatchDifficultyEvaluator
) -> Dict:
    results = {}
    all_targets = []
    all_actual = []
    
    for target, patches in samples.items():
        actual_scores = []
        for patch in patches:
            eval_result = difficulty_evaluator.evaluate_patch(patch, {})
            actual_scores.append(eval_result['scores']['difficulty_score'])
            all_targets.append(target)
            all_actual.append(eval_result['scores']['difficulty_score'])
        
        mean_actual = np.mean(actual_scores)
        std_actual = np.std(actual_scores)
        
        results[target] = {
            'mean': mean_actual,
            'std': std_actual,
            'error': abs(mean_actual - target),
            'samples': actual_scores
        }
    
    mae = np.mean([results[t]['error'] for t in results])
    correlation = np.corrcoef(all_targets, all_actual)[0, 1] if len(set(all_targets)) > 1 else 0
    correct = sum(1 for t, a in zip(all_targets, all_actual) if t == a)
    accuracy = correct / len(all_targets) * 100 if all_targets else 0
    
    results['overall'] = {
        'mae': mae,
        'correlation': correlation,
        'accuracy': accuracy
    }
    
    return results


def evaluate_playability(samples: Dict[float, List[np.ndarray]]) -> Dict:
    results = {}
    all_playable = []
    all_path_lengths = []
    
    for target, patches in samples.items():
        playable_count = 0
        path_lengths = []
        
        for patch in patches:
            result = test_playability(patch)
            if result['playable']:
                playable_count += 1
                path_lengths.append(result['path_length'])
                all_playable.append(1)
            else:
                all_playable.append(0)
        
        rate = playable_count / len(patches) * 100 if patches else 0
        avg_path = np.mean(path_lengths) if path_lengths else 0
        
        results[target] = {
            'playable_count': playable_count,
            'total': len(patches),
            'playability_rate': rate,
            'avg_path_length': avg_path
        }
        all_path_lengths.extend(path_lengths)
    
    results['overall'] = {
        'playability_rate': np.mean(all_playable) * 100 if all_playable else 0,
        'avg_path_length': np.mean(all_path_lengths) if all_path_lengths else 0
    }
    
    return results


def evaluate_diversity(samples: Dict[float, List[np.ndarray]]) -> Dict:
    results = {}
    all_entropies = []
    
    for target, patches in samples.items():
        entropies = []
        for patch in patches:
            flat = patch.flatten()
            counts = Counter(flat)
            total = len(flat)
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropies.append(entropy)
        
        results[target] = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies)
        }
        all_entropies.extend(entropies)
    
    results['overall'] = {
        'mean_entropy': np.mean(all_entropies) if all_entropies else 0,
        'std_entropy': np.std(all_entropies) if all_entropies else 0
    }
    
    return results


def create_comparison_report(
    diffusion_results: Dict,
    baseline_results: Dict,
    save_path: str = 'output/visualizations/model_comparison.png'
):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Diffusion Model':<20} {'Baseline Model':<20}")
    print("-" * 70)
    
    diff_ctrl = diffusion_results.get('controllability', {}).get('overall', {})
    base_ctrl = baseline_results.get('controllability', {}).get('overall', {})
    
    print(f"{'Controllability MAE':<30} {diff_ctrl.get('mae', 'N/A'):<20.4f} {base_ctrl.get('mae', 'N/A') if base_ctrl else 'N/A':<20}")
    print(f"{'Controllability Accuracy':<30} {diff_ctrl.get('accuracy', 'N/A'):<20.1f}% {str(base_ctrl.get('accuracy', 'N/A')) + '%' if base_ctrl else 'N/A':<20}")
    print(f"{'Controllability Correlation':<30} {diff_ctrl.get('correlation', 'N/A'):<20.4f} {base_ctrl.get('correlation', 'N/A') if base_ctrl else 'N/A':<20}")
    
    diff_play = diffusion_results.get('playability', {}).get('overall', {})
    base_play = baseline_results.get('playability', {}).get('overall', {})
    
    print(f"{'Playability Rate':<30} {diff_play.get('playability_rate', 'N/A'):<20.1f}% {str(base_play.get('playability_rate', 'N/A')) + '%' if base_play else 'N/A':<20}")
    print(f"{'Avg Path Length':<30} {diff_play.get('avg_path_length', 'N/A'):<20.1f} {base_play.get('avg_path_length', 'N/A') if base_play else 'N/A':<20}")
    
    diff_div = diffusion_results.get('diversity', {}).get('overall', {})
    base_div = baseline_results.get('diversity', {}).get('overall', {})
    
    print(f"{'Diversity (Entropy)':<30} {diff_div.get('mean_entropy', 'N/A'):<20.4f} {base_div.get('mean_entropy', 'N/A') if base_div else 'N/A':<20}")
    
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    targets = [0.0, 0.5, 1.0]
    diff_means = [diffusion_results['controllability'].get(t, {}).get('mean', 0) for t in targets]
    
    x = np.arange(len(targets))
    width = 0.35
    
    ax1.bar(x - width/2, diff_means, width, label='Diffusion Model', color='blue', alpha=0.7)
    
    if baseline_results.get('controllability'):
        base_means = [baseline_results['controllability'].get(t, {}).get('mean', 0) for t in targets]
        ax1.bar(x + width/2, base_means, width, label='Baseline Model', color='orange', alpha=0.7)
    
    ax1.plot([-0.5, len(targets)-0.5], [0, 1], 'r--', alpha=0.5, label='Perfect')
    ax1.set_xlabel('Target Difficulty')
    ax1.set_ylabel('Actual Difficulty')
    ax1.set_title('Controllability Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Easy (0.0)', 'Medium (0.5)', 'Hard (1.0)'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    diff_play_rates = [diffusion_results['playability'].get(t, {}).get('playability_rate', 0) for t in targets]
    
    ax2.bar(x - width/2, diff_play_rates, width, label='Diffusion Model', color='blue', alpha=0.7)
    
    if baseline_results.get('playability'):
        base_play_rates = [baseline_results['playability'].get(t, {}).get('playability_rate', 0) for t in targets]
        ax2.bar(x + width/2, base_play_rates, width, label='Baseline Model', color='orange', alpha=0.7)
    
    ax2.set_xlabel('Target Difficulty')
    ax2.set_ylabel('Playability Rate (%)')
    ax2.set_title('Playability Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    ax3 = axes[2]
    diff_entropies = [diffusion_results['diversity'].get(t, {}).get('mean_entropy', 0) for t in targets]
    
    ax3.bar(x - width/2, diff_entropies, width, label='Diffusion Model', color='blue', alpha=0.7)
    
    if baseline_results.get('diversity'):
        base_entropies = [baseline_results['diversity'].get(t, {}).get('mean_entropy', 0) for t in targets]
        ax3.bar(x + width/2, base_entropies, width, label='Baseline Model', color='orange', alpha=0.7)
    
    ax3.set_xlabel('Target Difficulty')
    ax3.set_ylabel('Tile Entropy (bits)')
    ax3.set_title('Diversity Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Easy', 'Medium', 'Hard'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    difficulty_targets = [0.0, 0.5, 1.0]
    samples_per_target = 20
    temperature = 1.0
    guidance_scale = 5.0
    
    print("\n" + "=" * 60)
    print("Loading Diffusion Model...")
    print("=" * 60)
    autoencoder, sampler, normalizer = load_diffusion_model(device)
    
    parser = LevelParser()
    difficulty_evaluator = PatchDifficultyEvaluator(parser)
    
    print("\n" + "=" * 60)
    print("Generating Diffusion Model Samples")
    print("=" * 60)
    diffusion_samples = generate_diffusion_samples(
        autoencoder, sampler, normalizer,
        difficulty_targets, samples_per_target,
        temperature, guidance_scale, device
    )
    
    print("\n" + "=" * 60)
    print("Generating Baseline Model Samples")
    print("=" * 60)
    baseline_samples = generate_baseline_samples(
        difficulty_targets, samples_per_target
    )
    
    print("\n" + "=" * 60)
    print("Evaluating Models")
    print("=" * 60)
    
    diffusion_results = {
        'controllability': evaluate_controllability(diffusion_samples, difficulty_evaluator),
        'playability': evaluate_playability(diffusion_samples),
        'diversity': evaluate_diversity(diffusion_samples)
    }
    
    baseline_results = {}
    if baseline_samples:
        baseline_results = {
            'controllability': evaluate_controllability(baseline_samples, difficulty_evaluator),
            'playability': evaluate_playability(baseline_samples),
            'diversity': evaluate_diversity(baseline_samples)
        }
    
    create_comparison_report(diffusion_results, baseline_results)
    
    return diffusion_results, baseline_results


if __name__ == "__main__":
    main()
