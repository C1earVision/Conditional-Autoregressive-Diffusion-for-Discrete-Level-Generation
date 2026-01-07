import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class ModelPerformanceEvaluator:    
    def test_cfg_signal(self,
                        unet,
                        latent_dim: int = 128,
                        difficulties_to_test: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
                        timesteps_to_test: List[int] = [499, 400, 300, 200, 100, 50, 0],
                        guidance_scales: List[float] = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                        target_difficulty: float = 0.8,
                        device: str = 'cuda',
                        save_path: Optional[str] = None) -> Dict:
        
        print("="*70)
        print("CFG DIAGNOSTIC TEST")
        print("="*70)
        
        batch_size = 1
        x = torch.randn(batch_size, latent_dim, device=device)
        prev_lat = torch.zeros((batch_size, 1, latent_dim), device=device)
        prev_play = torch.zeros((batch_size, 1), device=device)
        
        results = {}
        
        print(f"\n{'='*70}")
        print("TEST 1: Noise Prediction Difference (Cond vs Uncond)")
        print(f"{'='*70}")
        
        with torch.no_grad():
            for t in timesteps_to_test:
                t_batch = torch.tensor([t], device=device, dtype=torch.long)
                
                noise_uncond = unet(
                    x=x,
                    timesteps=t_batch,
                    previous_latents=prev_lat,
                    previous_difficulties=prev_play,
                    target_difficulty=None
                )
                
                results[t] = {'uncond_norm': noise_uncond.norm().item(), 'diffs': []}
                
                for diff in difficulties_to_test:
                    target_diff = torch.tensor([diff], device=device, dtype=torch.float32)
                    
                    noise_cond = unet(
                        x=x,
                        timesteps=t_batch,
                        previous_latents=prev_lat,
                        previous_difficulties=prev_play,
                        target_difficulty=target_diff
                    )
                    
                    diff_norm = (noise_cond - noise_uncond).norm().item()
                    cond_norm = noise_cond.norm().item()
                    
                    results[t]['diffs'].append({
                        'difficulty': diff,
                        'cond_norm': cond_norm,
                        'diff_norm': diff_norm,
                        'pct_diff': (diff_norm / cond_norm * 100) if cond_norm > 0 else 0
                    })
        
        print(f"\n{'Timestep':<10} {'Uncond':<10} {'Diff=0.0':<12} {'Diff=0.5':<12} {'Diff=1.0':<12}")
        print("-"*60)
        
        for t in timesteps_to_test:
            r = results[t]
            diffs = r['diffs']
            d0 = next(d for d in diffs if d['difficulty'] == 0.0)
            d5 = next(d for d in diffs if d['difficulty'] == 0.5)
            d10 = next(d for d in diffs if d['difficulty'] == 1.0)
            
            print(f"{t:<10} {r['uncond_norm']:<10.4f} {d0['diff_norm']:<12.4f} {d5['diff_norm']:<12.4f} {d10['diff_norm']:<12.4f}")
        
        print(f"\n{'='*70}")
        print("TEST 2: Different Difficulties Produce Different Outputs?")
        print(f"{'='*70}")
        
        t_test = 250
        t_batch = torch.tensor([t_test], device=device, dtype=torch.long)
        
        noise_predictions = []
        with torch.no_grad():
            for diff in difficulties_to_test:
                target_diff = torch.tensor([diff], device=device, dtype=torch.float32)
                noise = unet(
                    x=x,
                    timesteps=t_batch,
                    previous_latents=prev_lat,
                    previous_difficulties=prev_play,
                    target_difficulty=target_diff
                )
                noise_predictions.append(noise)
        
        print(f"\nPairwise L2 distance between noise predictions at t={t_test}:")
        print(f"{'Diff1':<8} {'Diff2':<8} {'L2 Distance':<15} {'Cosine Sim':<15}")
        print("-"*50)
        
        pairwise_results = []
        for i, d1 in enumerate(difficulties_to_test):
            for j, d2 in enumerate(difficulties_to_test):
                if i < j:
                    l2_dist = (noise_predictions[i] - noise_predictions[j]).norm().item()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        noise_predictions[i].flatten().unsqueeze(0),
                        noise_predictions[j].flatten().unsqueeze(0)
                    ).item()
                    print(f"{d1:<8.2f} {d2:<8.2f} {l2_dist:<15.4f} {cos_sim:<15.4f}")
                    pairwise_results.append({'d1': d1, 'd2': d2, 'l2': l2_dist, 'cos_sim': cos_sim})
        
        print(f"\n{'='*70}")
        print("TEST 3: CFG Effect Analysis")
        print(f"{'='*70}")
        
        target_diff_tensor = torch.tensor([target_difficulty], device=device, dtype=torch.float32)
        
        with torch.no_grad():
            noise_cond = unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_difficulties=prev_play,
                target_difficulty=target_diff_tensor
            )
            
            noise_uncond = unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_difficulties=prev_play,
                target_difficulty=None
            )
        
        diff_vector = noise_cond - noise_uncond
        print(f"\nBase noise_cond norm: {noise_cond.norm().item():.4f}")
        print(f"Base noise_uncond norm: {noise_uncond.norm().item():.4f}")
        print(f"Difference vector norm: {diff_vector.norm().item():.4f}")
        print(f"Difference as % of cond: {(diff_vector.norm() / noise_cond.norm() * 100).item():.2f}%")
        
        print(f"\n{'Scale':<10} {'Guided Norm':<15} {'Change from Uncond':<20}")
        print("-"*50)
        
        cfg_effect_results = []
        for scale in guidance_scales:
            noise_guided = noise_uncond + scale * diff_vector
            change = (noise_guided - noise_uncond).norm().item()
            print(f"{scale:<10.1f} {noise_guided.norm().item():<15.4f} {change:<20.4f}")
            cfg_effect_results.append({'scale': scale, 'guided_norm': noise_guided.norm().item(), 'change': change})
        
        print(f"\n{'='*70}")
        print("DIAGNOSIS SUMMARY")
        print(f"{'='*70}")
        
        diag_timestep = 200 if 200 in results else timesteps_to_test[len(timesteps_to_test)//2]
        avg_diff_norm = np.mean([d['diff_norm'] for d in results[diag_timestep]['diffs']])
        avg_noise_norm = np.mean([d['cond_norm'] for d in results[diag_timestep]['diffs']])
        pct_diff = avg_diff_norm / avg_noise_norm * 100
        
        print(f"\nAverage difference between cond/uncond: {avg_diff_norm:.4f}")
        print(f"Average noise norm: {avg_noise_norm:.4f}")
        print(f"Difference as percentage: {pct_diff:.2f}%")
        
        if pct_diff < 5:
            diagnosis = "VERY_WEAK"
            print("\n⚠️  WARNING: CFG signal is VERY WEAK (<5%)")
            print("   The model may not have learned meaningful conditioning.")
            print("   Possible causes:")
            print("   - cond_dropout too low during training")
            print("   - Difficulty values not correlated with level features")
            print("   - Model ignoring conditioning signals")
        elif pct_diff < 15:
            diagnosis = "WEAK"
            print("\n⚠️  WARNING: CFG signal is WEAK (5-15%)")
            print("   CFG will have limited effect on generation.")
        elif pct_diff < 30:
            diagnosis = "MODERATE"
            print("\n✓  CFG signal is MODERATE (15-30%)")
            print("   CFG should work with higher guidance scales (5-10).")
        else:
            diagnosis = "STRONG"
            print("\n✓  CFG signal is STRONG (>30%)")
            print("   CFG should work well even with moderate guidance scales.")
        
        print(f"\n{'='*70}")
        print("TEST COMPLETE")
        print(f"{'='*70}")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("CFG DIAGNOSTIC TEST RESULTS\n")
                f.write("="*70 + "\n\n")
                f.write(f"Diagnosis: {diagnosis}\n")
                f.write(f"Average diff/uncond difference: {pct_diff:.2f}%\n\n")
                f.write("Timestep results:\n")
                for t in timesteps_to_test:
                    r = results[t]
                    f.write(f"  t={t}: uncond_norm={r['uncond_norm']:.4f}\n")
            print(f"✓ Results saved to {save_path}")
        
        return {
            'timestep_results': results,
            'pairwise_results': pairwise_results,
            'cfg_effect_results': cfg_effect_results,
            'diagnosis': diagnosis,
            'pct_diff': pct_diff
        }
    def evaluate_difficulty_comparison(self,
                                      sampler,
                                      normalizer,
                                      autoencoder,
                                      parser,
                                      difficulty_evaluator,
                                      target_difficulties: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                                      num_samples_per_target: int = 5,
                                      guidance_scale: float = 3.0,
                                      temperature: float = 0.5,
                                      device: str = 'cuda',
                                      save_path: Optional[str] = None) -> Dict:

        print(f"\n{'='*70}")
        print(f"DIFFICULTY EVALUATION COMPARISON")
        print(f"{'='*70}")
        print(f"Target difficulties: {target_difficulties}")
        print(f"Samples per target: {num_samples_per_target}")
        print(f"Guidance Scale: {guidance_scale}")
        print(f"{'='*70}\n")

        results = {
            'target_difficulties': target_difficulties,
            'evaluations': [],
            'summary': {}
        }

        all_targets = []
        all_actual_scores = []

        for target_diff in target_difficulties:
            print(f"\nGenerating {num_samples_per_target} patches with target difficulty = {target_diff}...")

            target_scores = []
            actual_scores = []
            patches_list = []

            for sample_idx in range(num_samples_per_target):
                latent = sampler.sample_single_patch(
                    normalizer=normalizer,
                    previous_latent=None,
                    target_difficulty=target_diff,
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
                    elif decoded_logits.dim() == 3:
                        patch = decoded_logits.long()
                    elif decoded_logits.dim() == 5:
                        decoded_logits = decoded_logits.squeeze(1)
                        patch = torch.argmax(decoded_logits, dim=1)
                    else:
                        patch = torch.argmax(decoded_logits, dim=-1)

                    patch = patch.cpu().numpy()[0]

                eval_result = difficulty_evaluator.evaluate_patch(
                    patch,
                    metadata={'target_difficulty': target_diff, 'sample_idx': sample_idx}
                )

                difficulty_score = eval_result['scores']['difficulty_score']

                target_scores.append(target_diff)
                actual_scores.append(difficulty_score)
                patches_list.append(patch)

                all_targets.append(target_diff)
                all_actual_scores.append(difficulty_score)

                results['evaluations'].append({
                    'target_difficulty': target_diff,
                    'actual_difficulty': difficulty_score,
                    'sample_idx': sample_idx,
                    'patch': patch,
                    'full_evaluation': eval_result
                })

            mean_actual = np.mean(actual_scores)
            std_actual = np.std(actual_scores)

            results['summary'][target_diff] = {
                'mean_difficulty': mean_actual,
                'std_difficulty': std_actual,
                'target_difficulty': target_diff,
                'error': abs(mean_actual - target_diff),
                'samples': actual_scores
            }

            print(f"  Target: {target_diff:.2f} | "
                  f"Actual Difficulty: {mean_actual:.3f} ± {std_actual:.3f} | "
                  f"Error: {abs(mean_actual - target_diff):.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        target_vals = list(results['summary'].keys())
        mean_vals = [results['summary'][t]['mean_difficulty'] for t in target_vals]
        std_vals = [results['summary'][t]['std_difficulty'] for t in target_vals]

        ax1.errorbar(target_vals, mean_vals, yerr=std_vals,
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    label='Generated (Mean ± Std)', color='blue', alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax1.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax1.set_title('Target vs Actual Difficulty (Aggregated)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[0, 1]
        ax2.scatter(all_targets, all_actual_scores, alpha=0.5, s=50, color='green')
        ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Alignment', alpha=0.5)
        ax2.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Difficulty Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Individual Samples', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        ax3 = axes[1, 0]
        errors = [results['summary'][t]['error'] for t in target_vals]
        ax3.bar(range(len(target_vals)), errors, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(target_vals)))
        ax3.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax3.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Error by Target', fontsize=14, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)

        ax4 = axes[1, 1]
        data_for_box = [results['summary'][t]['samples'] for t in target_vals]
        bp = ax4.boxplot(data_for_box, positions=range(len(target_vals)),
                        widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax4.plot(range(len(target_vals)), target_vals, 'go-',
                linewidth=2, markersize=8, label='Target', alpha=0.7)
        ax4.set_xticks(range(len(target_vals)))
        ax4.set_xticklabels([f'{t:.1f}' for t in target_vals])
        ax4.set_xlabel('Target difficulty', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Difficulty Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_title('Distribution of Generated Difficulties', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Evaluation plot saved to {save_path}")

        plt.show()

        print(f"\n{'='*90}")
        print(f"DETAILED EVALUATION RESULTS")
        print(f"{'='*90}")
        print(f"{'Target':<10} {'Generated':<12} {'Std Dev':<12} {'Error':<12} {'% Error':<12} {'Range':<20}")
        print(f"{'-'*90}")

        for target in target_vals:
            summary = results['summary'][target]
            samples = summary['samples']
            range_str = f"[{min(samples):.3f}, {max(samples):.3f}]"
            pct_error = (summary['error'] / target * 100) if target != 0 else 0
            print(f"{target:<10.2f} {summary['mean_difficulty']:<12.3f} "
                  f"{summary['std_difficulty']:<12.3f} {summary['error']:<12.3f} "
                  f"{pct_error:<12.1f}% {range_str:<20}")

        overall_mae = np.mean([results['summary'][t]['error'] for t in target_vals])
        overall_correlation = np.corrcoef(all_targets, all_actual_scores)[0, 1]

        print(f"\n{'='*90}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*90}")
        print(f"Mean Absolute Error (MAE): {overall_mae:.4f}")
        print(f"Correlation Coefficient: {overall_correlation:.4f}")
        print(f"Total Samples Generated: {len(all_targets)}")
        print(f"{'='*90}\n")

        results['overall'] = {
            'mae': overall_mae,
            'correlation': overall_correlation,
            'total_samples': len(all_targets)
        }

        return results
