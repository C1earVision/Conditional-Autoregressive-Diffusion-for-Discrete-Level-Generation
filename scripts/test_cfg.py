
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.diffusion import DiffusionUNet
from models.latent_normalizer import LatentNormalizer
from models.noise_scheduler import NoiseSchedule
from config.model_config import DiffusionConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
diff_config = DiffusionConfig()

print("="*70)
print("CFG DIAGNOSTIC TEST")
print("="*70)

# Load model
unet = DiffusionUNet(
    latent_dim=diff_config.latent_dim,
    time_emb_dim=diff_config.time_emb_dim,
    context_emb_dim=diff_config.context_emb_dim,
    hidden_dims=diff_config.hidden_dims,
    num_res_blocks=diff_config.num_res_blocks,
    cond_dropout=diff_config.cond_dropout
).to(device)

checkpoint = torch.load('checkpoints/diffusion_best.pth', map_location=device)
unet.load_state_dict(checkpoint['unet_state_dict'])
unet.eval()
print(f"✓ Loaded model from checkpoints/diffusion_best.pth")

normalizer = LatentNormalizer(target_norm=11)
normalizer.load('checkpoints/latent_normalizer.pth')
schedule = NoiseSchedule(num_timesteps=500)

# Test parameters
batch_size = 1
latent_dim = diff_config.latent_dim
difficulties_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]
timesteps_to_test = [499, 400, 300, 200, 100, 50, 0]

print(f"\n{'='*70}")
print("TEST 1: Noise Prediction Difference (Cond vs Uncond)")
print(f"{'='*70}")

# Create test inputs
x = torch.randn(batch_size, latent_dim, device=device)
prev_lat = torch.zeros((batch_size, 1, latent_dim), device=device)
prev_play = torch.zeros((batch_size, 1), device=device)

results = {}

with torch.no_grad():
    for t in timesteps_to_test:
        t_batch = torch.tensor([t], device=device, dtype=torch.long)
        
        # Unconditional prediction
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

# Print results
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

# Test if different difficulties produce different noise predictions
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

for i, d1 in enumerate(difficulties_to_test):
    for j, d2 in enumerate(difficulties_to_test):
        if i < j:
            l2_dist = (noise_predictions[i] - noise_predictions[j]).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                noise_predictions[i].flatten().unsqueeze(0),
                noise_predictions[j].flatten().unsqueeze(0)
            ).item()
            print(f"{d1:<8.2f} {d2:<8.2f} {l2_dist:<15.4f} {cos_sim:<15.4f}")

print(f"\n{'='*70}")
print("TEST 3: CFG Effect Analysis")
print(f"{'='*70}")

guidance_scales = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
target_diff = torch.tensor([0.8], device=device, dtype=torch.float32)

with torch.no_grad():
    noise_cond = unet(
        x=x,
        timesteps=t_batch,
        previous_latents=prev_lat,
        previous_difficulties=prev_play,
        target_difficulty=target_diff
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

for scale in guidance_scales:
    noise_guided = noise_uncond + scale * diff_vector
    change = (noise_guided - noise_uncond).norm().item()
    print(f"{scale:<10.1f} {noise_guided.norm().item():<15.4f} {change:<20.4f}")

print(f"\n{'='*70}")
print("DIAGNOSIS SUMMARY")
print(f"{'='*70}")

avg_diff_norm = np.mean([d['diff_norm'] for d in results[200]['diffs']])
avg_noise_norm = np.mean([d['cond_norm'] for d in results[200]['diffs']])
pct_diff = avg_diff_norm / avg_noise_norm * 100

print(f"\nAverage difference between cond/uncond: {avg_diff_norm:.4f}")
print(f"Average noise norm: {avg_noise_norm:.4f}")
print(f"Difference as percentage: {pct_diff:.2f}%")

if pct_diff < 5:
    print("\n⚠️  WARNING: CFG signal is VERY WEAK (<5%)")
    print("   The model may not have learned meaningful conditioning.")
    print("   Possible causes:")
    print("   - cond_dropout too low during training")
    print("   - Difficulty values not correlated with level features")
    print("   - Model ignoring conditioning signals")
elif pct_diff < 15:
    print("\n⚠️  WARNING: CFG signal is WEAK (5-15%)")
    print("   CFG will have limited effect on generation.")
elif pct_diff < 30:
    print("\n✓  CFG signal is MODERATE (15-30%)")
    print("   CFG should work with higher guidance scales (5-10).")
else:
    print("\n✓  CFG signal is STRONG (>30%)")
    print("   CFG should work well even with moderate guidance scales.")

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")
