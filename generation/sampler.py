import torch
import time
from typing import Optional
from tqdm import tqdm
class Sampler:
    def __init__(
        self,
        unet,
        noise_schedule,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.unet = unet.to(device)
        self.schedule = noise_schedule
        self.device = device
        self.unet.eval()

        for name in ["sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                     "posterior_variance", "alphas", "betas", "alphas_cumprod"]:
            val = getattr(self.schedule, name, None)
            if isinstance(val, torch.Tensor):
                setattr(self.schedule, name, val.to(device))

        print(f"\n{'='*70}")
        print(f"Sampler Initialized")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Timesteps: {self.schedule.num_timesteps}")
        print(f"{'='*70}\n")

    @torch.no_grad()
    def _denoise_step(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        predicted_noise: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        alpha_t = self.schedule.alphas[t_batch].to(device=device, dtype=dtype).view(-1, 1)
        beta_t = self.schedule.betas[t_batch].to(device=device, dtype=dtype).view(-1, 1)
        alpha_cumprod_t = self.schedule.alphas_cumprod[t_batch].to(device=device, dtype=dtype).view(-1, 1)

        eps = 1e-8
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=eps))
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt((1.0 - alpha_cumprod_t).clamp(min=eps))

        mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        if t_batch[0] > 0:
            variance = beta_t
            std = torch.sqrt(variance.clamp(min=eps))
            noise = torch.randn_like(x)
            x_prev = mean + std * temperature * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def sample_single_patch(
        self,
        normalizer,
        previous_latent: Optional[torch.Tensor],
        target_difficulty: float,
        previous_difficulties: Optional[list] = None,
        temperature: float = 0.9,
        guidance_scale: float = 3.0,
        show_progress: bool = False
    ) -> torch.Tensor:

        device = self.device
        latent_dim = self.unet.latent_dim

        x = torch.randn(1, latent_dim, device=device) * temperature

        target_diff_tensor = torch.tensor([target_difficulty], device=device, dtype=torch.float32)

        if previous_latent is None:
            prev_lat = torch.zeros((1, 1, latent_dim), device=device)
            prev_diff = torch.zeros((1, 1), device=device)
        else:
            if previous_latent.dim() == 1:
                previous_latent = previous_latent.unsqueeze(0)
            prev_lat = previous_latent.unsqueeze(0).to(device)
            if previous_difficulties is None:
                prev_diff = torch.zeros((1, prev_lat.shape[1]), device=device)
            else:
                prev_diff = torch.tensor(previous_difficulties, device=device).unsqueeze(0)

        timesteps = range(self.schedule.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"Sampling (CFG scale={guidance_scale})")

        self.unet.eval()
        for t in timesteps:
            t_batch = torch.tensor([t], device=device, dtype=torch.long)

            noise_cond = self.unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_difficulties=prev_diff,
                target_difficulty=target_diff_tensor
            )

            noise_uncond = self.unet(
                x=x,
                timesteps=t_batch,
                previous_latents=prev_lat,
                previous_difficulties=prev_diff,
                target_difficulty=None
            )

            noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            x = self._denoise_step(x, t_batch, noise_guided, temperature)

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e8, neginf=-1e8)

        return x.squeeze(0)

    @torch.no_grad()
    def sample_level(
        self,
        normalizer,
        num_patches: int,
        difficulty_target: float = 0.5,
        temperature: float = 0.9,
        guidance_scale: float = 3.0,
        num_prev: int = 1,
        show_progress: bool = True,
        seed_latent: Optional[torch.Tensor] = None,
        seed_difficulty: Optional[float] = None,
    ) -> torch.Tensor:
        generated = []
        prev_buffer = []
        prev_diff_buffer = []
        
        if seed_latent is not None:
            if seed_latent.dim() == 1:
                seed_latent = seed_latent.unsqueeze(0)
            prev_buffer.append(seed_latent.squeeze(0).cpu())
            prev_diff_buffer.append(seed_difficulty if seed_difficulty is not None else 0.0)
            print(f"  Using seed context: latent norm={seed_latent.norm():.2f}, difficulty={prev_diff_buffer[0]:.3f}")

        if isinstance(difficulty_target, (list, tuple)):
            difficulty_schedule = [float(p) for p in difficulty_target]
        else:
            scalar_target = float(difficulty_target)
            if scalar_target < 0.25:
                target_class = 0.0
            elif scalar_target < 0.75:
                target_class = 0.5
            else:
                target_class = 1.0
            
            if num_patches == 1:
                difficulty_schedule = [target_class]
            else:
                # Bell curve using only discrete values: 0.0, 0.5, 1.0
                # Pattern: 0.0 -> 0.5 -> target -> target -> 0.5 -> 0.0
                difficulty_schedule = []
                mid = (num_patches - 1) / 2.0
                for i in range(num_patches):
                    dist = abs(i - mid) / mid if mid > 0 else 0
                    
                    # Map distance to discrete difficulty class
                    if dist <= 0.25:  # Center - use target class
                        diff = target_class
                    elif dist <= 0.6:  # Transition - one step below target
                        diff = max(0.0, target_class - 0.5)
                    else:  # Edges - easy
                        diff = 0.0
                    difficulty_schedule.append(diff)
        
        print(f"  Difficulty schedule: {[f'{d:.2f}' for d in difficulty_schedule]}")


        iterator = range(num_patches)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating level")

        for i in iterator:
            if len(prev_buffer) == 0:
                prev_lat = None
                prev_diff = None
            else:
                prev_lat = torch.stack(prev_buffer[-num_prev:], dim=0).to(self.device)
                prev_diff = prev_diff_buffer[-num_prev:]

            latent = self.sample_single_patch(
                normalizer=normalizer,
                previous_latent=prev_lat,
                target_difficulty=difficulty_schedule[i],
                previous_difficulties=prev_diff,
                temperature=temperature,
                guidance_scale=guidance_scale,
                show_progress=False
            )

            generated.append(latent)
            prev_buffer.append(latent.cpu())
            prev_diff_buffer.append(difficulty_schedule[i])

        result = torch.stack(generated, dim=0)

        if show_progress:
            print(f"\n✓ Generated {num_patches} patches with CFG (scale={guidance_scale})")

        return result