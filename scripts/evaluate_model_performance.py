import torch
import yaml
from models.autoencoder import Autoencoder
from models.diffusion import DiffusionUNet
from generation.sampler import Sampler
from models.latent_normalizer import LatentNormalizer
from models.noise_scheduler import NoiseSchedule
from evaluation.difficulty_evaluator import PatchDifficultyEvaluator
from evaluation.model_performance_evaluator import ModelPerformanceEvaluator
from data.parser import LevelParser
from config.model_config import (
    AutoencoderConfig,
    DiffusionConfig,
    NoiseScheduleConfig,
    NormalizerConfig
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

ae_config = AutoencoderConfig()
diff_config = DiffusionConfig()
schedule_config = NoiseScheduleConfig()
normalizer_config = NormalizerConfig()

with open('config/generation_config.yaml', 'r') as f:
    gen_config = yaml.safe_load(f)

with open('config/eval_config.yaml', 'r') as f:
    eval_config = yaml.safe_load(f)['evaluation']

autoencoder = Autoencoder(
    num_tile_types=ae_config.num_tile_types,
    embedding_dim=ae_config.embedding_dim,
    latent_dim=ae_config.latent_dim,
    patch_height=ae_config.patch_height,
    patch_width=ae_config.patch_width
)
ae_checkpoint = torch.load(gen_config['models']['autoencoder_path'], map_location=device)
autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
autoencoder.to(device)
autoencoder.eval()
print("✓ Autoencoder loaded")

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
print("✓ Diffusion model loaded")

normalizer = LatentNormalizer(target_norm=normalizer_config.norm)
normalizer.load(gen_config['models']['normalizer_path'])

schedule = NoiseSchedule(
    num_timesteps=schedule_config.num_timesteps,
    schedule_type=schedule_config.schedule_type,
    device=device,
)

sampler = Sampler(unet, schedule, device=device)
parser = LevelParser()
difficulty_evaluator = PatchDifficultyEvaluator(parser)
performance_evaluator = ModelPerformanceEvaluator()


print("\n" + "="*70)
print("TEST 1: Difficulty Evaluation Comparison")
print("="*70)

diff_results = performance_evaluator.evaluate_difficulty_comparison(
    sampler=sampler,
    normalizer=normalizer,
    autoencoder=autoencoder,
    difficulty_evaluator=difficulty_evaluator,
    target_difficulties=eval_config['target_difficulties'],
    num_samples_per_target=eval_config['num_samples_per_target'],
    guidance_scale=eval_config['guidance_scale'],
    temperature=eval_config['temperature'],
    device=device,
    save_path=eval_config['difficulty_evaluation_path']
)

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print(f"Difficulty Evaluation plot saved to: {eval_config['difficulty_evaluation_path']}")
print(f"\nOverall MAE: {diff_results['overall']['mae']:.4f}")
print(f"Overall Correlation: {diff_results['overall']['correlation']:.4f}")
