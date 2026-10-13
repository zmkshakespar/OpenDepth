"""Run single-step OpenDepth or the generative disparity baseline on RGB images."""
import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True, help='Local LoRA safetensors file')
    p.add_argument('--input', type=Path, required=True, help='RGB file or directory (recursive)')
    p.add_argument('--output', type=Path, default=Path('outputs/inference'))
    p.add_argument('--mode', choices=['discriminative', 'generative'], default='discriminative')
    p.add_argument('--encoding', choices=['white-near', 'white-far'], required=True,
                   help='Encoding used by the selected checkpoint; epoch016 uses white-far')
    p.add_argument('--steps', type=int, default=10, help='Used only in generative mode')
    p.add_argument('--size-level', type=int, default=768)
    p.add_argument('--timestep', type=float, default=1.0, help='Discriminative timestep')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attention', choices=['auto', 'flash', 'torch'], default='auto')
    p.add_argument('--offload', action='store_true', help='Offload DiT/VAE between stages')
    args = p.parse_args()
    if args.steps < 1 or args.size_level < 256:
        p.error('--steps must be positive and --size-level must be at least 256')
    required = [args.model_dir / 'step1x-edit-i1258.safetensors',
                args.model_dir / 'vae.safetensors',
                args.model_dir / 'Qwen2.5-VL-7B-Instruct', args.checkpoint]
    for path in required:
        if not path.exists():
            p.error(f'Missing model resource: {path}')
    if not args.input.exists():
        p.error(f'Input does not exist: {args.input}')
    return args


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from matplotlib import colormaps
    from opendepth.engine import ImageGenerator

    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required by the Step1X-Edit/Qwen backbone.')
    attention = args.attention
    if attention == 'auto':
        from modules.attention import flash_attn_func
        attention = 'flash' if flash_attn_func is not None else 'torch'
    if attention == 'flash':
        from modules.attention import flash_attn_func
        if flash_attn_func is None:
            raise SystemExit('Flash Attention is unavailable; use --attention torch.')

    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    if args.input.is_file():
        files = [args.input]
        input_root = args.input.parent
    else:
        files = sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)
        input_root = args.input
    if not files:
        raise SystemExit('No RGB images found.')
    # Preserve directories and extensions so foo.jpg and foo.png never overwrite each other.
    generator = ImageGenerator(
        dit_path=str(args.model_dir / 'step1x-edit-i1258.safetensors'),
        ae_path=str(args.model_dir / 'vae.safetensors'),
        qwen2vl_model_path=str(args.model_dir / 'Qwen2.5-VL-7B-Instruct'),
        lora=str(args.checkpoint), max_length=640, version='v1.0',
        mode=attention, offload=args.offload,
    )
    records = []
    with torch.inference_mode():
        for path in files:
            # The VAE samples its posterior even for zero-initialized depth inference.
            torch.manual_seed(args.seed)
            image = Image.open(path).convert('RGB')
            if args.mode == 'discriminative':
                prediction = generator.generate_depth_discriminative(
                    image, seed=args.seed, size_level=args.size_level, prompt='',
                    timestep=args.timestep, show_progress=False,
                )
            else:
                prediction = generator.generate_depth(
                    image, num_steps=args.steps, seed=args.seed,
                    size_level=args.size_level, prompt='', show_progress=False,
                )
            encoded_image = prediction.convert('L').resize(image.size, Image.Resampling.LANCZOS)
            encoded = np.asarray(encoded_image, dtype=np.float32) / 255.0
            disparity = 1.0 - encoded if args.encoding == 'white-far' else encoded
            if not np.isfinite(disparity).all():
                raise ValueError(f'Non-finite prediction: {path}')
            low, high = np.percentile(disparity, [1, 99])
            normalized = np.clip((disparity - low) / max(high - low, 1e-8), 0, 1)
            color = (colormaps['Spectral_r'](1 - normalized)[..., :3] * 255).astype(np.uint8)
            relative = path.relative_to(input_root)
            dest = args.output / relative.parent / relative.name
            dest.mkdir(parents=True, exist_ok=True)
            encoded_image.save(dest / 'encoded.png')
            np.save(dest / 'disparity.npy', disparity)
            Image.fromarray(np.rint(disparity * 255).astype(np.uint8)).save(dest / 'disparity.png')
            Image.fromarray(color).save(dest / 'color.png')
            records.append({'input': str(relative), 'output': str(dest),
                            'input_sha256': sha256(path), 'width': image.width, 'height': image.height})
            print(f'Saved {relative} -> {dest}', flush=True)
    metadata = {
        'checkpoint_name': args.checkpoint.name, 'checkpoint_sha256': sha256(args.checkpoint),
        'mode': args.mode, 'steps': args.steps if args.mode == 'generative' else 1,
        'encoding': args.encoding, 'canonical_output': 'disparity: white=near, black=far',
        'seed': args.seed, 'size_level': args.size_level, 'attention': attention,
        'precision_note': 'NPY stores the 8-bit decoded grayscale prediction as float32; it is not metric depth.',
        'samples': records,
    }
    (args.output / 'inference.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
