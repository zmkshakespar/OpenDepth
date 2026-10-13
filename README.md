# OpenDepth

Repurposing Image Editing Priors for Robust Open-World Monocular Depth Estimation.

ACM Multimedia 2026. Mengke Zhang (Project Leader), Tianyi Yang, Xinzhuo Li, and Anlong Ming.

**[Project page](https://zmkshakespar.github.io/OpenDepth/) · [Paper](https://zmkshakespar.github.io/OpenDepth/assets/OpenDepth.pdf)**

This first source release contains **training, inference, and evaluation**. It retains the research model and its required `modules/` and `library/` dependencies, without the image-editing UI, dataset archives, training logs, or model weights. The arXiv link is pending.

## Environment and model files

The validated inference environment is Linux, Python 3.10, PyTorch 2.3.1, CUDA 12.1, and a 48 GB GPU. Smaller GPUs may need `--offload`; this has not been validated as a memory guarantee. A CUDA GPU is required by the Step1X-Edit/Qwen backbone.

```bash
git clone https://github.com/zmkshakespar/OpenDepth.git
cd OpenDepth
python -m venv .venv
source .venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Optional: install a flash-attn wheel matching your Python, CUDA, and PyTorch.
```

Obtain the **v1.0** base files from [Step1X-Edit](https://huggingface.co/stepfun-ai/Step1X-Edit) and the Qwen encoder separately. Use this directory layout:

```text
models/step1x-edit/
  step1x-edit-i1258.safetensors
  vae.safetensors
  Qwen2.5-VL-7B-Instruct/
checkpoints/
  <your-opendepth-lora>.safetensors
```

LoRA checkpoint files are **not included in this source release**, and no public download URL is currently provided. Supply a local checkpoint path. Base models and datasets retain their own licenses.

## Training

`train.py` runs robust discriminative training with perturbed inputs, shallow feature alignment, and RGB latent reconstruction. The original trainer implementations are retained in `finetuning*.py`.

Edit `configs/dataset.example.toml` to point to your three prepared metadata files. Paths are resolved from the repository root. Each JSON object maps a target disparity file to its RGB reference:

```json
{
  "data/hypersim/target/frame.npy": {
    "ref_image_path": "data/hypersim/rgb/frame.jpg",
    "caption": ""
  }
}
```

For the robust training configuration, targets are `(H, W, 3)` float32 NPY arrays in `[-1, 1]`: **high values mean near**, low values mean far. The original dataset-specific preprocessing is not included in this release. RGB and target dimensions must match the dataset's configured resolution.

```bash
MODEL_DIR="$PWD/models/step1x-edit" \
DATA_CONFIG="$PWD/configs/dataset.example.toml" \
NUM_GPUS=1 OUTPUT_DIR="$PWD/outputs/training" \
bash scripts/train_robust.sh
```

The launcher uses a configurable GPU count, 8:1:1 dataset sampling, LoRA rank 64/alpha 32, and no server-specific paths or mandatory resume checkpoint. Its per-device batch size defaults to 1 in the example; tune batch size and accumulation for your hardware. These launch settings are a portable starting configuration, not a claim of full benchmark reproduction.

To inspect all original training options, run `python train.py --help`. Additional launcher arguments are forwarded to the trainer. Generative baseline training remains available through `finetuning_p.py`; it must use targets with the encoding appropriate to that checkpoint.

## Inference

The two checkpoint families use different inference paths and output conventions. Select both explicitly:

**Robust OpenDepth checkpoint: zero-initialized, single-step inference.**

```bash
python infer.py --model-dir models/step1x-edit \
  --checkpoint checkpoints/<robust-lora>.safetensors \
  --input path/to/rgb --output outputs/robust \
  --mode discriminative --encoding white-near
```

**Disparity epoch016 checkpoint used for the project-page qualitative images: generative flow matching, 10 steps, inverted disparity encoding.**

```bash
python infer.py --model-dir models/step1x-edit \
  --checkpoint checkpoints/step1x-edit_disparity_epoch016.safetensors \
  --input path/to/rgb --output outputs/epoch016 \
  --mode generative --steps 10 --encoding white-far
```

Inference accepts a file or recursively scans a directory. Each input gets its own output folder containing `encoded.png` (original grayscale model output), `disparity.npy`/`disparity.png` (canonical white = near), and `color.png`. `inference.json` records settings, checkpoint hash, and input hashes. NPY values come from the model's 8-bit decoded image; they are **relative disparity, not metric depth**. Color normalization is only for display.

`--attention auto` chooses available Flash Attention or PyTorch SDPA. Use `--attention torch` to avoid the optional Flash Attention dependency.

The VAE samples its encoding posterior. The inference CLI resets the PyTorch seed for each image; evaluation seeds the complete run. Keep the seed, sample order, batch size, software, and attention backend fixed for repeatability.

## Evaluation

`evaluate.py` retains the original dataset adapters and per-image alignment/cropping pipeline. It supports KITTI, NYUv2, Sintel, ETH3D, DIODE, ScanNet, nuScenes, RobotCar, DrivingStereo, KITTI-C, and DA-2K variants.

```bash
python evaluate.py --model_path models/step1x-edit \
  --lora_path checkpoints/<robust-lora>.safetensors \
  --dataset nyuv2 --dataset_dir path/to/nyuv2 \
  --output_dir outputs/eval/nyuv2 \
  --discriminative_mode --encoding white-near \
  --align_method disparity --crop_type eigen --batch_size 1
```

For epoch016, use `--use_depth_mode --num_steps 10 --encoding white-far` instead of `--discriminative_mode --encoding white-near`. Omitting both mode switches selects discriminative depth inference. Canonical predictions are saved in `<output_dir>/predictions`; metrics are saved in `results.json`.

For previously generated evaluation predictions, add `--skip_inference`; model weights are then unnecessary. These predictions must use the adapter's sample filenames and canonical white = near encoding. `infer.py` uses a different output layout and is not directly interchangeable with this cache. Use a **new output directory for each checkpoint/settings combination**, as the evaluator reuses existing predictions.

KITTI uses Garg cropping by default, NYUv2 uses Eigen cropping, and dataset-specific depth ranges are applied by the evaluator. Set alignment, crop, and depth bounds deliberately when comparing methods. The project-page benchmark numbers remain those reported in the supplied paper; the 10-step epoch016 visualizations are not a re-evaluation of those numbers.

Validation covers the three CLI entry points, launcher argument parsing, both inference modes on the authors' GPU, and the complete prediction/alignment/metric path on one NYUv2 sample. Both inference modes reproduced the original implementation's tested outputs exactly with matched inputs and random seeds. Full training and the full benchmark suite have not been rerun for this source release.

## Attribution and citation

This release derives from the authors' modified [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit) checkout. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_LICENSES/` for retained upstream notices.

```bibtex
@inproceedings{zhang2026opendepth,
  title={OpenDepth: Repurposing Image Editing Priors for Robust Open-World Monocular Depth Estimation},
  author={Zhang, Mengke and Yang, Tianyi and Li, Xinzhuo and Ming, Anlong},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3836413}
}
```
