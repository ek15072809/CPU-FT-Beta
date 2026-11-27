# CPU-FineTuning

> # EXPERIMENTAL SOFTWARE
>
> **This project is an experiment. Do NOT use it for production or for any model you depend on.**
>
> - It was built to explore whether LLM training is feasible on **CPU-only environments**.
> - Compared with genuine fine-tuning, the gradient estimates are **much noisier**, and training **may not improve model quality** (it can even make the model worse).
> - Validation cross-entropy (CE) may rise while the training loss falls. Always check the validation CE and compare generations before and after training.
> - Keep the original model. Treat every exported model as an experimental artifact.

**An LLM fine-tuning tool that runs on both CPU and GPU.**
Using its own fast training method, **ASCOT (Anchor-State COmpressed Training)**, it runs
**about 157 training steps per hour, each with a 16,384-token context**, on a 1.7B-class hybrid LLM
on a Windows laptop CPU with 16 GB RAM
(measured with LFM3-1.7B / i7-1360P / fp32; 32,768 tokens are also supported).

- Conventional LoRA / full fine-tuning is **not used** (neither can be accelerated on CPUs).
- Supports both PT (continued pre-training) and SFT (instruction tuning).
- Each step is designed so that the gradient estimate is correct *in expectation*, i.e. equivalent in expectation to one step of regular fine-tuning. Any individual step is far noisier than a regular step (see the warning above).

---

## Table of Contents

1. [Requirements and Performance Target](#requirements-and-performance-target)
2. [Main Features](#main-features)
3. [Supported Models and Architectures](#supported-models-and-architectures)
4. [Installation (Windows)](#installation-windows)
5. [Quick Start](#quick-start)
6. [Using Each Screen](#using-each-screen)
7. [Console Log During Training](#console-log-during-training)
8. [Checkpoints and Automatic Export](#checkpoints-and-automatic-export)
9. [Dataset Formats](#dataset-formats)
10. [Configuration Reference](#configuration-reference)
11. [Speed and Memory Guidelines](#speed-and-memory-guidelines)
12. [Overview of the ASCOT Method](#overview-of-the-ascot-method)
13. [Command Reference](#command-reference)
14. [Project Structure](#project-structure)
15. [Troubleshooting](#troubleshooting)
16. [License](#license)

---

## Requirements and Performance Target

| Item | Value |
|---|---|
| Intended OS | Windows 10/11 (also works on Linux/macOS) |
| CPU | About a 13th Gen Intel Core i7-1360P (4 P-cores + 8 E-cores) or better |
| Memory | 16 GB RAM |
| Models | Up to about 2B-class (roughly 1.7B–2.6B parameters; e.g. LFM2.5-2.6B / LFM3-1.7B / Gemma4-E2B / Qwen3.5-2B) |
| Context | Up to 32,768 tokens per step (processed in a single step, **not split into chunks**) |
| Speed target | **150 steps/hour or more** (16k context, 2B-class) |
| Measured | **156.6 steps/h** (LFM3-1.7B / 16,384 tokens / SFT / i7-1360P / fp32) |
| Training method | ASCOT (no LoRA / full FT) |

> **How to think about speed:** With conventional methods, a 32,768-token step is not realistically measurable.
> A full backprop of a 2B model takes hundreds of thousands of GFLOP per step,
> which takes tens of minutes per step on a laptop CPU. ASCOT compresses the actual compute of a step to
> **roughly the amount needed for inference at about 40 tokens/s for 24 seconds (about 900 tokens of forward-equivalent compute)**,
> while preserving the gradient direction in expectation.
> The measured value above is for a 1.7B model; figures for 2.6B-class models are extrapolations (see [Speed and Memory Guidelines](#speed-and-memory-guidelines)).

---

## Main Features

- **PT / SFT**: Continued pre-training on raw text, and instruction tuning with chat-template
  application and loss computed only on response tokens (assistant mask).
- **Multiple architectures**: LFM2 / LFM2.5 / LFM3 / Qwen3.5 / Gemma4 families.
  Models of the same architecture with a different parameter count are supported automatically (based on `layer_types` in the config).
- **Model/data input and output**: Both HuggingFace Hub and local folders are supported.
  When training finishes, **the full weights (HF format, safetensors) are exported automatically with no action required**.
  For the generation model you can type freely, or pick from suggestions of base models used in training and
  exported models, ordered by most recently used.
- **Before/after generation comparison**: The base model and the trained model are **never loaded at the same time**;
  they generate one after the other (to fit in 16 GB RAM).
- **Dataset builder**: Mixing of multiple sources (weighted), automatic format detection,
  32k packing, deduplication, length filtering, validation split, statistics, and preview.
- **Time estimates**: Rough times for dataset building and training are shown in the Web UI and the training log,
  based on model size, data amount and ASCOT settings. Local JSONL/JSON/CSV/
  TSV/Parquet/Arrow files, directories, quoted paths and first-N limits are counted by actual record count,
  and `max_sequences` and the validation ratio are taken into account.
- **Pre-validation of data settings**: Empty data sources are ignored in the UI; at API / training start,
  the run stops with a reason before loading the model. The watchdog does not retry configuration errors.
- **Web UI (Japanese/English)**: Configuration and dataset building are done in the browser. Beginner-friendly presets
  (speed / balanced / quality) and advanced detailed settings.
- **Console log**: During training, close the browser; the console shows loss, learning rate, gradient norm,
  memory, elapsed/remaining time and steps/h at a configurable interval.
- **Checkpoints**: Save interval, number kept and milestone saves are configurable.
  Emergency save on a crash and automatic resume by the watchdog.
- **Sleep prevention**: Prevents sleep/hibernation during training with Windows `SetThreadExecutionState`.
- **Self-contained in the project**: Models, datasets and caches are all stored inside the project folder under
  `models/`, `datasets/`, `state/` and so on. No caches are created outside it (e.g. in the user profile).
  Derived caches (int8-quantized copies) are managed and displayed so that they never exceed the original model size.

---

## Supported Models and Architectures

Detection is based on `model_type` / `layer_types` in `config.json`, so
**all models of the same architecture with a different parameter count are supported**
(e.g. future size variants of the LFM2 family, other sizes of Qwen3.5).

For multimodal models (gemma-4-E2B-it / Qwen3.5-2B, etc.), **only the text stack is trained**, and the
whole original structure is written out on export.

---

## Installation (Windows)

```bat
:: 1) Install Python 3.10 or later (3.11/3.12 recommended)
::    Check "Add python.exe to PATH" at https://www.python.org/downloads/

:: 2) Open a command prompt in the folder where you extracted this project and install the dependencies
cd CPU-FineTuning
pip install -r requirements.txt

:: 3) Start the Web UI (the browser opens automatically)
python run.py
```

The first run downloads about 1.5 GB (CPU-only PyTorch and others).

> **Uninstall:** Just delete the project folder.
> Nothing is written to the registry or the user profile.

---

## Quick Start

1. Run `python run.py` → the browser opens `http://127.0.0.1:8765/`
2. **Model** tab: Enter the 2B-class model of your choice → "Fetch info"
3. **Dataset** tab: Add a HuggingFace dataset ID (e.g. `HuggingFaceH4/no_robots`) →
   set "In order from the start" and "Long-document structure learning" if needed → "Build dataset"
4. **Training** tab: Preset "balanced" → leave total steps at `0`
   (follow the time/epoch limits; if all are 0, one epoch) → "Start training"
5. The Web UI shuts down automatically and the training log appears in a **new console** (you can close the browser meanwhile)
6. After training, the Web UI restarts automatically; check the loss graph, validation CE and export destination in the **Results** tab
   - The complete model has been exported automatically to `exports/<name>/` (no extra action needed)

> **Reminder:** This tool is experimental. After training, always check the validation CE and compare generations with the base model. The trained model may be worse than the original.

---

## Using Each Screen

### Model tab
- **Source**: A HuggingFace repository ID (`org/name`) or a local folder path (`C:\models\my-model`).
- **Fetch info**: Lets you check the structure (number of layers, vocabulary, layer types) without downloading the weights.
- **Compute precision**: `auto` (default; fp32, which is fast on Windows CPUs, when it fits in RAM),
  `bfloat16` (half the memory), `float32` (high precision, heavy RAM use).
- **Specifying the generation model**: For generation tests you can type freely, or choose from the models used for training and
  the exported models in `exports/`. Run names such as `run1` are resolved to a local export,
  and bare names that do not exist are not looked up on the Hub.

### Dataset tab
- **Add sources**: Multiple sources are allowed. Each row has:
  - Source (HF ID or local file/folder)
  - split (train / validation, etc.)
  - config name (e.g. `Math`). If empty, a single config is used automatically; if there are several, all configs are concatenated
  - Format (empty = auto-detect)
  - Mixing ratio (ratio between multiple sources)
  - First N records (partial use)
- **Data order**:
  - `In order from the start` (default): Nothing is shuffled globally; smaller data B is randomly mixed into the first N records of
    the largest data A. Validation data is taken from the end.
  - `Shuffle everything`: Random order. Structure-learning samples are not assigned to validation.
- **Long-document structure learning (general-purpose)**: Not HTML-specific. From long texts such as articles, Markdown, books,
  reports and code explanations (by default at least 1,200 characters and 4 sentences), a "document plan" is
  extracted automatically and a sample pairing it with the latter part of the document is generated.
  - Title, headings, numbered outline
  - Key facts that reappear later (selected by vocabulary overlap between early and later sentences)
  - Recurring proper nouns, technical terms, class/id names
  - For HTML, a summary of tags/classes/ids is also added (HTML is treated as one form among others)

  ASCOT uses the document plan as the prefix of later spans and additionally boosts the token ranges where words in the plan
  reappear in the later part. This trains the ability to finish a long text to the end, to refer to facts at distant positions, and
  the consistency of names, terminology, headings and formatting, all at once. The full text is also learned through
  normal packing, so plan-attached samples are an auxiliary curriculum.
  The compute per step does not change.
- **Build dataset**: Packs into 32k-token units and saves to `datasets/built/`.
  Shows statistics (number of sequences, total tokens, loss-target ratio, number of structure samples, front-mix information).
- **Estimated time**: The "Estimate" button shows a rough estimate from model size, data amount and settings.
- **Built datasets**: Lists and lets you select previously built caches.
  An explicit selection is saved in the settings; with "Auto", only the name determined from the current settings is used
  (a cache with the same name is reused if present, otherwise a new one is built).

### Training tab
- **Task**: SFT (instruction tuning) / PT (continued pre-training)
- **Presets**:
  | Preset | Purpose | Learning rate | Guideline (2B-class assumed) |
  |---|---|---|---|
  | `speed` | Fastest. Lightweight settings with fewer spans | `1e-5` | About 160–220 steps/h (extrapolated) |
  | `balanced` | Recommended. Strict global objective (default) | `5e-6` | **Measured 157 steps/h (16k / LFM3-1.7B)** |
  | `quality` | Highest quality (more spans and CE positions) | `2e-6` | About 90–110 steps/h (extrapolated) |
- **Total steps**: With `0`, the stop conditions apply. If the time limit and epoch limit are also 0, training ends after
  **exactly one epoch**. The epoch limit counts one pass over all training sequences as 1.
- **Learning rate**: The default is `5e-6` with a cosine scheduler. Earlier fixed `1e-4` worsened the
  validation CE on small datasets, so the default was lowered step by step. When the guard is set to "restart", the rate is automatically lowered
  to `1/2, 1/3, 1/4, …` on deterioration, and training stops at the lower limit of `1e-8`. For runs with a time limit,
  the LR horizon is corrected to match the measured speed.
- **Execution device**: The default `device="auto"` chooses in the order CUDA → MPS → CPU.
  To force CPU only, use `--device cpu` or select `CPU` in the UI.
- **Validation CE guard**: Choose one of "Restart / Stop / Off". Restart returns to the best point and
  lowers the LR; Stop ends training while keeping the best checkpoint; Off trains as is.
  The exponential multiplier can be edited only in exponential mode.
- **Advanced ASCOT settings** (for advanced users): Span length, number of spans, prefix length, sink length,
  slice size, mode (global = strict / local = fastest / auto = global), and so on.
- **Reproducibility**: The default `model_dtype="auto"` selects fp32 when it fits in RAM, and
  the int8 shadow is enabled only when a measured benchmark shows it is faster.
- **Logs and checkpoints**: Display interval, checkpoint interval and number kept, milestones,
  saving of optimizer state, and automatic export name.
- **Start training**: Pressing it ends the Web UI and starts training in a separate console.
  The start log shows "epoch n/N" and a rough time estimate.

### Results tab
- Summary (steps/h, loss trend, validation CE, device/precision, dataset,
  best/final checkpoint, export source).
- **Explanation of every metric**: The help icon next to each number shows its meaning, how it is calculated and caveats.
- **Main events**: Shows best updates, deteriorations, restarts with LR reduction, stops, final evaluation, export and
  graph generation in step order.
- **PNG graphs**: Learning rate, training loss (raw + EMA), gradient norm, validation CE (with best marker),
  and a 2×2 overview are shown in the Results screen.
- **Training log**: The end of the log can be viewed from the Results screen even after training finishes.
- **Generation test**: Send a prompt to the exported model and check the response.
  With comparison ON, the base model and the trained model are shown in the left and right panes (run one model at a time).

---

## Console Log During Training

During training, logs like the following appear in the command prompt (the display interval is configurable).
The same content is also saved in `logs/<run name>.log`:

```
Estimate: 266 steps × 23.48 s/step + 3.0 s data build = about 1:44:09
Epoch 1/1 | step 100/266 | loss 3.1696 (EMA 2.8446) | actual loss 3.2276 | lr 3.80e-06 | 24.1s/step | elapsed 40:10 | remaining 1:06:41
  slice [7,11) mode=global γ=1.000 grad norm=6.313 memory=2508MB
Validation CE: 3.1906 → 3.1874 (+0.0032)
Checkpoint saved: checkpoints\run1\step_000100
```

- **Epoch n/N**: Current epoch / number of epochs actually reached. If the total steps are fewer than
  the epoch limit, this is corrected to the number of epochs that will actually be reached.
- **loss**: Span-weighted loss (expected value = mean CE over the whole sequence)
- **EMA**: Exponential moving average of the loss (for checking the trend)
- **actual loss**: Actual CE of the current model, measured in a frozen pass
- **slice**: Layer range and mode trained in this step
- **γ**: Scale of the local calibration (in local mode)
- **memory**: Current RSS

`Ctrl+C` stops training safely (a checkpoint is saved immediately).
If the process terminates abnormally, the watchdog automatically resumes from the checkpoint.

---

## Checkpoints and Automatic Export

| Setting | Meaning |
|---|---|
| `checkpoint_every` | Save every n steps |
| `checkpoint_keep_last` | Keep the latest n (older ones are deleted automatically) |
| `checkpoint_keep_every` | Keep multiples of n steps as milestones, **never deleting them** |
| `checkpoint_optimizer` | Also save the 8-bit Adam state (full resume; more disk use) |
| `checkpoint_delta_dtype` | `int8` (default, smallest) / `bfloat16` / `float16` / `float32` |
| `guard_mode` | `restart` (default) / `stop` / `off` |
| `val_patience` / `val_min_delta` | Number of consecutive occurrences judged as deterioration / noise threshold (default `1` / `0.001`) |
| `export_best` | Export the checkpoint with the truly best CE (default ON) |
| `max_restarts` | Retry limit. `0` = unlimited down to the LR floor (default `0`) |
| `restart_lr_mode` | `harmonic` (1/2, 1/3, 1/4…, default) / `exponential` (multiplier) |
| `restart_lr_decay` | Multiplier for `exponential` (default `0.5`) |
| `restart_lr_floor` | Lower limit of the retry LR (default `1e-8`) |
| `save_plots` | Generate 5 PNG graphs at the end (default ON) |

- **The best-CE checkpoint is always kept**: Each time the validation CE truly improves, that exact
  step is saved in delta format and pinned in `best.json`. If there is no improvement, the pre-training
  baseline (step 0) is saved as the best.
- **At the end of training, with no export action required**, the full weights are written to `exports/<export name>/`
  (config / safetensors / tokenizer / chat template / training metadata).
  By default they are restored from the **best-CE checkpoint** and written, so the best model remains even if
  the final step got worse. An existing non-empty export destination is never overwritten; a timestamp is added instead.
- **Final checkpoint**: The last trained weights are also saved on normal termination. `latest` means
  "the weights trained last", and `best` means "the weights with the lowest validation CE".
- The exported folder can be used as is with `transformers` / `llama.cpp` (GGUF conversion), etc.

---

## Dataset Formats

### Supported file formats (local)
`.json` `.jsonl` `.csv` `.tsv` `.parquet` `.arrow` `.txt` `.md` (folders are read recursively)

**Quoted paths** such as `"C:\...\data.jsonl"` pasted via Windows "Copy as path" can be used as is (the quotes are stripped automatically). The same applies to local model paths.

### Supported data schemas (auto-detected)
| Schema | Example |
|---|---|
| `messages` (OpenAI/HF chat format) | `[{"role":"user","content":...},...]` |
| `conversations` (ShareGPT format) | `[{"from":"human","value":...},...]` |
| Alpaca format | `instruction` (+`input`) / `output` |
| prompt/response style | `prompt`/`response`, `question`/`answer`, `q`/`a`, `input`/`completion` |
| Preference format | `chosen`/`rejected` (the chosen side is used) |
| Raw text (for PT) | `text` / `content` / `body` / any other long string column |

For SFT, the tokenizer's chat template is applied and
**the loss mask covers only response (assistant) tokens**.
For templates without `{% generation %}` markers (e.g. Qwen family),
the response boundary (including the end-of-turn token = EOS) is restored exactly by prefix recursive rendering.

### Model loading and trust_remote_code
Repositories containing `auto_map` (such as LFM3) are loaded in three stages:
1. Normal `AutoConfig` (recent transformers support them natively)
2. Build with a native config class with `auto_map` removed (no code execution)
3. As a last resort, `trust_remote_code=True` (can be disabled with the environment variable `CPU_FT_TRUST_REMOTE_CODE=0`)

---

## Configuration Reference

Settings are managed in `runs/<run name>/config.json` (saved from the Web UI). Main items:

```jsonc
{
  "model_source": "LiquidAI/LFM2.5-2.6B",   // HF ID or local path
  "model_dtype": "auto",                     // auto | bfloat16 | float16 | float32
  "device": "auto",                          // auto | cpu | cuda | cuda:N | mps
  "task": "sft",                             // sft | pt
  "dataset_specs": [                         // Data sources (multiple allowed)
    // If name is omitted: a single config is used automatically; if several, all configs are concatenated
    {"source": "HuggingFaceH4/no_robots", "split": "train", "weight": 1.0}
  ],
  "dataset_cache": "",                       // Built cache name (empty = auto)
  "data": {
    "context_length": 32768,                 // Max tokens per step
    "examples_per_step": 0,                  // Examples packed per step (0 = fill automatically)
    "max_sequences": 4000,                   // Safety cap on the number of built sequences
    "val_ratio": 0.02, "dedup": true, "min_chars": 10,
    "dataset_order": "front",                // front (from the start; small data mixed at the front) | shuffle
    "long_doc_structure": true,              // Long-document plan learning, any format
    "long_doc_min_chars": 1200,              // Minimum number of characters to be eligible
    "long_doc_min_sentences": 4,             // Minimum number of sentences for documents with few headings
    "long_doc_max_docs": 512,                // Upper limit of generated long-document samples
    "long_doc_tail_chars": 16000,            // Number of characters learned as the latter part
    "long_doc_prefix_tokens": 64             // Max tokens of the document plan
  },
  "val_tokens": 4096,                        // Validation tokens. Evaluated in 2048-token windows (0 = all)
  "ascot": {                                 // Method parameters
    "span_length": 128,   // Length of an anchor span
    "num_spans": 4,       // Spans per step (more = higher quality, slower)
    "prefix_len": 48,     // Tokens used to compute the state just before a span
    "sink_len": 16,       // Tokens at the start of the document (attention sink)
    "slice_size": 4,      // Layers trained per step
    "mode": "global",     // global (strict, default) | local (fast, experimental) | auto
    "head_positions": 64, // Positions per span for which the head CE is computed
    "head_lr_scale": 0.10,        // Learning-rate scale of the tied head
    "slice_sampling": "balanced", // balanced (default) | uniform | top | adaptive
    "shadow_refresh_every": 4,    // Amortization interval for refreshing the int8 cache
    "uniform_floor": 0.25,        // Uniform floor of importance sampling
    "max_grad_norm": 1.0,
    "long_plan_tokens": 64        // Document-plan tokens added only to long-document samples
  },
  "lr": 5e-6, "weight_decay": 0.01,
  "scheduler": "cosine", "warmup_steps": 20, // Automatically capped at 10% of total steps
  "total_steps": 0, "eval_every": 20,        // 0 = follow the stop conditions
  "train_probe_tokens": 512,                 // Training CE probe (fixed token sequence, 0 = disabled)
  "guard_mode": "restart",                   // restart | stop | off
  "val_patience": 1, "val_min_delta": 0.001, // Consecutive count / noise threshold for judging deterioration
  "export_best": true,                       // Export the model with the truly best CE
  "restart_lr_mode": "harmonic",             // harmonic = 1/2, 1/3… / exponential = multiplier
  "restart_lr_decay": 0.5,                   // Multiplier for exponential
  "restart_lr_floor": 1e-8,                  // Lower limit of the retry LR
  "max_restarts": 0,                         // 0 = unlimited down to the floor
  "grad_spike_mode": "off",                  // off | skip | scale (mutually exclusive)
  "grad_spike_threshold_pct": 100,
  "grad_spike_window": 20,
  "max_hours": 0, "max_epochs": 1,           // Stop conditions (0 = disabled; if all are 0, one epoch)
  "checkpoint_every": 100, "checkpoint_keep_last": 2,
  "checkpoint_keep_every": 500, "checkpoint_optimizer": false,
  "checkpoint_delta_dtype": "int8",           // int8 | bfloat16 | float16 | float32
  "auto_export": true, "export_name": "finetuned",
  "save_plots": true,                         // Generate 5 PNG graphs
  "log_every": 1, "lang": "ja", "run_name": "run1"
}
```

---

## Speed and Memory Guidelines

### Learning-rate and validation-CE drift check (LFM3-1.7B / i7-1360P / 16GB RAM / Windows)

In earlier testing, 2 epochs / 70 steps on a small dataset showed training loss falling while the validation CE got worse.
The check was repeated with the same data and the same ASCOT settings, changing only the learning rate and schedule.

The cause was that `1e-4` was too high for 70 steps repeating a small dataset, and that
`total_steps=1e15` made the learning rate barely decay.
The defaults are now `5e-6` + cosine, and when `total_steps` is larger than the epoch limit, the
schedule horizon is corrected to the number of steps actually reached.

### Hardware measurement (LFM3-1.7B / i7-1360P / 16GB RAM / Windows)

| Configuration | Seconds/step | steps/h |
|---|---|---|
| SFT 16,384 ctx / 80 examples / lr 1e-4 / 10 steps / fp32 | **22.99** | **156.6** (steady state 158.5) |
| Dataset build (80 examples → 31 sequences / 383,609 tokens) | 4.0 s | — |

- Note: This measurement was a speed benchmark run with lr 1e-4. That is higher than the current default (`5e-6`) and is not recommended for real training.
- Validation CE: **1.9363 → 1.7520 (9.5% improvement)**; loss (mean of first 3 steps → mean of last 3 steps): 2.754 → 1.715.
- The exported weights were reloaded, and the same validation CE of 1.7520 and the same generated output were confirmed.
- `model_dtype="auto"` selected fp32 (1.73B parameters, about 6.9 GB of weights; machine RAM 16.8 GB). The int8 shadow was automatically disabled because it was measured to be slower.

Extrapolation to a 2.6B-class model (i7-1360P / 16GB / LFM2.5-2.6B). **These are estimates, not measurements:**

| Preset | Compute/step | Estimated sec/step | Estimated steps/h |
|---|---|---|---|
| speed | About 700 fwd-equivalent tokens | About 16–22 | **About 160–220 (150 target achievable)** |
| quality | About 2,300 fwd-equivalent tokens | About 32–40 | About 90–110 |

Memory (assuming a 2.6B model / 16GB RAM): bf16 master about 5.2 GB + int8 shadow about 2.6 GB + 8-bit Adam about 2.7 GB + activations ≈ **about 11 GB in total**
(for a 2.0B model: about 4.0 GB + 2.0 GB + 2.1 GB ≈ 9 GB). The derived int8 cache is controlled to stay **below** the size of the original model (always shown in the log).

---

## Overview of the ASCOT Method

1. **Importance-sampled anchor spans**: Scores the 32k tokens and trains only the spans in important regions.
   The weighting builds an **unbiased gradient estimator**.
2. **State transplantation**: The context just before a span is computed by a frozen int8 forward pass, and the conv/SSM/KV states are transplanted.
   Works especially well with the recurrent layers of hybrid architectures.
3. **Layer roulette**: Only a few consecutive layers are updated per step (block coordinate descent).
4. **Sparse embedding updates**: Only the vocabulary rows that appear are updated, preventing random-walk degradation of the whole vocabulary.
5. **Block-wise 8-bit Adam**: Memory-saving, non-diverging optimization plus bf16 stochastic rounding.

Because each step uses only a small, sampled part of the full computation, the gradient is noisy even though it is unbiased in expectation. This is why the tool is experimental.

---

## Command Reference

```bat
python run.py                                  :: Start the Web UI (recommended)
python run.py --lang en                        :: English UI
python -m cpu_finetuning serve --port 8765     :: Same as above (specify port)
python -m cpu_finetuning train --config runs\run1\config.json
                                               :: Training with watchdog (console)
python -m cpu_finetuning train --model <model> --dataset-cache <cache> --device auto
python -m cpu_finetuning train --config base.json --set ascot.num_spans=6 --set lr=3e-6
                                               :: Override any setting from the CLI (no Web UI needed)
python -m cpu_finetuning resume --config ...   :: Resume
python -m cpu_finetuning export --config ... --name myexport
                                               :: Explicit export
python -m cpu_finetuning info                  :: Show environment, GPU and caches
python -m cpu_finetuning generate --model exports\finetuned --prompt "Hello"
                                               :: Generation test
```

`--set` works on all fields, including `data.*` and `ascot.*`. Unknown keys are
rejected with a `ConfigError` before training starts. `python run.py` starts the Web UI.
Adding `--no-browser` to `serve` waits for connections without opening the browser.

With "Export CLI command" in the Web UI's Training tab, you can generate a PowerShell command that reproduces
the current settings exactly. The generated command uses `--%`
(the PowerShell stop-parsing token) to pass JSON arguments such as `--data-json` safely.
When saved as a `.ps1`, it is saved with a UTF-8 BOM so that Japanese paths are read correctly.

---

## Project Structure

```
CPU-FineTuning/
├── run.py                     # Entry point
├── requirements.txt
├── README.md / REPORT.md / DESIGN.md / CHANGELOG.md
├── config.example.json
├── cpu_finetuning/
│   ├── cli.py                 # CLI (serve/train/resume/export/info/generate)
│   ├── core/                  # Config, device, i18n, watchdog, path management
│   ├── models/                # Loader, delta checkpoints, full HF export
│   ├── data/                  # Multi-format loader, dataset builder (cache, val_seq_lengths)
│   ├── train/                 # ASCOT, 8-bit Adam, spike guard, checkpoints, PNG graphs
│   └── webui/                 # Web UI server + frontend (JA/EN, events, split generation)
│   * (generated automatically at runtime)
├── models/huggingface/        # Downloaded models (HF cache)
├── datasets/huggingface/      # Downloaded raw datasets
├── datasets/built/            # Built (packed) datasets
├── checkpoints/               # Training checkpoints
├── exports/                   # Automatically exported full weights
├── runs/                      # Per-run config, history, summary, graphs
├── logs/                      # Training logs (logs/<run name>.log)
├── state/                     # UI state
```

Legacy `.cache/huggingface` / `.cache/datasets` folders are migrated automatically to the new folders on first launch.

---

## Troubleshooting

| Symptom | Remedy |
|---|---|
| Crashes with out-of-memory | Check that `model_dtype="auto"` (default) is set. On CUDA, bf16-capable GPUs automatically use bf16, while T4 etc. use fp32. If memory is short, explicitly use `bfloat16`/`float16`, the `speed` preset, and set `shadow_refresh_every` to 4 |
| GPU is not used | Check CUDA/MPS detection with `python -m cpu_finetuning info`. Check that you have the CUDA build of PyTorch and that `device` is not `cpu` |
| Training is slow | Use the `speed` preset, reduce `num_spans`, set `shadow_refresh_every` to 4 |
| Dataset build is slow / seems stuck | int8 is benchmarked and automatically disabled if slow. The reason is shown in the `compute settings` log line. Progress is shown by phase, count and % |
| Validation CE gets worse in local mode | Use the default `mode="global"` (strict CE). Local is for fast experiments after calibration has progressed |
| You want higher quality | Use the `quality` preset, set `num_spans` to 8 or more, increase the number of epochs |
| Loss falls but validation CE gets worse | This is a known risk of this experimental method. Lower the learning rate (default is `balanced=5e-6`; `quality=2e-6` for stability), use cosine, and match `total_steps` to the actual amount of training. With `guard_mode="restart"` and `export_best=true`, the best-CE model is kept, and training retries from the best point with the LR lowered as `1/2, 1/3, 1/4…` (`restart_lr_floor` default `1e-8`). The events in the Results screen show whether a restart actually happened |
| Unsure how `total_steps=0` behaves | It follows the time and epoch limits. If both are 0, it is exactly one epoch |
| Generation keeps querying the Hub for `run1` | `run1` is resolved to `exports/`, and bare names that do not exist produce an error. Specify HF models as `owner/name` |
| Long-document structure samples are not created | Check that the target is at least `long_doc_min_chars` (default 1,200 characters), has at least `long_doc_min_sentences` (default 4 sentences) or has headings/line structure, and that `long_doc_structure` is ON. It is not HTML-only, so samples are also generated for Markdown, articles and reports |
| Too many long-document samples crowd out normal data | Lower `long_doc_max_docs`. This tool reserves at most 1/8 of `max_sequences` for long-document samples so that normal sequences are not starved |
| Out of memory in generation comparison | This tool runs one model at a time. If memory is still short, use `bfloat16` or a smaller model |
| Download fails | Check the proxy on a corporate network. A local path can also be specified |
| "Unknown format" in dataset build | Specify the format explicitly in the Dataset tab (e.g. `messages`) |
| The PC goes to sleep | Run training via the `train` command (sleep prevention is enabled) |
| You want to interrupt training | Press `Ctrl+C` in the console (a checkpoint is saved and training exits) |
| Disk is running out | Checkpoints are deltas (int8 by default). Delete old runs under `checkpoints/` and `models/huggingface/`, and unneeded `exports/`. `checkpoint_delta_dtype=float32` makes them larger |

---

## License

MIT License ([LICENSE](LICENSE)). This tool does not reuse code from other tools and is implemented from scratch,
depending only on Python / PyTorch / HuggingFace transformers.