"""CPU-FineTuning: practical LLM pre-training and supervised fine-tuning.

ASCOT (Anchor-State COmpressed Training) enables 32k-context training steps on
2B-class hybrid LLMs at a small fraction of the compute cost of dense
backpropagation, by combining importance-sampled anchor spans, recurrent state
transplantation, layer-roulette slice updates and a calibrated local objective.
v13 keeps the CPU path, automatically uses CUDA/MPS when available, writes
browser-safe JSON, keeps training logs under logs/, exports Web UI settings as
a CLI command, keeps module future imports in a valid position for Linux,
loads remote-code models through their auto_map without interactive prompts,
and automatically combines multi-config Hugging Face datasets when no config
name is specified.
"""

__version__ = "13.0.0"
APP_NAME = "CPU-FineTuning"
