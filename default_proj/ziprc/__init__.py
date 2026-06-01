"""ZIP-RC-Lite extension for Countdown (Qwen2.5-0.5B).

Stretch extension implementing a frozen-backbone ZIP-RC head (Manvi et al. 2025,
github.com/rohinmanvi/ZIP-RC) specialized to a structured 3-outcome verifier.

Light core modules (config, grid, alignment, verifier) are dependency-minimal
(stdlib + numpy) so M0 unit tests run without torch/transformers/vllm. Heavy
modules (gen_rollouts, label_rollouts, ziprc_dataset, train_head_only,
score_joint_head, value_select) require the full training stack and run on Modal.
"""
