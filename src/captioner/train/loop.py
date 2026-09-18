"""Generic training loop shared by stage 1 and stage 2. Must resume cleanly after kill -9 (§9
step 6 acceptance) — every `save_every_n_steps` we write a full checkpoint (middle.pt,
optimizer.pt, rng_state.pt, state.json) and `resume` reloads all of it, including RNG state, so
the resumed data order is reproducible.

Loss history is durable, not just printed: every optimizer step's train loss is appended as a
JSON line (`split="train"`) to `{run_dir}/metrics.jsonl` (main process only), independent of
checkpoint cadence, and every epoch additionally gets one `split="train_epoch"` (mean over that
epoch's steps) and one `split="val"` row sharing the same `epoch` value — read it with
`pd.read_json(path, lines=True)` and filter to those two splits for a clean one-point-per-epoch
curve, or plot the raw `split="train"` rows for per-step granularity. Durable even if the console
output that ran alongside it is gone.

Single-node multi-GPU (DDP, via `accelerate`) is supported directly — see
`captioner/README.md`'s cluster section. `accelerator.accumulate(model)` handles gradient
accumulation; only the main process writes checkpoints and logs, gated by
`accelerator.is_main_process`, with `wait_for_everyone()` around the save so no process races
ahead into the next epoch on a half-written checkpoint.

Checkpointing here assumes DDP — a full frozen-LLM replica per GPU, which needs no gradient/
optimizer sharding because the LLM's ~27B parameters are either frozen (stage 1) or adapted only
via small LoRA matrices (stage 2). FSDP-style parameter sharding is not implemented: it would
need `accelerator.get_state_dict(..., True)`-style full-state-dict gathering wired in for both
the model and the optimizer, which DDP doesn't require. Use DDP unless a single GPU can't hold a
full bf16 27B replica plus activations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from captioner.model.captioner import Captioner
from captioner.train.checkpoint import load_checkpoint, save_checkpoint
from captioner.utils.logging import get_logger
from captioner.utils.seeding import load_rng_state

logger = get_logger(__name__)


def _append_metric(run_dir: Path, record: dict) -> None:
    """Append one JSON line to `run_dir/metrics.jsonl` — the only durable record of loss values;
    `state.json` in each checkpoint tracks step/epoch/config but not loss history, and console
    output (tqdm/accelerator.print) is lost once the terminal scrolls or the session ends.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), **record}
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def _is_degenerate_batch(batch: dict) -> bool:
    """True iff every caption position in this micro-batch is padding — i.e. every sample's
    labels are entirely IGNORE_INDEX (see model/captioner.py). HF's CrossEntropyLoss computes a
    mean over all valid label positions in the batch; with zero valid positions that's 0/0 = NaN,
    which then poisons this step's gradients/Adam state for every trainable parameter (they're
    all shared across the batch). Confirmed real cause: `data/dataset.py`'s subset sampling used
    to consider a subset "available" purely from has_<modality> manifest flags, without checking
    whether a caption actually exists for that (object_id, subset) pair — e.g. spectra-tier
    captions only cover ~2,021 of the has_spectra=True objects. That's fixed at the source now
    (dataset.py only samples captioned subsets), but this is kept as a second line of defense —
    skip and warn rather than silently train on/report a NaN.
    """
    return not bool(batch["caption_attn_mask"].any())


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return candidates[-1] if candidates else None


def _save(
    accelerator: Accelerator,
    model: Captioner,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: DictConfig,
    run_dir: Path,
    tier_histogram: dict[str, int],
    lora_state_fn: Callable[[], dict] | None,
) -> None:
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    fusion_state = accelerator.get_state_dict(unwrapped.fusion_stack)
    lora_state = lora_state_fn() if lora_state_fn is not None else None

    if accelerator.is_main_process:
        save_checkpoint(
            run_dir, unwrapped.fusion_stack, optimizer, step, epoch, cfg, tier_histogram,
            lora_state, fusion_state_dict_override=fusion_state,
        )
    accelerator.wait_for_everyone()


def run_training(
    accelerator: Accelerator,
    model: Captioner,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: DictConfig,
    run_dir: Path,
    epochs: int,
    early_stop_patience: int,
    tier_histogram: dict[str, int],
    lora_state_fn: Callable[[], dict] | None = None,
) -> dict:
    step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    latest = find_latest_checkpoint(run_dir)
    if latest is not None:
        state = load_checkpoint(latest)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.fusion_stack.load_state_dict(
            torch.load(state["middle_path"], map_location="cpu", weights_only=False)
        )
        optimizer.load_state_dict(torch.load(state["optimizer_path"], map_location="cpu", weights_only=False))
        if accelerator.is_main_process:
            # rng_state.pt holds numpy's RNG state tuples, not on torch's weights_only allowlist —
            # safe here since these are our own checkpoint files, not third-party weights.
            load_rng_state(torch.load(state["rng_path"], map_location="cpu", weights_only=False))
        step = state["step"]
        start_epoch = state["epoch"]
        accelerator.print(f"Resumed from {latest} at step={step} epoch={start_epoch}")

    accelerator.print(f"Starting training: epochs={epochs}, batches/epoch={len(train_loader)}")

    for epoch in range(start_epoch, epochs):
        # Re-roll each object's modality subset and (system, instruction) draw. Without this the
        # draw is fixed for the whole run — see CaptionerDataset.set_epoch.
        base_ds = getattr(train_loader, "dataset", None)
        if hasattr(base_ds, "set_epoch"):
            base_ds.set_epoch(epoch)
        model.train()
        # `disable=` rather than skipping the wrap entirely: on non-main ranks tqdm still needs to
        # exist as a plain no-op iterator wrapper, since the `for batch in ...` below is unchanged
        # either way — only its console output is suppressed, avoiding 4 interleaved bars in DDP.
        pbar = tqdm(
            train_loader, desc=f"epoch {epoch}", disable=not accelerator.is_main_process, leave=False
        )
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for batch in pbar:
            if _is_degenerate_batch(batch):
                accelerator.print(
                    f"Skipping degenerate micro-batch (no valid caption tokens): "
                    f"object_id={batch['object_id']} shown={batch['shown']}"
                )
                continue
            with accelerator.accumulate(model):
                out = model(
                    batch["modality_batch"],
                    batch["pre_ids"],
                    batch["post_ids"],
                    batch["caption_ids"],
                    batch["pre_attn_mask"],
                    batch["post_attn_mask"],
                    batch["caption_attn_mask"],
                )
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                loss_value = out.loss.item()
                epoch_loss_sum += loss_value
                epoch_loss_count += 1
                pbar.set_postfix(step=step, loss=f"{loss_value:.4f}")
                if accelerator.is_main_process:
                    _append_metric(run_dir, {"split": "train", "epoch": epoch, "step": step, "loss": loss_value})
                save_every = cfg.get("checkpoint", {}).get("save_every_n_steps", 500)
                if step % save_every == 0:
                    accelerator.print(f"step={step} loss={loss_value:.4f} — checkpointing")
                    _save(accelerator, model, optimizer, step, epoch, cfg, run_dir / f"step_{step}", tier_histogram, lora_state_fn)

        train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        val_loss = evaluate_loss(accelerator, model, val_loader)
        accelerator.print(f"epoch={epoch} step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if accelerator.is_main_process:
            # One `train_epoch`/`val` pair per epoch, on top of the per-step `train` records above —
            # filtering metrics.jsonl to split=="train_epoch" or split=="val" gives a clean,
            # one-point-per-epoch series to plot without needing to average the per-step rows first.
            _append_metric(run_dir, {"split": "train_epoch", "epoch": epoch, "step": step, "loss": train_loss})
            _append_metric(run_dir, {"split": "val", "epoch": epoch, "step": step, "loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _save(accelerator, model, optimizer, step, epoch, cfg, run_dir / "best", tier_histogram, lora_state_fn)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                accelerator.print(f"Early stopping at epoch={epoch} (patience={early_stop_patience})")
                break

    return {"best_val_loss": best_val_loss, "final_step": step}


@torch.no_grad()
def evaluate_loss(accelerator: Accelerator, model: Captioner, loader: DataLoader) -> float:
    model.eval()
    total_loss = torch.zeros(1, device=accelerator.device)
    total_n = torch.zeros(1, device=accelerator.device)
    for batch in loader:
        if _is_degenerate_batch(batch):
            accelerator.print(
                f"Skipping degenerate val micro-batch (no valid caption tokens): "
                f"object_id={batch['object_id']} shown={batch['shown']}"
            )
            continue
        out = model(
            batch["modality_batch"],
            batch["pre_ids"],
            batch["post_ids"],
            batch["caption_ids"],
            batch["pre_attn_mask"],
            batch["post_attn_mask"],
            batch["caption_attn_mask"],
        )
        bs = batch["pre_ids"].shape[0]
        total_loss += out.loss.detach() * bs
        total_n += bs
    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_n = accelerator.reduce(total_n, reduction="sum")
    return (total_loss / total_n.clamp(min=1)).item()
