import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import wandb

from configs.base import get_config
from configs.eval import get_config as get_eval_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging
from utils.utils import DummyWandb, print_banner, generate_run_name, get_lr


# Configure logging
setup_default_logging()
logger = logging.getLogger('justzoomin')


@torch.no_grad()
def run_evaluation(model, dataloader, criterion, device, sequence_length):
    """
    Runs a full evaluation pass on the provided dataloader.
    """
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in dataloader:
        # Move data to the correct device
        ground_images = batch["ground"].to(device)
        satellite_sequence = batch["satellite_sequence"].to(device)
        target_sequence = batch["sequence"].to(device)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                ground_images=ground_images,
                satellite_sequence=satellite_sequence,
                target_sequence=target_sequence
            )
            loss = criterion(logits.view(-1, logits.shape[-1]), target_sequence.view(-1))
        
        total_loss += loss.item()
        all_preds.append(torch.argmax(logits, dim=-1).cpu())
        all_targets.append(batch["sequence"])

    # Calculate Metrics 
    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    avg_loss = total_loss / len(dataloader)
    overall_accuracy = (predictions == targets).float().mean().item()

    # Create a dictionary of metrics to return
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": overall_accuracy
    }
    for i in range(sequence_length):
        step_accuracy = (predictions[:, i] == targets[:, i]).float().mean().item()
        metrics[f"val_accuracy_step_{i+1}"] = step_accuracy

    return metrics


def main():
    # Configuration and Initialization 
    print_banner()
    cfg = get_config()
    eval_cfg = get_eval_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high') # If your GPU supports it.
    # Wandb Setup
    if cfg.wandb.enable:
        run_name = generate_run_name(cfg)
        wandb.init(
            project=cfg.wandb.project_name,
            name=run_name,
            config=cfg.to_dict()
        )
        tracker = wandb
    else:
        tracker = DummyWandb()

    checkpoint_dir = Path(cfg.paths.checkpoint_dir + "/" + run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data Loading 
    train_transforms = build_transforms(cfg.data.target_image_size)
    train_dataset = ZoomDataset(cfg, transforms=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_transforms = build_transforms(eval_cfg.data.target_image_size)
    val_dataset = ZoomDataset(eval_cfg, transforms=val_transforms)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples.")
    logger.info(f"Loaded validation dataset with {len(val_dataset)} samples.")

    # Model, Optimizer, and Loss 
    model = GeoLocalizationModel(cfg)
    model.to(device)

    if cfg.training.compile:
        logger.info("Compiling the model... (this may take a minute)")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    if cfg.wandb.enable:
        tracker.watch(model, log="all", log_freq=100)

    # Training Loop
    best_val_accuracy = 0.0
    total_steps = 0
    logger.info("Starting training...")

    # Calculate total steps for the learning rate scheduler
    max_training_steps = len(train_dataloader) * cfg.training.num_epochs
    warmup_steps = int(max_training_steps * cfg.training.warmup_pct)

    for epoch in range(cfg.training.num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Update learning rate at each step based on the scheduler
            lr = get_lr(total_steps, cfg.training.learning_rate, warmup_steps, max_training_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            ground_images = batch["ground"].to(device)
            satellite_sequence = batch["satellite_sequence"].to(device)
            target_sequence = batch["sequence"].to(device)

            logits = model(
                ground_images=ground_images,
                satellite_sequence=satellite_sequence,
                target_sequence=target_sequence
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    ground_images=ground_images,
                    satellite_sequence=satellite_sequence,
                    target_sequence=target_sequence
                )
                loss = criterion(logits.view(-1, logits.shape[-1]), target_sequence.view(-1))
            
            # Scale loss and perform backward pass using the scaler
            scaler.scale(loss).backward()

            if cfg.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) 

            total_steps += 1
            loss_val = loss.item()
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

            if total_steps % cfg.training.log_interval == 0:
                tracker.log({"train_loss": loss_val, "step": total_steps, "epoch": epoch + 1})

        # --- End of Epoch Evaluation and Checkpointing ---
        if (epoch + 1) % cfg.training.eval_interval_epochs == 0:
            val_metrics = run_evaluation(model, val_dataloader, criterion, device, cfg.data.sequence_length)
            
            log_str = f"Epoch {epoch+1} | Val Accuracy: {val_metrics['val_accuracy']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}"
            logger.info(log_str)
            tracker.log(val_metrics)

            current_accuracy = val_metrics["val_accuracy"]
            if current_accuracy > best_val_accuracy:
                best_val_accuracy = current_accuracy
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save({'model_state_dict': model.state_dict()}, best_checkpoint_path)
                logger.info(f"New best model saved to {best_checkpoint_path} (Accuracy: {best_val_accuracy:.4f})")

        if (epoch + 1) % cfg.training.save_interval_epochs == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved periodic checkpoint to {checkpoint_path}")

    tracker.finish()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()


