import torch
import logging
import pandas as pd
from pathlib import Path

from src.config import *
from src.utils import set_seed, get_cosine_schedule_with_warmup
from src.dataset import MultimodalDataset, collate_fn
from src.model import FiveModalDTA
from src.train_utils import train, evaluate


def main():
    set_seed(SEED)
    device = torch.device(DEVICE)
    if device.type == 'cpu':
        raise RuntimeError("CUDA is required for training.")

    # File paths
    file_paths = {
        split: {
            'complex_graph': DATA_DIR / f"{split}_complex_graph.pt",
            'ligand_graph': DATA_DIR / f"{split}_ligand_graph.pt",
            'seq': DATA_DIR / f"{split}_seq.pt",
            'distance': DATA_DIR / f"{split}_distance.pt",
            'morgan': DATA_DIR / f"{split}_ligand_morgan.pt",
        }
        for split in ['train', 'valid', 'test']
    }

    output_dir = BASE_DIR / 'results' / f"results{TIMESTAMP}"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_sample_weights_dir = output_dir / "best_sample_weights"
    best_sample_weights_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Datasets
    train_dataset = MultimodalDataset(DATA_DIR, 'train', device, LABEL_FILE, file_paths)
    train_stats = train_dataset.get_train_stats()
    valid_dataset = MultimodalDataset(DATA_DIR, 'valid', device, LABEL_FILE, file_paths, train_stats)
    test_dataset = MultimodalDataset(DATA_DIR, 'test', device, LABEL_FILE, file_paths, train_stats)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    # Model
    model = FiveModalDTA(
        protein_dim=1280,
        dist_dim=86,
        complex_graph_node_dim=57
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        WARMUP_EPOCHS,
        NUM_EPOCHS
    )

    best_val_corr = -float('inf')
    best_val_loss_for_corr = float('inf')
    best_epoch_corr = -1
    best_model_path_corr = output_dir / "best_model.pth"

    logger.info("=== Training Started ===")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train(model, train_loader, optimizer, criterion, epoch, device)
        val_metrics = evaluate(model, val_loader, criterion, "valid", device)
        test_metrics = evaluate(model, test_loader, criterion, "test", device)

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}")

        current_val_corr = val_metrics['CORR']
        current_val_loss = val_metrics['loss']

        updated_best_corr = False

        if epoch >= 110:
            is_better_corr = (
                current_val_corr > best_val_corr
                or (
                    abs(current_val_corr - best_val_corr) < 1e-6
                    and current_val_loss < best_val_loss_for_corr
                )
            )

            if is_better_corr:
                best_val_corr = current_val_corr
                best_val_loss_for_corr = current_val_loss
                best_epoch_corr = epoch

                torch.save(model.state_dict(), best_model_path_corr)
                updated_best_corr = True

        if updated_best_corr:
            for loader_name, loader in [
                ("train", train_loader),
                ("valid", val_loader),
                ("test", test_loader)
            ]:
                detail = evaluate(
                    model,
                    loader,
                    criterion,
                    loader_name,
                    device,
                    return_per_sample_weights=True
                )

                if 'per_sample_weights' in detail:
                    df = pd.DataFrame(detail['per_sample_weights'])
                    df = df[
                        [
                            'pdb_id',
                            'protein',
                            'ligand',
                            'morgan',
                            'distance',
                            'complex',
                            'true_pkd',
                            'pred_pkd'
                        ]
                    ]

                    save_path = best_sample_weights_dir / f"{loader_name}_weights_best.csv"
                    df.to_csv(save_path, index=False, float_format="%.4f")


        scheduler.step()

    logger.info("=== Training Completed ===")

    if best_epoch_corr != -1:
        model.load_state_dict(
            torch.load(
                best_model_path_corr,
                map_location=device,
                weights_only=True
            )
        )
        logger.info(
            f"Loaded best model "
        )

    logger.info("Starting final evaluation on all splits")

    final_train = evaluate(model, train_loader, criterion, "train_final", device)
    final_val = evaluate(model, val_loader, criterion, "valid_final", device)
    final_test = evaluate(model, test_loader, criterion, "test_final", device)

    logger.info("=== FINAL RESULTS ===")

    for split_name, metrics in [
        ("train", final_train),
        ("valid", final_val),
        ("test", final_test)
    ]:
        logger.info(f"\n{split_name.upper()} SET METRICS:")
        logger.info(f"  Loss:     {metrics['loss']:.4f}")
        logger.info(f"  c_index:  {metrics['c_index']:.4f}")
        logger.info(f"  RMSE:     {metrics['RMSE']:.4f}")
        logger.info(f"  MAE:      {metrics['MAE']:.4f}")
        logger.info(f"  CORR:     {metrics['CORR']:.4f}")
        logger.info(f"  SD:       {metrics['SD']:.4f}")

    for split_name, metrics in [
        ("train", final_train),
        ("valid", final_val),
        ("test", final_test)
    ]:
        df = pd.DataFrame({
            'ID': metrics['pdb_ids'],
            'True_Affinity': metrics['targets'],
            'Predicted_Affinity': metrics['outputs']
        })

        csv_path = output_dir / f"predictions_{split_name}.csv"
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()