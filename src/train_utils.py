import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import torch_geometric
from torch_geometric.data import Batch
from .utils import c_index, RMSE, MAE, CORR, SD

def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    total_entropy_loss = 0
    outputs, targets = [], []
    scaler = GradScaler('cuda')
    optimizer.zero_grad()
    accum_steps = 1
    FIXED_ENTROPY_WEIGHT = 0.10

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        batch = {k: v.to(device) if isinstance(v, (torch.Tensor, Batch)) else v 
                 for k, v in batch.items()}

        with autocast('cuda'):
            out, entropy_loss, weights = model(
                batch['seq_embed'], batch['ligand_data'], batch['morgan_embed'],
                batch['dist_embed'], batch['complex_graph_data']
            )
            loss_main = criterion(out.flatten(), batch['pkd'])
            loss = loss_main + FIXED_ENTROPY_WEIGHT * entropy_loss
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_main.item() * accum_steps * len(batch['pdb_id'])
        total_entropy_loss += entropy_loss.item() * len(batch['pdb_id'])
        outputs.extend(out.detach().cpu().numpy().flatten())
        targets.extend(batch['pkd'].cpu().numpy())

    total_loss /= len(train_loader.dataset)
    total_entropy_loss /= len(train_loader.dataset)

    return {'loss': total_loss, 'entropy_loss': total_entropy_loss}


def evaluate(model, loader, criterion, split_name="valid", device=None, return_per_sample_weights=False):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_entropy_loss = 0.0
    outputs, targets, pdb_ids = [], [], []
    weight_sum = None
    total_samples = 0
    per_sample_data = [] if return_per_sample_weights else None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            batch = {k: v.to(device) if isinstance(v, (torch.Tensor, Batch)) else v 
                     for k, v in batch.items()}
            batch_size = len(batch['pdb_id'])

            with autocast('cuda'):
                out, entropy_loss, weights = model(
                    batch['seq_embed'], batch['ligand_data'], batch['morgan_embed'],
                    batch['dist_embed'], batch['complex_graph_data']
                )
                loss = criterion(out.flatten(), batch['pkd'])

            total_loss += loss.item() * batch_size
            total_entropy_loss += entropy_loss.item() * batch_size

            batch_outputs = out.cpu().numpy().flatten()
            batch_targets = batch['pkd'].cpu().numpy()
            batch_pdb_ids = batch['pdb_id']

            outputs.extend(batch_outputs)
            targets.extend(batch_targets)
            pdb_ids.extend(batch_pdb_ids)

            batch_weight_mean = weights.mean(dim=0)
            if weight_sum is None:
                weight_sum = batch_weight_mean * batch_size
            else:
                weight_sum += batch_weight_mean * batch_size
            total_samples += batch_size

            if return_per_sample_weights:
                batch_weights = weights.cpu().numpy()
                for i in range(batch_size):
                    per_sample_data.append({
                        'pdb_id': batch_pdb_ids[i],
                        'protein': float(batch_weights[i, 0]),
                        'ligand': float(batch_weights[i, 1]),
                        'morgan': float(batch_weights[i, 2]),
                        'distance': float(batch_weights[i, 3]),
                        'complex': float(batch_weights[i, 4]),
                        'true_pkd': float(batch_targets[i]),
                        'pred_pkd': float(batch_outputs[i])
                    })

    total_loss /= len(loader.dataset)
    total_entropy_loss /= len(loader.dataset)
    outputs = np.array(outputs)
    targets = np.array(targets)

    modal_weights = (weight_sum / total_samples).cpu().numpy() if total_samples > 0 else None

    result = {
        'loss': total_loss,
        'entropy_loss': total_entropy_loss,
        'c_index': c_index(targets, outputs),
        'RMSE': RMSE(targets, outputs),
        'MAE': MAE(targets, outputs),
        'CORR': CORR(targets, outputs),
        'SD': SD(targets, outputs),
        'outputs': outputs,
        'targets': targets,
        'pdb_ids': pdb_ids,
        'modal_weights': modal_weights
    }

    if return_per_sample_weights:
        result['per_sample_weights'] = per_sample_data

    return result