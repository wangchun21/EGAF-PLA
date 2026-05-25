import os
import re
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split, device, label_file, file_paths, train_stats=None):
        self.data_dir = data_dir
        self.split = split
        self.device = device
        self.label_file = label_file
        self.file_paths = file_paths[split]
        self.train_stats = train_stats

        self.labels = {}
        with open(label_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pdb_id = parts[0].strip().upper()
                    label_str = parts[1].strip()
                    try:
                        value = float(label_str.lstrip('=~>< '))
                        self.labels[pdb_id] = {'value': value}
                    except ValueError:
                        pass

        def clean_pdb_id(pdb_id):
            match = re.match(r'[0-9][A-Za-z0-9]{3}', str(pdb_id), re.IGNORECASE)
            return match.group(0).upper() if match else None

        complex_graph_file = self.file_paths['complex_graph']
        if not os.path.exists(complex_graph_file):
            raise FileNotFoundError(f"Missing {split} complex graph file: {complex_graph_file}")
        complex_graph_data = torch.load(complex_graph_file, weights_only=False, map_location='cpu')
        self.complex_graph_data = {clean_pdb_id(k): v for k, v in complex_graph_data.items() if clean_pdb_id(k)}

        ligand_data = torch.load(self.file_paths['ligand_graph'], weights_only=False, map_location='cpu')
        self.ligand_data = {clean_pdb_id(str(item['id']).upper()): item['graph_feature']
                            for item in ligand_data if clean_pdb_id(str(item['id']).upper())}

        morgan_data = torch.load(self.file_paths['morgan'], weights_only=False, map_location='cpu')
        self.morgan_data = {clean_pdb_id(str(item['id']).upper()): item['fingerprint'].float()
                            for item in morgan_data if clean_pdb_id(str(item['id']).upper())}

        seq_data = torch.load(self.file_paths['seq'], weights_only=False, map_location='cpu')
        self.seq_data = {clean_pdb_id(str(item['id']).upper()): item['seq_feature'].squeeze()
                         for item in seq_data if clean_pdb_id(str(item['id']).upper())}

        dist_data = torch.load(self.file_paths['distance'], map_location='cpu', weights_only=False)
        self.dist_features = {}
        for pdb_id, feat in zip(dist_data['ids'], dist_data['features']):
            cleaned_id = clean_pdb_id(str(pdb_id).upper())
            if cleaned_id and feat.size(0) > 0:
                self.dist_features[cleaned_id] = feat.float()

        if self.split == 'train':
            dist_features = torch.cat([v for v in self.dist_features.values()], dim=0)
            self.dist_feature_mean = dist_features.mean(dim=0)
            self.dist_feature_std = dist_features.std(dim=0) + 1e-8
        else:
            if train_stats is None:
                raise ValueError("Non-train splits require train statistics")
            self.dist_feature_mean, self.dist_feature_std = train_stats['dist_mean'], train_stats['dist_std']

        for k in self.dist_features:
            self.dist_features[k] = (self.dist_features[k] - self.dist_feature_mean) / self.dist_feature_std

        sets = [set(d.keys()) for d in [self.complex_graph_data, self.ligand_data,
                                        self.morgan_data, self.seq_data, self.dist_features, self.labels]]
        self.pdb_ids = list(set.intersection(*sets))

        if not self.pdb_ids:
            raise ValueError(f"No common PDB IDs found for {split} split")

        for d in [self.labels, self.complex_graph_data, self.ligand_data,
                  self.morgan_data, self.seq_data, self.dist_features]:
            for k in list(d.keys()):
                if k not in self.pdb_ids:
                    del d[k]

        if self.split == 'train':
            self.train_stats = {'dist_mean': self.dist_feature_mean, 'dist_std': self.dist_feature_std}

    def get_train_stats(self):
        return getattr(self, 'train_stats', None)

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        dist_embed = self.dist_features[pdb_id]
        morgan_embed = self.morgan_data[pdb_id]

        if self.split == 'train':
            noise_dist = torch.normal(mean=0, std=0.3, size=dist_embed.shape)
            dist_embed = dist_embed + noise_dist
            noise_morgan = torch.normal(mean=0, std=0.3, size=morgan_embed.shape)
            morgan_embed = morgan_embed + noise_morgan
            if random.random() < 0.2:
                dist_embed = torch.zeros_like(dist_embed)
            if random.random() < 0.2:
                morgan_embed = torch.zeros_like(morgan_embed)

        return {
            'pdb_id': pdb_id,
            'complex_graph_data': self.complex_graph_data[pdb_id],
            'ligand_data': self.ligand_data[pdb_id],
            'morgan_embed': morgan_embed,
            'seq_embed': self.seq_data[pdb_id],
            'dist_embed': dist_embed,
            'pkd': torch.tensor(float(self.labels[pdb_id]['value']), dtype=torch.float),
        }

def collate_fn(batch):
    seq_embed = torch.stack([item['seq_embed'] for item in batch])
    ligand_data = [item['ligand_data'] for item in batch]
    morgan_embed = torch.stack([item['morgan_embed'] for item in batch])
    dist_embed = torch.stack([item['dist_embed'] for item in batch])
    complex_graph_data = [item['complex_graph_data'] for item in batch]
    pkd = torch.stack([item['pkd'] for item in batch])

    ligand_batch = Batch.from_data_list(ligand_data)
    complex_graph_batch = Batch.from_data_list(complex_graph_data)

    return {
        'pdb_id': [item['pdb_id'] for item in batch],
        'seq_embed': seq_embed,
        'ligand_data': ligand_batch,
        'morgan_embed': morgan_embed,
        'dist_embed': dist_embed,
        'complex_graph_data': complex_graph_batch,
        'pkd': pkd,
    }