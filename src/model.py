import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GINConv, global_mean_pool, SAGPooling

class ComplexGraphEncoder(nn.Module):
    def __init__(self, node_dim=57, hidden_dim=256, n_layers=3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            )) for _ in range(n_layers)
        ])
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.pool = SAGPooling(hidden_dim, ratio=0.7)
        self.residual = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x_in = self.node_embed(x)
        x = x_in
        for conv in self.convs:
            x_res = x
            x = conv(x, edge_index)
            x = x + x_res
        x = self.norm(x)
        x = self.dropout(x)
        batch_before_pool = batch
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)
        x_shortcut = global_mean_pool(x_in, batch_before_pool)
        x = x + self.residual(x_shortcut)
        return x

class MoleculeEncoder(nn.Module):
    def __init__(self, node_dim=76, edge_dim=15, hidden_dim=256, n_layers=3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            )) for _ in range(n_layers)
        ])
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.pool = SAGPooling(hidden_dim, ratio=0.7)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        x = global_mean_pool(x, batch)
        return x

class MorganEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, hidden_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = self.residual(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = x + residual
        x = self.norm(x)
        return x

class DistanceEncoder(nn.Module):
    def __init__(self, input_dim=86, d_model=256, n_layers=3, n_heads=8):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.transformer(x).transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

class ImprovedProteinProjector(nn.Module):
    def __init__(self, protein_dim=1280, hidden_dim=256):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(protein_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
        )
        self.residual_path = nn.Linear(protein_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        main_out = self.main_path(x)
        residual_out = self.residual_path(x)
        out = main_out + residual_out
        out = self.norm(out)
        return out

class GatedFusion(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.interaction = nn.Sequential(
            nn.Linear(input_dim * 5, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 5, output_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(input_dim * 5, output_dim)
        self.modal_weights = nn.Sequential(nn.Linear(input_dim * 5, 5))

    def forward(self, *features):
        modal_features = torch.stack(features, dim=1)
        attn_output, _ = self.cross_attention(modal_features, modal_features, modal_features)
        concat = attn_output.reshape(attn_output.size(0), -1)

        raw_weights = self.modal_weights(concat)
        weights = F.softmax(raw_weights, dim=-1)
        weights = torch.clamp(weights, min=0.10, max=0.50)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        eps = 1e-8
        entropy = torch.sum(weights * torch.log(weights + eps), dim=-1).mean()

        stacked = torch.stack(features, dim=1)
        weighted_fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        interaction_out = self.interaction(concat)
        gate = self.gate(concat)
        transformed = self.transform(concat)
        fused = gate * transformed + weighted_fused + interaction_out

        return fused, entropy, weights

class FiveModalDTA(nn.Module):
    def __init__(self, protein_dim=1280, dist_dim=86, complex_graph_node_dim=57):
        super().__init__()
        common_dim = 256
        self.num_heads = 8

        self.proj_protein = ImprovedProteinProjector(protein_dim=protein_dim, hidden_dim=common_dim)
        self.ligand_encoder = MoleculeEncoder(node_dim=76, edge_dim=15, hidden_dim=common_dim, n_layers=3)
        self.morgan_encoder = MorganEncoder(input_dim=1024, hidden_dim=common_dim)
        self.dist_encoder = DistanceEncoder(input_dim=dist_dim, d_model=common_dim, n_heads=self.num_heads)
        self.complex_graph_encoder = ComplexGraphEncoder(node_dim=complex_graph_node_dim, hidden_dim=common_dim)

        self.local_attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=self.num_heads, dropout=0.1, batch_first=True)
        self.modal_embeddings = nn.Parameter(torch.randn(1, 5, common_dim))
        self.gated_fusion = GatedFusion(common_dim, common_dim, num_heads=self.num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, seq_embed, ligand_data, morgan_embed, dist_embed, complex_graph_data):
        protein_feat = self.proj_protein(seq_embed)
        ligand_feat = self.ligand_encoder(ligand_data.x, ligand_data.edge_index, ligand_data.edge_attr, ligand_data.batch)
        morgan_feat = self.morgan_encoder(morgan_embed)
        dist_feat = self.dist_encoder(dist_embed)
        complex_graph_feat = self.complex_graph_encoder(complex_graph_data.x, complex_graph_data.edge_index, complex_graph_data.batch)

        all_tokens = torch.stack([protein_feat, ligand_feat, morgan_feat, dist_feat, complex_graph_feat], dim=1)
        all_tokens = all_tokens + self.modal_embeddings
        local_output, _ = self.local_attention(all_tokens, all_tokens, all_tokens)

        fused, entropy_loss, weights = self.gated_fusion(protein_feat, ligand_feat, morgan_feat, dist_feat, complex_graph_feat)
        output = self.classifier(fused)

        return output, entropy_loss, weights