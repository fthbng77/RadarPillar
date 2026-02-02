import torch
import torch.nn as nn


class PillarAttention(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = self.model_cfg.ATTN_CHANNELS
        # 1. DEĞİŞİKLİK: Makaleye göre kanal sayısını sabitliyoruz (Uniform Scaling)
        self.attn_channels = self.model_cfg.get('ATTN_CHANNELS', input_channels)
        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.get('DROPOUT', 0.0)
        
        # Giriş kanalı ile attention kanalı farklıysa tek bir lineer katman yeterli
        self.pre_mlp = nn.Linear(input_channels, self.attn_channels) if input_channels != self.attn_channels else nn.Identity()

        # 2. DEĞİŞİKLİK: MultiheadAttention - batch_first=True hızı artırır
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(self.attn_channels)
        
        # 3. DEĞİŞİKLİK: FFN yapısını makale önerisine göre (sabit genişlik) sadeleştirdik
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_channels, self.attn_channels * 2),
            nn.GELU(),
            nn.Linear(self.attn_channels * 2, self.attn_channels),
        )
        self.norm2 = nn.LayerNorm(self.attn_channels)


    def forward(self, batch_dict):
            pillar_features = batch_dict['pillar_features'] # (num_pillars, C)
            coords = batch_dict['voxel_coords']             # (num_pillars, 4) [batch_idx, z, y, x]
            
            batch_size = coords[:, 0].max().int().item() + 1
            
            # 4. DEĞİŞİKLİK: FOR DÖNGÜSÜNDEN KURTULMA VE MASKELİ PADDING SİSTEMİ
            # Her batch'teki pillar sayılarını buluyoruz
            pillar_counts = []
            for b in range(batch_size):
                pillar_counts.append((coords[:, 0] == b).sum().item())
            
            max_pillars = max(pillar_counts)
            
            # Boş şablonlar oluştur (Batch, Max_Pillar, Channels)
            padded_features = torch.zeros((batch_size, max_pillars, pillar_features.shape[-1]), 
                                        device=pillar_features.device)
            
            # Seyreklik Maskesi: True olan yerler 'boş' kabul edilir ve attention tarafından yok sayılır
            key_padding_mask = torch.ones((batch_size, max_pillars), dtype=torch.bool, 
                                        device=pillar_features.device)

            # Veriyi düz listeden batch formatına taşı (Bu işlem GPU'da hızlıdır)
            for b in range(batch_size):
                mask = coords[:, 0] == b
                num_p = pillar_counts[b]
                padded_features[b, :num_p] = pillar_features[mask]
                key_padding_mask[b, :num_p] = False # Dolu sütunları maskeden çıkar

            # 5. DEĞİŞİKLİK: ATTENTION İŞLEMİ (Tek Seferde)
            x = self.pre_mlp(padded_features)
            
            # key_padding_mask: Modelin sadece gerçek noktalara odaklanmasını sağlar
            attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
            x = self.norm1(x + attn_out)

            # Feed Forward Network
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            # 6. DEĞİŞİKLİK: SONUCU ESKİ FORMATA (LIST) GERİ DÖNDÜRME
            # Padding'leri atıp tekrar OpenPCDet'in beklediği (num_pillars, C) haline getiriyoruz
            updated_features = []
            for b in range(batch_size):
                updated_features.append(x[b, :pillar_counts[b]])
                
            batch_dict['pillar_features'] = torch.cat(updated_features, dim=0)
            return batch_dict
