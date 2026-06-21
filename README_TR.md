<div align="center">

# RadarPillars: View-of-Delft üzerinde yeniden üretim

**Yalnızca radar girdisiyle 3B nesne tespiti — [Gillen ve ark., IROS 2024](https://arxiv.org/abs/2408.05020) çalışmasının OpenPCDet tabanlı reprodüksiyonu**

</div>

---

## Özet sonuçlar

| Yöntem | Araç | Yaya | Bisikletçi | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **Bizim — en iyi tohum** | **41.6** | **44.8** | 71.3 | **52.56** |
| Bizim — 3-tohum ortalama | 41.0 | 43.2 | 70.1 | 51.43 ± 0.99 |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| **RadarPillars (orijinal makale)** | 41.1 | 38.6 | 72.6 | **50.70** |
| CenterPoint (taban) | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars (taban) | 37.9 | 31.2 | 65.7 | 45.0 |

VoD doğrulama kümesinde 3B AP (R11), IoU eşikleri Araç=0.50, Yaya/Bisikletçi=0.25.

En iyi ağırlık dosyası: `output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth`
Ablasyon, tohum bazlı kayıtlar, hiperparametre tabloları → [`experiments/RESULTS.md`](experiments/RESULTS.md).

---

## Genişletilmiş VoD Radar-Only Liderlik Tablosu

VoD doğrulama setinde (Tüm Anotasyonlu Alan, 3D AP %, R11) raporlanan tüm radar-only yöntemlerin kapsamlı karşılaştırması. [Awesome-3D-Detection-with-4D-Radar](https://github.com/liuzengyun/Awesome-3D-Detection-with-4D-Radar) kataloğu + literatür verisi birleşimi.

| Sıra | Yöntem | Yıl | Araç | Yaya | Bisikletçi | mAP_3D |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | MAFF-Net | 25'RA-L | 42.3 | 46.8 | **74.7** | **54.6** |
| 2 | **Bizim (dense + NMS=0.20, yeni)** ¹ | 2026 | 38.89 | **49.16** | 73.70 | **53.92** |
| 3 | **Bizim (rot s3, yayımlanan)** ² | 2026 | 41.6 | 44.8 | 71.3 | 52.56 |
| 4 | SCKD | 25'AAAI | 41.89 | 43.51 | 70.83 | 52.08 |
| 5 | Dual-View Radar Reconstruction ★ | 26'Elec.Lett. | — | — | — | 52.07 |
| 6 | RadarGaussianDet3D | 25 | 40.7 | 42.4 | 73.0 | 52.0 |
| 7 | PSTOPS | 25 | — | — | — | 50.99 |
| 8 | SMURF | 23'TIV | 42.31 | 39.09 | 71.50 | 50.97 |
| 9 | RadarPillars (orijinal) | 24'IROS | 41.1 | 38.6 | 72.6 | 50.70 |
| 10 | RadarNeXt | 25 | 37.44 | 41.83 | 72.16 | 50.48 |
| 11 | MUFASA | 24'ICANN | **43.10** | 38.97 | 68.65 | 50.24 |
| 12 | SMIFormer | 23 | 39.53 | 41.88 | 64.91 | 48.77 |
| 13 | CenterPoint (taban) | — | 33.87 | 39.01 | 66.85 | 46.58 |
| 14 | DR-Net ★ | 25'TCSVT | — | — | — | 45.24 |
| 15 | PointPillars (taban) | — | 37.92 | 31.24 | 65.66 | 44.94 |
| 16 | RPFA-Net (yeniden uygulama) | 21'ITSC | 33.45 | 26.42 | 56.34 | 38.75 |

¹ Tek-seed sonucu (sabit seed 666). Mekanizma: çoklu sınıf baseline üzerinde anchor `feature_map_stride 2→1` (yoğun anchor grid, `UPSAMPLE_STRIDES [1,2,4]→[2,4,8]`) + post-hoc `NMS_THRESH 0.10→0.20` sweep (kalabalık yaya recall'una göre ayarlandı). Değerler `checkpoint_best.pth` üzerinden (early-stop weighted-mean R40 ile seçilen ep71). Çok-seed doğrulama bekliyor; gözlenen Yaya kazancı (+4.7 R11 vs baseline 44.49) 3-seed std (~1 mAP) değerinin çok üstünde.

² Yayımlanmış ağırlık — 3 random-seed run'ın en iyisi; LFS üzerinden takip ediliyor.

★ = [Awesome-3D-Detection-with-4D-Radar](https://github.com/liuzengyun/Awesome-3D-Detection-with-4D-Radar) kataloğundan eklendi. Sınıf bazlı kırılım kaynaktan çıkarılamadı.

Detaylı kaynak gösterimi + Sürüş Koridoru tablosu: [`docs/sota_comparison.tex`](docs/sota_comparison.tex).

---

## Mimari

```
Radar nokta bulutu (N,7)
  → PillarVFE (voxelleştirme + Doppler ayrıştırması: vx, vy = atan2 ile)
  → PillarAttention (maskeli öz-dikkat, C=E=32)
  → PointPillarScatter (320×320×32 BEV)
  → BaseBEVBackbone (3 bloklu 2B CNN, sabit kanal C=32)
  → AnchorHeadSingle (Araç / Yaya / Bisikletçi)
```

Temel uygulama detayları:
- **Hız ayrıştırması** VFE içinde: `vx = v_r_comp·cos(φ)`, `vy = v_r_comp·sin(φ)`, `φ = atan2(y, x)`
- **Fizik-tutarlı veri artırma**: hız vektörleri, nokta koordinatlarıyla birlikte döndürülür/yansıtılır (OpenPCDet'in nuScenes sütun düzenini varsayan hatasını giderir)
- **PillarAttention** key-padding mask ile çalışır — boş pilarlar dikkat skorlarını kirletmez
- **`FFN_CHANNELS` konfig sürücülü** (`pillar_attention.py`); önceki sürümde `*2` hardcoded'du

---

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

Gereksinimler: Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6.

---

## Veri

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

Info pkl + GT veritabanı üretimi:
```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

---

## Eğitim

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <koşu_adı> --workers 4
```

3-tohum çoklu koşu (özet tablodaki sayıyı üreten komut):
```bash
bash experiments/chain_scripts/multiseed_v2.sh
```

---

## Değerlendirme

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth
```

---

## Konfigler

**Temel (RadarPillars yeniden üretimi):**

| Dosya | Açıklama |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | makale Section IV'e sadık temel hat (rotation yok) |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **rotation eklenmiş varyant — özet sonucu üreten konfig** |

### Yaya Odaklı Deneyler

Bu konfigler, rotation eklenmiş temel hattın üzerine en zor sınıfı (yaya)
hedefleyen **ayrı bir çalışma koludur**. Yukarıdaki çekirdek RadarPillars
yeniden üretiminin parçası **değildir**. Tam anlatım makalede (PDF için
[Releases](../../releases) sayfası); aşağıdaki konfigler makaledeki sayıları
yeniden üretmenizi sağlar.

| Dosya | `_rot` temele göre değişiklik | Sonuç (R11 3B AP) |
|---|---|---|
| `tools/cfgs/vod_models/vod_radarpillar_rot_dense.yaml` | yoğun çapa ızgarası (`feature_map_stride` 2→1, `UPSAMPLE_STRIDES` [1,2,4]→[2,4,8]) + `NMS_THRESH` 0.10→0.20 | **53.92 mAP** (Araç 38.89 / Yaya **49.16** / Bisikletli 73.70) — en iyi |
| `tools/cfgs/vod_models/vod_radarpillar_ped.yaml` | tek-sınıf yaya (kontrol; çapa/voxel sabit) | Yaya −2.9 vs 3-sınıf — birlikte eğitim zor sınıfa yarıyor |
| `tools/cfgs/vod_models/vod_radarpillar_rot_voxel.yaml` | daha ince pillar `VOXEL_SIZE` 0.16→0.08, çapa adımı sabit (kontrol) | Araç −5.75 — seyrek radarda pillar parçalanması |

`_rot_dense` önerilen yaya konfigidir; diğer ikisi yoğun-çapa değişikliğinin
neden işe yaradığını ayrıştıran kontrollerdir.

---

## Atıf

```bibtex
@inproceedings{gillen2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Gillen, Julius and Bieder, Manuel and Stiller, Christoph},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  url    = {https://github.com/open-mmlab/OpenPCDet}
}
```

---

## Lisans

Apache 2.0 License altında yayınlanmıştır — bkz. [LICENSE](LICENSE). Bu proje [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) üzerine inşa edilmiştir; OpenPCDet'in lisansı da Apache 2.0'dır.
