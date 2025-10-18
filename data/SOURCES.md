# Data & Model Provenance — MedMNIST-EdgeAIv2

This document records the source, license, checksum, and citation for every external asset used in this repository.  
All entries are verified against the checksums defined in `scripts/fetch_assets.py`.  
Each dataset and model is used strictly for academic research and benchmarking purposes.

---

## 🧬 Datasets

### 1. HAM10000 — Human Against Machine with 10,000 Training Images
- **Source:** [Tschandl et al., 2018 — The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://doi.org/10.1038/sdata.2018.161)
- **Download URL:** [https://zenodo.org/record/4269852/files/HAM10000_images_part_1.zip](https://zenodo.org/record/4269852)
- **License:** CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)
- **Folder:** `data/HAM10000/`
- **Checksum (MD5):** `d6f7ce2a23d7cbdf8a7d4cdbdbf8d0f3`
- **Citation:**
```

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data, 5*, 180161.

```

---

### 2. OCT2017 — Optical Coherence Tomography Retina Dataset
- **Source:** [Kermany et al., 2018 — Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://data.vision.ee.ethz.ch/cvl/aiims/)
- **Download URL:** [https://data.vision.ee.ethz.ch/cvl/aiims/retina/OCT2017.tar.gz](https://data.vision.ee.ethz.ch/cvl/aiims/)
- **License:** Research-only; redistribution not permitted without permission from authors.
- **Folder:** `data/OCT2017/`
- **Checksum (MD5):** `8efb3c4b421a1c1f8f417d15eb0a5539`
- **Citation:**
```

Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5), 1122–1131.

```

---

### 3. ISIC (Optional Cross-Domain Subset)
- **Source:** [ISIC 2019 Challenge Dataset](https://challenge.isic-archive.com/data/)
- **License:** CC BY-NC-SA 4.0
- **Folder:** `data/ISIC/`
- **Citation:**
```

Codella, N. C. F., et al. (2019). Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the International Skin Imaging Collaboration (ISIC). *arXiv:1902.03368*.

```

---

## 🧠 Models

### Teacher — ResNet-50 (HAM10000)
- **Source:** Pretrained checkpoint released under MedMNIST-EdgeAIv2 experiment.
- **Download URL:** [https://huggingface.co/MedMNIST/teachers/resolve/main/resnet50_ham10000.pth](https://huggingface.co/MedMNIST/teachers)
- **Checksum (MD5):** `9ab6533f458b7d9c8394e78c56b4c3fa`
- **Intended Use:** Teacher model for KD/AT experiments; not for clinical inference.

---

### Students — Knowledge-Distilled Variants
| Model | Base Arch | Download URL | Checksum (MD5) |
|:------|:-----------|:-------------|:----------------|
| Student-1 | ResNet-18 | [link](https://huggingface.co/MedMNIST/students/resolve/main/resnet18_kd_ham10000.pth) | `2ac4aaf0569c63e2c9b8fd7e8a5a51da` |
| Student-2 | MobileNetV2 | [link](https://huggingface.co/MedMNIST/students/resolve/main/mobilenetv2_kd_ham10000.pth) | `d59ec4a8c3f4fbc0e444a36b7e176c47` |
| Student-3 | EfficientNet-B0 | [link](https://huggingface.co/MedMNIST/students/resolve/main/efficientnetb0_kd_ham10000.pth) | `de425a58bcb4c678ab563e7d99e2c567` |

All student models are released under the same research-only license as the teacher.

---

## 🔏 Verification

Each asset’s integrity is verified via MD5 checksum on download by  
`scripts/fetch_assets.py`.  
If verification fails, the script will terminate and print a checksum mismatch warning.

For each dataset, raw data and preprocessing logs are stored in:
```

external_data/checkpoints/{dataset}/preprocess.log

```
For each model, metadata including training seed and configuration hash are stored in:
```

models/{teachers|students}/metadata.json

```

---

**Maintainer:** STiFLeR7  
**Last Verified:** 2025-10-18
```

