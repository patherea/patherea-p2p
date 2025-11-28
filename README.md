# Patherea-P2P

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) [![arXiv](https://img.shields.io/badge/arXiv-2412.16425-b31b1b.svg)](https://arxiv.org/abs/2412.16425) [![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ds2268/patherea)

Implementation of:
**Patherea: Cell Detection and Classification for the 2020s**

📚 Read the paper on arXiv: [https://arxiv.org/abs/2412.16425](https://arxiv.org/abs/2412.16425)

## 📊 Dataset

The Patherea dataset is available on Hugging Face Hub:

🔗 **[ds2268/patherea](https://huggingface.co/datasets/ds2268/patherea)**

## 🛠️ Installation

### 📁 Dataset
```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download ds2268/patherea --repo-type dataset --local-dir ./patherea_dataset
find patherea_dataset -name '*.zip' -exec unzip -o {} -d patherea_dataset \; -exec rm {} \;
```

### 🧠 Model
```bash
conda create --name patherea python=3.9
conda activate patherea
pip install poetry
poetry install
```

### ⚙️ MultiScaleDeformableAttention from Deformable-DETR
```bash
# Download and install CUDA 12.1 Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.105/local_installers/cuda_12.1.105_530.30.02_linux.run
sudo sh cuda_12.1.105_530.30.02_linux.run
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version

# Install MultiScaleDeformableAttention
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/model/ops
sh ./make.sh
```

## 🚀 Usage

To train and evaluate the model, run:

```bash
bash train_eval_patherea.sh
```

This script will train the model and evaluate it. The resulting trained model and annotated predicted images will be saved in:

```
src/outputs/epochs/
```

## 📄 Citation

If you use this code or dataset and find it helpful, please cite our work:

```bibtex
@article{STEPEC2026103868,
title = {Patherea: Cell detection and classification for the 2020s},
journal = {Medical Image Analysis},
volume = {108},
pages = {103868},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2025.103868},
url = {https://www.sciencedirect.com/science/article/pii/S1361841525004141},
author = {Dejan Štepec and Maja Jerše and Snežana Đokić and Jera Jeruc and Nina Zidar and Danijel Skočaj},
}
```
