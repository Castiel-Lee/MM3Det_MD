# Requirements (tested)
- Linux (tested on Ubuntu 20.04)
- Python 3.7.9
- PyTorch 1.10.0
- CUDA 11.1
- [MMCV, MMDetection, MMSegmentation on UniBEV](https://github.com/tudelft-iv/UniBEV/blob/main/docs/installation.md)

# Setup
```bash
# Clone the repository
git clone [https://github.com/Castiel-Lee/MM3Det_MD](https://github.com/Castiel-Lee/MM3Det_MD)
cd MM3Det_MD

# Create environment and install dependencies
conda create -n mm3det_md python=3.7 -y
conda activate mm3det_md

# install requirements above

# install mmdet3d
pip install -v -e .  # or "python setup.py develop"