# AION - AstronomIcal Omnimodal Network 

Polymathic's Large Omnimodal Model for Astronomy


## Installation on Rusty

Setting up your environment on Rusty
```bash
module load modules/2.3-20240529
module load gcc python/3.10.13
python -m venv --system-site-packages venv/aion
source venv/mmoma/bin/activate
pip install --upgrade eventlet
pip3 install torch torchvision torchaudio
```