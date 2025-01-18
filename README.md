# AION-1 : AstronomIcal Omnimodal Network 

Polymathic's Large Omnimodal Model for Astronomy


## Installation on Rusty

Setting up your environment on Rusty
```bash
module load modules/2.3-20240529
module load gcc python/3.10.13
python -m venv --system-site-packages venv/aion
source venv/aion/bin/activate
pip install --upgrade eventlet
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```