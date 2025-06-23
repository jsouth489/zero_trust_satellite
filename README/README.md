# Zero Trust Satellite Research

Evaluates zero trust framework for satellite communications with post-quantum encryption using 40,000 virtualized tests.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv zero_trust_env
   source zero_trust_env/bin/activate  # Linux
   zero_trust_env\Scripts\activate    # Windows
   
### Dependencies

2. Install Dependencies
```bash
    pip install --upgrade pip
    pip install numpy
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install torch-geometric transformers stable-baselines3 liboqs-python pandas numpy networkx scapy scikit-learn gym

