# Zero Trust Satellite Research

Evaluates zero trust framework for satellite communications with post-quantum encryption using 40,000 virtualized tests.

## Setup

1. Create virtual environment:
   ```bash
   python -m venv zero_trust_env
   source zero_trust_env/bin/activate  # Linux
   zero_trust_env\Scripts\activate    # Windows
   ```

## Dependencies

1. Install Dependencies:
   ```bash
   pip install --upgrade pip
   pip install numpy
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch-geometric transformers stable-baselines3 liboqs-python pandas numpy networkx scapy scikit-learn gym
   ```

## Generate Data

1. Generate Data
   ```bash
   python scripts/generate_data.py
   
## Training

1. Training
   ```bash
   python scripts/train_gnn_network.py
   python scripts/train_distilbert_attack.py
   python scripts/train_rl_policy.py
   
## Testing
   ```bash
   python scripts/run_all_tests.py

## Analysis

1. Analysis

```bash
python scripts/analyze_results.py
```
```
