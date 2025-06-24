# Zero Trust Satellite Research with TPM/PQC Emulation

Evaluates zero trust framework for satellite communications with post-quantum encryption and TPM emulation using 40,000 virtualized tests.

## Setup

1. Install CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
2. Install cuDNN 8.9: https://developer.nvidia.com/cudnn-downloads
3. Install swTPM and TPM tools (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install swtpm tpm2-tools

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

The analysis script processes results from network simulations, attack detection, policy optimization, and cryptographic performance tests.

## Run Analysis

Execute the analysis script to generate summary reports:

```bash
python scripts/analyze_results.py
```

Notes
Requires RTX 5080 (CUDA 12.1), Ryzen 9 9950X3D, ~1–40 GB SSD space.

Monitor with nvidia-smi, htop.

Emulates TPM with swtpm and PQC with OQS (ML-KEM, ML-DSA).

If PyTorch fails, use Candle for attack detection: train_distilbert_attack_candle.py, run_attack_detection_candle.py.

Install CUDA 12.1, cuDNN 8.9, swtpm, tpm2-tools.

Set up virtual environment, install dependencies.

Configure project, generate data (python scripts/generate_data.py).

Set up TPM (python scripts/setup_tpm.py).

Train models (~1–24 hours): train_gnn_network.py, train_distilbert_attack.py, train_rl_policy.py.

Run tests (~10–75 minutes): python scripts/run_all_tests.py.

Analyze results: python scripts/analyze_results.py.

