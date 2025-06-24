import subprocess

def run_all_tests():
    print("Setting up swTPM...")
    subprocess.run(["python", "scripts/setup_tpm.py"])
    print("Running network simulations...")
    subprocess.run(["python", "scripts/run_network_sim.py"])
    print("Running attack detection...")
    subprocess.run(["python", "scripts/run_attack_detection.py"])
    print("Running policy optimization...")
    subprocess.run(["python", "scripts/run_policy_opt.py"])
    print("Running cryptographic performance...")
    subprocess.run(["python", "scripts/run_crypto_perf.py"])

if __name__ == "__main__":
    run_all_tests()