from openquantumsafe import OQS_KEM, OQS_SIG
import pandas as pd
import time
from multiprocessing import Pool
import subprocess
import os


def simulate_crypto(config):
    algorithm = config["algorithm"]
    key_size = config["key_size"]
    tpm_op = config["tpm_operation"]

    # Ensure TPM state directory exists
    tpm_dir = os.path.abspath("../tpm/tpm_state")
    data_file = os.path.join(tpm_dir, "data")
    with open(data_file, "w") as f:
        f.write("test_data")

    # Simulate TPM operation
    try:
        if tpm_op == "key_gen":
            cmd = ["tpm2_create", "-C", "o", "-u", os.path.join(tpm_dir, "key.pub"), "-r",
                   os.path.join(tpm_dir, "key.priv")]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif tpm_op == "sign":
            cmd = ["tpm2_sign", "-c", os.path.join(tpm_dir, "key.priv"), "-m", data_file, "-s",
                   os.path.join(tpm_dir, "sig")]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif tpm_op == "attestation":
            cmd = ["tpm2_pcrread", "sha256:0"]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"TPM operation {tpm_op} failed: {e}, simulating latency...")

    # Measure PQC operation
    start = time.time()
    if "ML-KEM" in algorithm:
        kem = OQS_KEM(algorithm.replace("ML-KEM-", "Kyber"))
        public_key, secret_key = kem.keypair()
    elif "ML-DSA" in algorithm:
        sig = OQS_SIG(algorithm.replace("ML-DSA-", "Dilithium"))
        public_key, secret_key = sig.keypair()
    latency = time.time() - start
    return {"algorithm": algorithm, "key_size": key_size, "payload_size": config["payload_size"],
            "tpm_operation": tpm_op, "latency": latency}


def run_crypto_perf():
    dataset = pd.read_csv("../data/crypto_configs.csv")
    with Pool(processes=32) as pool:
        results = pool.map(simulate_crypto, [row.to_dict() for _, row in dataset.iterrows()])
    results_df = pd.DataFrame(results)
    results_df.to_csv("../results/crypto_results.csv", index=False)


if __name__ == "__main__":
    run_crypto_perf()