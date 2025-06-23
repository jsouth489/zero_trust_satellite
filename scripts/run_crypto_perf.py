from openquantumsafe import OQS_KEM
import pandas as pd
import time
from multiprocessing import Pool

def simulate_crypto(config):
    algorithm = config["algorithm"]
    key_size = config["key_size"]
    kem = OQS_KEM(algorithm)
    start = time.time()
    public_key, secret_key = kem.keypair()
    latency = time.time() - start
    return {"algorithm": algorithm, "key_size": key_size, "payload_size": config["payload_size"], "latency": latency}

def run_crypto_perf():
    dataset = pd.read_csv("../data/crypto_configs.csv")
    with Pool(processes=32) as pool:
        results = pool.map(simulate_crypto, [row.to_dict() for _, row in dataset.iterrows()])
    results_df = pd.DataFrame(results)
    results_df.to_csv("../results/crypto_results.csv", index=False)

if __name__ == "__main__":
    run_crypto_perf()