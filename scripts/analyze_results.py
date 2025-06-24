import pandas as pd


def analyze_results():
    network_results = pd.read_csv("results/network_results.csv")
    print("Network Simulations Summary (with TPM/PQC):")
    print(network_results.describe())

    attack_results = pd.read_csv("results/attack_results.csv")
    accuracy = (attack_results['malicious_prob'] > 0.5).mean()
    print("Attack Detection Accuracy (with TPM logs):", accuracy)

    policy_results = pd.read_csv("results/policy_results.csv")
    print("Policy Optimization Actions Distribution (with TPM policies):")
    print(policy_results['action'].value_counts())

    crypto_results = pd.read_csv("results/crypto_results.csv")
    print("Cryptographic Performance by Algorithm and TPM Operation:")
    print(crypto_results.groupby(["algorithm", "tpm_operation"])["latency"].mean())


if __name__ == "__main__":
    analyze_results()