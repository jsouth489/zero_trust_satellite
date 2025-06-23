import pandas as pd
import numpy as np
import networkx as nx
from scapy.all import Ether, IP

def generate_network_data(n_samples, filename, test=False):
    data = []
    for _ in range(n_samples):
        G = nx.random_geometric_graph(np.random.randint(100, 1001), 0.1)
        nodes = ','.join([str(i) for i in G.nodes])
        edges = ','.join([f"{u},{v}" for u, v in G.edges])
        traffic = np.random.uniform(1, 10)
        encryption = np.random.choice(['Kyber-512', 'Kyber-1024', 'Dilithium-2'])
        policy = np.random.choice(['1s_auth', '5s_auth', 'microseg'])
        if not test:
            latency = np.random.uniform(50, 500)
            throughput = np.random.uniform(0.5, 10)
            security = np.random.uniform(0.9, 1.0)
            data.append([nodes, edges, traffic, encryption, policy, latency, throughput, security])
        else:
            data.append([nodes, edges, traffic, encryption, policy])
    columns = ['nodes', 'edges', 'traffic', 'encryption', 'policy'] + (['latency', 'throughput', 'security'] if not test else [])
    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)

def generate_packet_data(n_samples, filename, test=False):
    data = []
    for _ in range(n_samples):
        pkt = Ether()/IP()
        sequence = str(pkt)
        label = np.random.randint(0, 2) if not test else None
        data.append([sequence, label] if not test else [sequence])
    columns = ['sequence'] + (['label'] if not test else [])
    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)

def generate_policy_data(n_samples, filename, test=False):
    data = []
    for _ in range(n_samples):
        state = ','.join([str(x) for x in np.random.rand(10)])
        action = np.random.randint(0, 5)
        reward = np.random.uniform(-10, 10) if not test else None
        data.append([state, action, reward] if not test else [state])
    columns = ['state'] + (['action', 'reward'] if not test else [])
    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)

def generate_crypto_data(n_samples, filename):
    data = []
    for _ in range(n_samples):
        algorithm = np.random.choice(['Kyber-512', 'Kyber-1024', 'Dilithium-2'])
        key_size = np.random.choice([512, 1024, 2048])
        payload_size = np.random.choice([1000, 10000, 100000])
        data.append([algorithm, key_size, payload_size])
    pd.DataFrame(data, columns=['algorithm', 'key_size', 'payload_size']).to_csv(filename, index=False)

if __name__ == "__main__":
    generate_network_data(100000, "data/network_training.csv")
    generate_network_data(40000, "data/network_configs.csv", test=True)
    generate_packet_data(100000, "data/packet_training.csv")
    generate_packet_data(40000, "data/packet_sequences.csv", test=True)
    generate_policy_data(100000, "data/policy_training.csv")
    generate_policy_data(4000, "data/policy_configs.csv", test=True)
    generate_crypto_data(40000, "data/crypto_configs.csv")