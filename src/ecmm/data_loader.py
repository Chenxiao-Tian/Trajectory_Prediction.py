import pandas as pd


def load_network_topology(csv_topology):
    df = pd.read_csv(csv_topology)
    network_topology = df.values
    return network_topology


csv_file_path = "network_topology.csv"
network_topology = load_network_topology(csv_file_path)
