import subprocess
import time
import os

def validate_bb84():
    # Start qkd-net components for node1 and node2
    subprocess.Popen(["java", "-jar", "path/to/qkd-net.jar", "config_node1"])
    subprocess.Popen(["java", "-jar", "path/to/qkd-net.jar", "config_node2"])

    # Wait for key generation (adjust time as needed)
    time.sleep(60)

    # Define paths to key files
    node1_key_path = os.path.expanduser("~/.qkd/qnl/keys/node2/key_file")
    node2_key_path = os.path.expanduser("~/.qkd/qnl/keys/node1/key_file")

    # Read key files
    try:
        with open(node1_key_path, "r") as f:
            node1_key = f.read().strip()
        with open(node2_key_path, "r") as f:
            node2_key = f.read().strip()

        # Compare keys
        if node1_key == node2_key:
            print("Keys match: BB84 validation successful")
            return True
        else:
            print("Keys do not match: BB84 validation failed")
            return False
    except FileNotFoundError:
        print("Key files not found, validation failed")
        return False

if __name__ == "__main__":
    validate_bb84()