import subprocess
import os
import shutil


def setup_tpm():
    """
    Initializes swTPM emulator, creating tpm/tpm_state/ directory and starting TPM socket.
    """
    tpm_dir = "../tpm/tpm_state"
    # Clear existing TPM state if present
    if os.path.exists(tpm_dir):
        shutil.rmtree(tpm_dir)
    os.makedirs(tpm_dir)

    # Command to start swTPM with TPM 2.0 support
    cmd = [
        "swtpm", "socket",
        "--tpmstate", f"dir={tpm_dir}",
        "--ctrl", f"type=unixio,path={tpm_dir}/swtpm-sock",
        "--tpm2"
    ]
    try:
        # Start swTPM in background
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"swTPM started at {tpm_dir}/swtpm-sock")
    except FileNotFoundError:
        raise Exception("swTPM not installed. Install with: sudo apt-get install swtpm")


if __name__ == "__main__":
    setup_tpm()