# ============================================================================
# HUGGING FACE AUTHENTICATION
# ============================================================================
from huggingface_hub import login, snapshot_download
import os

# Login with your token (set HF_TOKEN environment variable)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set. You may need to login manually with `huggingface-cli login`")

print("Logged in to Hugging Face!")
print("\n" + "="*80)
print("DOWNLOADING MODELS FOR HOUSING PROJECT")
print("="*80)

# Create Models directory if it doesn't exist
os.makedirs("/home/anshulk/Housing/models", exist_ok=True)

# ============================================================================
# DOWNLOAD QWEN MODEL
# ============================================================================
print("\n[1/2] Downloading Qwen3-4B-Instruct-2507 model...")
print("Size: ~8 GB | This may take a few minutes depending on your connection.")
print("-" * 80)

snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    local_dir="/home/anshulk/Housing/models/Qwen3-4B-Instruct-2507",
    local_dir_use_symlinks=False
)

print("✓ Qwen3-4B-Instruct-2507 downloaded successfully!")
print("  Location: /home/anshulk/Housing/models/Qwen3-4B-Instruct-2507")

# ============================================================================
# DOWNLOAD LLAMA MODEL
# ============================================================================
print("\n[2/2] Downloading Llama-3.2-3B-Instruct model...")
print("Size: ~13 GB | This may take a few minutes depending on your connection.")
print("-" * 80)

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir="/home/anshulk/Housing/models/Llama-3.2-3B-Instruct",
    local_dir_use_symlinks=False
)

print("✓ Llama-3.2-3B-Instruct downloaded successfully!")
print("  Location: /home/anshulk/Housing/models/Llama-3.2-3B-Instruct")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("="*80)
print("\nDownloaded Models:")
print("  1. Qwen3-4B-Instruct-2507     → models/Qwen3-4B-Instruct-2507")
print("  2. Llama-3.2-3B-Instruct      → models/Llama-3.2-3B-Instruct")
print("\nTotal size: ~21 GB")
print("\nYou can now use these models in your Housing project!")
print("="*80)
