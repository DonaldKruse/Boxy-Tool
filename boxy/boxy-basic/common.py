"""
common.py

Holds shared configuration data:
- MODEL_DIR: the directory where model files live
- models: a list of available model names
- container_images: a mapping from model name to its container image reference
"""

# Directory (relative or absolute) where your models are stored
MODEL_DIR = "./models"

# List of model identifiers your application knows about
models = [
    # Llama 4 Scout Instruct, as shipped by Meta, fits on 4x 80 GB GPUs
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # Quantized version of Llama4 Scout Instruct, fits on 2x 80 GB GPUs
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    # Llama 3.1 405B Instruct, as shipped by Meta, fits on 16x 80 GB GPUs
    "meta-llama/Llama-3.1-405B-Instruct",
]

# Mapping from model name → container image URI
container_images = {
    "GIT_CONTAINER_IMAGE"    : "alpine/git:latest",
    "AWSCLI_CONTAINER_IMAGE" : "amazon/aws-cli:latest",
}


#    IMAGE_NAME="${REGISTRY}vllm/vllm-openai:v0.9.1"
#    TARGET="cuda"
#elif [ "$CLUSTER" = "eldorado" ]; then
#    IMAGE_NAME="${REGISTRY}rocm/vllm:rocm6.4.1_vllm_0.9.1_20250702"
#    TARGET="rocm"
