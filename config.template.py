# Configuration template file
# Copy this file to config.py and fill in your API keys

# Groq API Configuration
GROQ_API_KEYS = [
    "your-groq-api-key-here"  # Replace with your Groq API key
]
GROQ_MODEL = "mixtral-8x7b-32768"

# OpenRouter API Configuration
OPENROUTER_API_KEY = "your-openrouter-api-key-here"
OPENROUTER_MODEL = "anthropic/claude-3-sonnet"

# A4F API Configuration
A4F_API_KEY = "your-a4f-api-key-here"

# Hugging Face API Configuration
HUGGINGFACE_API_KEYS = [
    "your-huggingface-api-key-here"  # Replace with your Hugging Face API key
]

# Model Configurations
IMAGE_MODELS = [
    "dall-e-3",
    "dall-e-2",
    "stable-diffusion-xl",
    "stable-diffusion-2.1",
    "kandinsky-2.2",
    "deepfloyd-if"
]

HUGGINGFACE_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4"
]

# API Endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
IMAGE_MODELS_URL = "https://api.a4f.com/v1/models"
IMAGE_GENERATE_URL = "https://api.a4f.com/v1/images/generations" 