# Multi-Model Support

This update adds support for multiple local language models beyond TinyLlama:

## Added Models
- **TinyLlama 1.1B Chat**: Small but efficient model, perfect for testing and less demanding tasks
- **Llama 3 8B Instruct**: Powerful open source model from Meta, great balance between size and capability
- **Microsoft Phi-3 Mini**: Microsoft's efficient small language model with strong reasoning capabilities
- **Llama 2 7B Chat**: Meta's previous generation model, widely available and well-optimized

## How to Use
1. Download your preferred model:
```bash
# Download a model interactively (will prompt for selection)
python download_model.py

# Or specify a model directly
python download_model.py tinyllama
python download_model.py llama3
python download_model.py phi3
python download_model.py llama2

# Show help
python download_model.py --help
```

> **Note**: Some models like Llama 3 and Phi-3 may require HuggingFace login for access.
> To login, run: `huggingface-cli login`

2. Update your `.env` file to use the downloaded model:
```bash
# For TinyLlama
DEFAULT_MODEL=./src/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
MODEL_PROVIDER=huggingface

# For Llama 3
DEFAULT_MODEL=./src/models/Llama-3-8B-Instruct.Q4_K_M.gguf
MODEL_PROVIDER=huggingface

# For Phi-3
DEFAULT_MODEL=./src/models/phi-3-mini-4k-instruct-q4_k_m.gguf
MODEL_PROVIDER=huggingface

# For Llama 2
DEFAULT_MODEL=./src/models/llama-2-7b-chat.Q4_K_M.gguf
MODEL_PROVIDER=huggingface
```

3. Setup HuggingFace CLI tools (optional but recommended):
```bash
# Install HuggingFace CLI and login (recommended for accessing Llama 3 and Phi-3 models)
python setup_huggingface.py
```

4. Test your model:
```bash
# List available models
python test_models.py --list

# Test a specific model
python test_models.py --model ./src/models/your-model-file.gguf
```

5. Run the application:
```bash
streamlit run app.py
```

The application will automatically detect and list all downloaded GGUF models in the UI.
