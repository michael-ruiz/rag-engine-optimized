import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Configuration & Model Loading
# Using the text-only path or standard causal LM configuration
MODEL_ID = "Qwen/Qwen3.5-0.8B"

print(f"Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load in bfloat16 to optimize memory and replicate precise runtime conditions
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print(f"Model successfully loaded onto device: {model.device}\n")

# 2. Inspect Model Structure & Layer Tensors
print("=" * 60)
print("MODEL ARCHITECTURE & TENSOR SHAPE INSPECTION")
print("=" * 60)

total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    
    # Print a filtered subset of parameters (e.g., embeddings, first layer, final layers)
    if any(k in name for k in ["embed_tokens", "layers.0.", "layers.15.", "lm_head"]):
        print(f"Tensor Name: {name:<50} | Shape: {str(tuple(param.shape)):<20} | Dtype: {param.dtype}")

print(f"\nTotal Parameters: {total_params / 1e9:.2f} Billion")
print(f"Trainable Parameters: {trainable_params / 1e9:.2f} Billion\n")

# 3. Setting up a Forward Pass Activation Hook
# This lets you inspect internal activations (e.g., hidden states or MLP outputs) on the fly
activation_cache = {}

def get_activation_hook(layer_name):
    def hook(module, input, output):
        # output[0] typically contains the hidden states tensor
        activation_cache[layer_name] = output[0].detach().cpu()
    return hook

# Register a hook on a specific transformer layer (e.g., Layer 15, which was targeted in experimental weight patches)
target_layer_idx = 15
target_module = None

for name, module in model.named_modules():
    if f"layers.{target_layer_idx}" in name and "mlp" in name:
        target_module = module
        break

if target_module is not None:
    hook_handle = target_module.register_forward_hook(get_activation_hook(f"mlp_layer_{target_layer_idx}"))
    print(f"Successfully registered activation hook on MLP of layer {target_layer_idx}")
else:
    print(f"Could not find targeted layer module, hooking into general model output instead.")

# 4. Run a Sample Forward Pass
print("\nRunning a sample forward pass...")
prompt = "Calculate the value of x if 3x + 5 = 20. Please reason step by step:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(f"Input tokens shape: {inputs['input_ids'].shape}")
print(f"Output logits shape: {outputs.logits.shape}")

# Inspect captured hook activations if available
if f"mlp_layer_{target_layer_idx}" in activation_cache:
    captured_tensor = activation_cache[f"mlp_layer_{target_layer_idx}"]
    print(f"Captured activation tensor shape from Layer {target_layer_idx} MLP: {captured_tensor.shape}")

# Clean up hook
if target_module is not None:
    hook_handle.remove()

print("\nInspection script completed successfully!")