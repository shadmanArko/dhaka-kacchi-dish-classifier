import torch
import transformers
import datasets

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)

if torch.backends.mps.is_available():
    print("Apple MPS backend available — we'll train on GPU (Metal)")
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA available:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("No GPU backend detected — will run on CPU (slower for fine-tuning)")
    device = torch.device("cpu")

print("Selected device:", device)

# quick sanity op on that device
x = torch.rand(3, 3).to(device)
y = x @ x
print("Test tensor op ran fine on:", y.device)