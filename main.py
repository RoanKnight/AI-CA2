import torch

def main():
  print("🔍 Checking GPU...")
  print(f"PyTorch version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")

  if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print("🎉 Ready for GPU-accelerated training!")
  else:
    print("❌ No GPU detected - using CPU")

if __name__ == "__main__":
  main()
