import torch

# Create tensors and move to MPS (Mac M2 GPU) or CUDA
device = torch.device(
   "mps" if torch.backends.mps.is_available() 
   else "cuda" if torch.cuda.is_available() 
   else "cpu"
)
tensor_a = torch.randn(1000, 1000).to(device)
tensor_b = torch.randn(1000, 1000).to(device)

# Perform operation on GPU
result_gpu = torch.matmul(tensor_a, tensor_b)

# Move result to CPU for comparison or further processing
result_cpu = result_gpu.cpu()
# result_cpu = result_gpu.to('cpu')

# Example: Compare with a CPU-based result
cpu_tensor_a, cpu_tensor_b = tensor_a.cpu(), tensor_b.cpu()
# cpu_tensor_a, cpu_tensor_b = tensor_a.to('cpu'), tensor_b.to('cpu')
result_cpu_reference = torch.matmul(cpu_tensor_a, cpu_tensor_b)
print("Results match:", torch.allclose(result_cpu, result_cpu_reference, atol=1e-5))