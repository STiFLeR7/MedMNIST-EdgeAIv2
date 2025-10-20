import time, torch
m = torch.jit.load("D:\MedMNIST-EdgeAIv2\models\students\distilled_resnet18_ham10000\export\resnet18_student_traced.pt").to("cuda")
inp = torch.randn(1,3,224,224).to("cuda")
# warmup
for _ in range(20):
    _ = m(inp)
torch.cuda.synchronize()
# time
N = 200
t0 = time.time()
for _ in range(N):
    _ = m(inp)
torch.cuda.synchronize()
t = (time.time() - t0) / N
print(f"Avg GPU latency (ms): {t*1000:.2f}")
