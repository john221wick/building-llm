import torch
from SelfAttention import SelfAttentionV1, selfAttentionV2
x = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0]],
    [[5.0, 6.0, 7.0, 8.0]],
    [[9.0, 0.0, 1.0, 2.0]],
    [[3.0, 4.0, 5.0, 6.0]]
])

model = SelfAttentionV1(d_in=4, d_out=2)
modelV2 = selfAttentionV2(d_in=4, d_out = 2)

output = model(x)
output1 = model(x)

print("Output shape:", output.shape)
print("Output:", output)

print("Output shape of V2:", output1.shape)
print("Output:", output1)
