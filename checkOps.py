from torch.onnx import export as onnx_export
from pathlib import Path
from torch.nn import MaxPool2d
# import torchtune.modules

import torch.nn.functional
class RandomSDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        query, key, value = x
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0, enable_gqa=True)


# model = MaxPool2d((2,2))
# dummy_inputs = (torch.randn(2,2,3,4, device="cpu"))

# model = torchtune.modules.RotaryPositionalEmbeddings(128)
# dummy_inputs = (torch.randn(1, 128, 8, 8, device="cpu"))

model = RandomSDPA()
batch_size = 2
seq_len = 3
embed_dim = 8
num_heads = 8
device='cpu'
dummy_inputs = (
    torch.randn(batch_size, num_heads, seq_len, embed_dim // num_heads, device=device),
    torch.randn(batch_size, 1, seq_len, embed_dim // num_heads, device=device),
    torch.randn(batch_size, 1, seq_len, embed_dim // num_heads, device=device)
)
outputDir = Path('C:\\Users\\Local_Admin\\Downloads\\Kshitij\\Models\\PyTorchOnnxDynamo\\Test\\model.onnx')

onnx_export(
    model,
    (dummy_inputs,),
    f=outputDir,
    do_constant_folding=True,
    opset_version=23,
    dynamo=True,
    verify=True,
    artifacts_dir = outputDir.parent,
    report=True,
    dump_exported_program=True
)
