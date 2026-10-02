#!/bin/bash
#SBATCH --job-name=torch2_probe
#SBATCH --output=torch2_probe_%j.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mimosa2

python - <<'PY'
import torch, torch.nn.functional as F
print('torch', torch.__version__, '| arch list', torch.cuda.get_arch_list())
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    x = torch.randn(8, 8, 512, 64, device='cuda', dtype=torch.float16)
    y = F.scaled_dot_product_attention(x, x, x)
    torch.cuda.synchronize()
    print('sdpa fp16 ok', tuple(y.shape))
except Exception as e:
    print('sdpa failed:', type(e).__name__, str(e)[:300])
PY
