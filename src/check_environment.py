import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
import matplotlib
import seaborn

print("=" * 50)
print("MODEL COMPRESSION PROJECT: ENVIRONMENT CHECK")
print("=" * 50)

print(f"Python version:       {sys.version.split()[0]}")
print(f"Python executable:    {sys.executable}")
print(f"Operating system:     {platform.platform()}")
print(f"PyTorch version:      {torch.__version__}")
print(f"Torchvision version:  {torchvision.__version__}")
print(f"NumPy version:        {np.__version__}")
print(f"Pandas version:       {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Matplotlib version:   {matplotlib.__version__}")
print(f"Seaborn version:      {seaborn.__version__}")
print(f"CPU threads available: {torch.get_num_threads()}")
print(f"CUDA available:       {torch.cuda.is_available()}")

print("=" * 50)
print("Environment check completed successfully.")
