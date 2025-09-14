import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import warnings
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import talib
import yfinance as yf
from datetime import datetime, timedelta


if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")  # Use the GPU
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")  # Use the CPU
