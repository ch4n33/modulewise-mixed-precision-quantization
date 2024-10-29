import torch
from torch import nn

# this module track the range of the input tensor and apply fake quantization to the tensor.

class RangeTracker(nn.Module):
  def __init__(self):
    super(RangeTracker).__init__()

class MinMaxRangeTracker(RangeTracker):
  def forward(self, tensor):
    # this calculation should not involve the gradient
    with torch.no_grad():
      min = tensor.min()
      max = tensor.max()
    return min, max
  
class HistogramRangeTracker(RangeTracker):
  def __init__(self, coverage=0.99):
    super(HistogramRangeTracker, self).__init__(bits)
    if coverage <= 0 or coverage > 1:
      raise ValueError("coverage should be in (0, 1]")
    self.coverage = coverage
  def forward(self, tensor):
    # this calculation should not involve the gradient
    with torch.no_grad():
      hist, bin_edges = torch.histogram(tensor, bins=256, min=tensor.min(), max=tensor.max())
      cumulative_hist = torch.cumsum(hist, dim=0)
      total_count = cumulative_hist[-1]
      
      lower_idx = torch.argmax(cumulative_hist > total_count * (1 - self.coverage)/2)
      upper_idx = torch.argmax(cumulative_hist > total_count * (1 + self.coverage)/2)
      
      min_value = bin_edges[lower_idx]
      max_value = bin_edges[upper_idx]
      
    return min_value, max_value