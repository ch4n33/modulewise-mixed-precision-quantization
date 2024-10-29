import torch
from torch import nn

# this module track the range of the input tensor and apply fake quantization to the tensor.

class RangeTracker(nn.Module):
  def __init__(self):
    super(RangeTracker, self).__init__()

class MinMaxRangeTracker(RangeTracker):
  def __init__(self):
    super(MinMaxRangeTracker, self).__init__()
  def forward(self, tensor):
    # this calculation should not involve the gradient
    with torch.no_grad():
      min = tensor.min()
      max = tensor.max()
    return min, max
  
class HistogramRangeTracker(RangeTracker):
  def __init__(self, coverage=0.99):
    super(HistogramRangeTracker, self).__init__()
    if coverage <= 0 or coverage > 1:
      raise ValueError("coverage should be in (0, 1]")
    self.coverage = coverage
  def forward(self, tensor):
    # this calculation should not involve the gradient
    # Move tensor to CPU if it's on CUDA
    tensor = tensor.cpu()
    if not isinstance(tensor, torch.Tensor):
      raise ValueError("input should be a tensor, not {}".format(type(tensor)))
    # Convert Parameter to Tensor if necessary
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.detach()
    with torch.no_grad():
      hist, bin_edges = torch.histogram(tensor, bins=256, range=(tensor.min().item(), tensor.max().item()))
      cumulative_hist = torch.cumsum(hist, dim=0)
      cumulative_hist = cumulative_hist.cuda()
      
      total_count = cumulative_hist[-1]
      target_lower = total_count * (1 - self.coverage) / 2
      target_upper = total_count * (1 + self.coverage) / 2

      # 조건을 만족하는 인덱스 찾기
      lower_idx = torch.nonzero(cumulative_hist > target_lower, as_tuple=True)[0][0]
      upper_idx = torch.nonzero(cumulative_hist > target_upper, as_tuple=True)[0][0]

      
      min_value = bin_edges[lower_idx]
      max_value = bin_edges[upper_idx]
      
    return min_value, max_value