
import torch
class STERoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward에서는 rounding 적용
        return torch.round(input)
    @staticmethod
    def backward(ctx, grad_output):
        # Backward에서는 원래 gradient 그대로 통과
        return grad_output