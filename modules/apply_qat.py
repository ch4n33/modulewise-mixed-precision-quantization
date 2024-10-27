
import torch
import torch.nn as nn
from .round import STERoundFunction
def apply_QAT(layer, precision=8, mode='attention'):
    class CustomQuantizationLayer(nn.Module):
        def __init__(self, layer, bits):
            super(CustomQuantizationLayer, self).__init__()
            self.bits = bits
            self.layer = layer
            self.mode = mode  
            self.layer.requires_grad_(True)  # 양자화 레이어의 gradient 활성화
        
        def calculate_scale_zp(self, min,max, qmin, qmax):
            scale = (max - min) / (qmax - qmin)
            zero_point = qmin - STERoundFunction.apply(min / scale)
            return scale, zero_point
        def apply_fake_quant(self, tensor, scale, zero_point, qmin, qmax):
            quantized = STERoundFunction.apply(tensor / scale + zero_point)
            quantized = torch.clamp(quantized, qmin, qmax)
            
            fake_quantized = (quantized - zero_point) * scale
            return fake_quantized
        def forward(self, hidden_states, *args, **kwargs):
            # `requires_grad`와 `grad_fn`을 체크하는 코드
            # if hidden_states.requires_grad:
            #     pass
            # else:
            #     print("Gradient not enabled for input tensor.")
            epsilon = 1e-6 
            # 양자화 적용
            qmax = (2 ** (self.bits-1)) - 1
            qmin = -(2 ** (self.bits-1))
            if self.mode == 'attention':
                query = self.layer.query(hidden_states)
                key = self.layer.key(hidden_states)
                value = self.layer.value(hidden_states)
                
                # Quantize query, key, value
                query_scale, query_zp = self.calculate_scale_zp(query.min(), query.max(), qmin, qmax)
                key_scale, key_zp = self.calculate_scale_zp(key.min(), key.max(), qmin, qmax)
                value_scale, value_zp = self.calculate_scale_zp(value.min(), value.max(), qmin, qmax)
                
                query = self.apply_fake_quant(query, query_scale, query_zp, qmin, qmax)
                key = self.apply_fake_quant(key, key_scale, key_zp, qmin, qmax)
                value = self.apply_fake_quant(value, value_scale, value_zp, qmin, qmax)
                
                
                
                # Weight tensor의 requires_grad 확인
                # for tensor_name, tensor in zip(["query", "key", "value"], [query, key, value]):
                #     if tensor.requires_grad:
                #         pass
                #     else:
                #         print(f"Gradient not enabled for {tensor_name} tensor.")

                # Attention 연산 (양자화와 관련된 코드 추가)
                # query shape: (batch_size, num_heads, seq_len, head_dim)
                # key shape: (batch_size, num_heads, seq_len, head_dim)
                # key.transpose(-1, -2) shape: (batch_size, num_heads, head_dim, seq_len)
                # attention_scores shape: (batch_size, num_heads, seq_len, seq_len)
                
                attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (query.size(-1) ** 0.5)
                attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
                attention_output = torch.matmul(attention_probs, value)
                return (attention_output, )
            elif self.mode == 'ffn':
                # Linear layer를 통해 hidden_states를 계산
                layer_output = self.layer(hidden_states)
                
                # quantize
                output_scale, output_zp = self.calculate_scale_zp(layer_output.min(), layer_output.max(), qmin, qmax)
                layer_output = self.apply_fake_quant(layer_output, output_scale, output_zp, qmin, qmax)
                
                
                return layer_output

            return self.layer(hidden_states, *args, **kwargs)

    quant_layer = CustomQuantizationLayer(layer=layer, bits=precision)
    return quant_layer
