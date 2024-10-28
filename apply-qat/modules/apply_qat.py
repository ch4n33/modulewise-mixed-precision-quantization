
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
            with torch.no_grad():
                scale = (max - min) / (qmax - qmin)
                zero_point = qmin - STERoundFunction.apply(min / scale)
            return scale, zero_point
        def apply_fake_quant(self, tensor, scale, zero_point, qmin, qmax):
            quantized = STERoundFunction.apply(tensor / scale + zero_point)
            quantized = torch.clamp(quantized, qmin, qmax)
            
            fake_quantized = (quantized - zero_point) * scale
            return fake_quantized
        
        def apply_weight_fake_quant(self, weight, qmin, qmax):
            # weight의 min, max 값으로 scale, zero_point 계산
            weight_scale, weight_zp = self.calculate_scale_zp(weight.min(), weight.max(), qmin, qmax)
            quantized_weight = self.apply_fake_quant(weight, weight_scale, weight_zp, qmin, qmax)
            return quantized_weight
        
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
            
            # input value quantization
            hidden_states_scale, hidden_states_zp = self.calculate_scale_zp(hidden_states.min(), hidden_states.max(), qmin, qmax)
            hidden_states = self.apply_fake_quant(hidden_states, hidden_states_scale, hidden_states_zp, qmin, qmax)
            
            if self.mode == 'attention':
                # Attention weights 양자화
                query_weight = self.apply_weight_fake_quant(self.layer.query.weight, qmin, qmax)
                key_weight = self.apply_weight_fake_quant(self.layer.key.weight, qmin, qmax)
                value_weight = self.apply_weight_fake_quant(self.layer.value.weight, qmin, qmax)
                
                # Query, Key, Value 연산에서 양자화된 weight 사용
                query = nn.functional.linear(hidden_states, query_weight, self.layer.query.bias)
                key = nn.functional.linear(hidden_states, key_weight, self.layer.key.bias)
                value = nn.functional.linear(hidden_states, value_weight, self.layer.value.bias)
                
                # Quantize query, key, value
                query_scale, query_zp = self.calculate_scale_zp(query.min(), query.max(), qmin, qmax)
                key_scale, key_zp = self.calculate_scale_zp(key.min(), key.max(), qmin, qmax)
                value_scale, value_zp = self.calculate_scale_zp(value.min(), value.max(), qmin, qmax)
                
                query = self.apply_fake_quant(query, query_scale, query_zp, qmin, qmax)
                key = self.apply_fake_quant(key, key_scale, key_zp, qmin, qmax)
                value = self.apply_fake_quant(value, value_scale, value_zp, qmin, qmax)
                
                # assert q, k, v does not containe nan or inf
                if torch.isnan(query).any() or torch.isnan(key).any() or torch.isnan(value).any():
                    print("query, key, value contains nan")
                if torch.isinf(query).any() or torch.isinf(key).any() or torch.isinf(value).any():
                    print("query, key, value contains inf")
                # 왜  q, k, v의 결과(hiddens_states를 곱한 결과물)을 양자화하니 validation.loss가 nan이 되는지 확인 필요
                # 일단 q,k,v 결과물이 nan이나 inf를 포함하지는 않는 것을 확인했음
                
                
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
                
                ap_scale, ap_zp = self.calculate_scale_zp(attention_probs.min(), attention_probs.max(), qmin, qmax)
                quantized_attention_probs = self.apply_fake_quant(attention_probs, ap_scale, ap_zp, qmin, qmax)
                
                attention_output = torch.matmul(quantized_attention_probs, value)

                # ao_scale, ao_zp = self.calculate_scale_zp(attention_output.min(), attention_output.max(), qmin, qmax)
                # attention_output = self.apply_fake_quant(attention_output, ao_scale, ao_zp, qmin, qmax)

                # apply dropout to attention_output
                attention_output = self.layer.dropout(attention_output)
                return (attention_output, )
            elif self.mode == 'ffn':
                # FFN의 weight 양자화
                ffn_weight = self.apply_weight_fake_quant(self.layer.weight, qmin, qmax)
                
                # 양자화된 weight 사용하여 FFN forward 연산 수행
                layer_output = nn.functional.linear(hidden_states, ffn_weight, self.layer.bias)
                
                # quantize
                # output_scale, output_zp = self.calculate_scale_zp(layer_output.min(), layer_output.max(), qmin, qmax)
                # layer_output = self.apply_fake_quant(layer_output, output_scale, output_zp, qmin, qmax)
                return layer_output
            return self.layer(hidden_states, *args, **kwargs)

    class CustomQuantizationEmbeddingLayer(CustomQuantizationLayer):
        def forward(self, input_ids, *args, **kwargs):
            qmin = -(2 ** (self.bits-1))
            qmax = (2 ** (self.bits-1)) - 1
            # Word Embedding weight 양자화
            word_embeddings = self.apply_weight_fake_quant(self.layer.weight, qmin, qmax)
            
            # 양자화된 weight 사용하여 Embedding forward 연산 수행
            word_embeddings = nn.functional.embedding(input_ids, word_embeddings)
            
            return word_embeddings
    if mode == 'attention' or mode == 'ffn':
        quant_layer = CustomQuantizationLayer(layer=layer, bits=precision)
    elif mode == 'embedding':
        quant_layer = CustomQuantizationEmbeddingLayer(layer=layer, bits=precision)
    return quant_layer
