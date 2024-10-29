
import torch
import torch.nn as nn
from .round import STERoundFunction
import math
from .range_tracker import RangeTracker
def apply_QAT(layer, precision=8, mode='attention', range_tracker=None):
    class CustomQuantizationLayer(nn.Module):
        def __init__(self, layer, bits):
            super(CustomQuantizationLayer, self).__init__()
            if (range_tracker is None) or (not isinstance(range_tracker, RangeTracker)):
                raise ValueError("range_tracker should be an instance of RangeTracker")
            self.bits = bits
            self.layer = layer
            self.mode = mode  
            self.layer.requires_grad_(True)  # 양자화 레이어의 gradient 활성화
            self.range_tracker = range_tracker

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
        
        def apply_weight_fake_quant(self, layer, qmin, qmax):
            # weight의 min, max 값으로 scale, zero_point 계산
            weight = layer.weight
            weight_scale, weight_zp = self.calculate_scale_zp(*self.range_tracker(weight), qmin, qmax)
            quantized_weight = self.apply_fake_quant(weight, weight_scale, weight_zp, qmin, qmax)
            layer.weight = torch.nn.Parameter(quantized_weight) #quantized_weight
    class CustomQuantizationAttentionLayer(CustomQuantizationLayer):
        def forward(self, 
                    hidden_states, 
                    attention_mask=None, 
                    head_mask=None, 
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    past_key_value=None,
                    output_attentions=None,
                    ):
            qmin, qmax = -(2 ** (self.bits-1)), (2 ** (self.bits-1)) - 1

            mixed_query_layer = self.layer.query(hidden_states)

            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention and past_key_value is not None:
                # reuse k,v, cross_attentions
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self.layer.transpose_for_scores(self.layer.key(encoder_hidden_states))
                value_layer = self.layer.transpose_for_scores(self.layer.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self.layer.transpose_for_scores(self.layer.key(hidden_states))
                value_layer = self.layer.transpose_for_scores(self.layer.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self.layer.transpose_for_scores(self.layer.key(hidden_states))
                value_layer = self.layer.transpose_for_scores(self.layer.value(hidden_states))

            query_layer = self.layer.transpose_for_scores(mixed_query_layer)
            query_scale, query_zp = self.calculate_scale_zp(*self.range_tracker(query_layer), qmin, qmax)
            query_layer = self.apply_fake_quant(query_layer, query_scale, query_zp, qmin, qmax)
            
            key_scale, key_zp = self.calculate_scale_zp(*self.range_tracker(key_layer), qmin, qmax)
            key_layer = self.apply_fake_quant(key_layer, key_scale, key_zp, qmin, qmax)
            
            value_scale, value_zp = self.calculate_scale_zp(*self.range_tracker(value_layer), qmin, qmax)
            value_layer = self.apply_fake_quant(value_layer, value_scale, value_zp, qmin, qmax)
            # self.apply_weight_fake_quant(query_layer, qmin, qmax)
            # self.apply_weight_fake_quant(key_layer, qmin, qmax)
            # self.apply_weight_fake_quant(value_layer, qmin, qmax)
            use_cache = past_key_value is not None
            if self.layer.is_decoder:
                # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
                # Further calls to cross_attention layer can then reuse all cross-attention
                # key/value_states (first "if" case)
                # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
                # all previous decoder key/value_states. Further calls to uni-directional self-attention
                # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
                # if encoder bi-directional self-attention `past_key_value` is always `None`
                past_key_value = (key_layer, value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if self.layer.position_embedding_type == "relative_key" or self.layer.position_embedding_type == "relative_key_query":
                query_length, key_length = query_layer.shape[2], key_layer.shape[2]
                if use_cache:
                    position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                        -1, 1
                    )
                else:
                    position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r

                positional_embedding = self.layer.distance_embedding(distance + self.layer.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.layer.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.layer.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

            attention_scores = attention_scores / math.sqrt(self.layer.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.layer.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            
            attn_pb_scale, attn_pb_zp = self.calculate_scale_zp(*self.range_tracker(attention_probs), qmin, qmax)
=======
            attention_probs = self.apply_fake_quant(attention_probs, attn_pb_scale, attn_pb_zp, qmin, qmax)

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.layer.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            if self.layer.is_decoder:
                outputs = outputs + (past_key_value,)
            return outputs
    class CustomQuantizationFFNLayer(CustomQuantizationLayer):
        def forward(self, hidden_states, *args, **kwargs):
            qmin, qmax = -(2 ** (self.bits-1)), (2 ** (self.bits-1)) - 1
            # FFN의 weight 양자화
            self.apply_weight_fake_quant(self.layer, qmin, qmax)
            
            # 양자화된 weight 사용하여 FFN forward 연산 수행
            layer_output = self.layer(hidden_states, *args, **kwargs)
            
            return layer_output
    class CustomQuantizationEmbeddingLayer(CustomQuantizationLayer):
        def forward(self, input_ids, *args, **kwargs):
            qmin = -(2 ** (self.bits-1))
            qmax = (2 ** (self.bits-1)) - 1
            # Word Embedding weight 양자화
            self.apply_weight_fake_quant(self.layer, qmin, qmax)
            
            # 양자화된 weight 사용하여 Embedding forward 연산 수행
            word_embeddings = self.layer(input_ids, *args, **kwargs)
            
            return word_embeddings
    if mode == 'attention':
        quant_layer = CustomQuantizationAttentionLayer(layer=layer, bits=precision)
    elif mode == 'ffn':
        quant_layer = CustomQuantizationFFNLayer(layer=layer, bits=precision)
    elif mode == 'embedding':
        quant_layer = CustomQuantizationEmbeddingLayer(layer=layer, bits=precision)
    return quant_layer
