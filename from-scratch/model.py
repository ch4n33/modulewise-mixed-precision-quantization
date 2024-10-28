import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class quantize(torch.autograd.Function):
    @staticmethod
    def scaling(input, bits): #min-max
        qmax = (2 ** bits) - 1
        qmin = -(2 ** bits)

        scale = ((2 * max(abs(input.max().item()), abs(input.min().item()))) / (qmax - qmin))
        if scale == 0:
            scale = 1e-5 #any value

        return qmin, qmax, scale
    @staticmethod
    def forward(ctx, input, scale, qmin, qmax): #symmetric quantization
        ctx.scale = scale

        float_min = input.min().item() 

        out = (torch.clamp(torch.round(input/scale), qmin, qmax)) * scale

        return out
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output / ctx.scale 

        return grad_input, None, None, None

class myEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, bits):
        super(myEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.bits = bits

    def forward(self, token_ids):
        token_embeds = self.token_embedding(token_ids)
        print(f"embedded token: {token_embeds}")

        #qmin, qmax, scale = quantize.scaling(token_embeds, self.bits)
        #print(f"scale at embedding: {scale}")
        #q_embeds = quantize.apply(token_embeds, scale, qmin, qmax)


        return self.dropout(token_embeds)

class multiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout, bits):
        super().__init__()
        assert d_model % h == 0

        # let d_k = d_v
        self.d_k = d_model // h #demantion of each head
        self.h = h #num of heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) #for q, k, v
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.bits = bits

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        #quantization
        qu_qmin, qu_qmax, qu_scale = quantize.scaling(query, self.bits)
        print(f"scale at att:query: {qu_scale}")
        q_query = quantize.apply(query, qu_scale, qu_qmin, qu_qmax)

        key_qmin, key_qmax, key_scale = quantize.scaling(key, self.bits)
        print(f"scale at att:key: {key_scale}")
        q_key = quantize.apply(key, key_scale, key_qmin, key_qmax)

        va_qmin, va_qmax, va_scale = quantize.scaling(value, self.bits)
        print(f"scale at att:value: {va_scale}")
        q_value = quantize.apply(value, va_scale, va_qmin, va_qmax)

        #linear projections
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (q_query, q_key, q_value))]


        scores = torch.matmul(query, key.transpose(-1, -2)) / (query.size(-1) ** 0.5)

        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)

        probs = F.softmax(scores, dim=-1) #attention_probs
        p_qmin, p_qmax, p_scale = quantize.scaling(probs, self.bits)
        print(f"scale at att:prons: {p_scale}")
        q_probs = quantize.apply(probs, p_scale, p_qmin, p_qmax)

        if self.dropout is not None: p_attn = self.dropout(probs)

     
        output = torch.matmul(p_attn, value) #attention_output
        o_qmin, o_qmax, o_scale = quantize.scaling(output, self.bits)
        print(f"scale at att:output: {o_scale}")
        q_output = quantize.apply(output, o_scale, o_qmin, o_qmax)

        output = q_output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(output)

class feedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout, bits):
        super(feedForwardNetwork, self).__init__()
        #self.intermediate_dense = nn.Linear(d_model, d_ff)
        self.w_1 = nn.Linear(d_model, d_ff) #intermediate.dense
        self.w_2 = nn.Linear(d_ff, d_model) #output.dense
        self.dropout = nn.Dropout(dropout)
        self.bits = bits

    def gelu(self, x): #activation function
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.tensor(torch.pi)) * (x + 0.044715 * x ** 3)))

    def forward(self, x):
        #print("-------------------------")
        #print(x.shape)
        qmin, qmax, scale = quantize.scaling(x, self.bits)
        print(f"scale at ffn: {scale}")
        x = quantize.apply(x, scale, qmin, qmax)
        tmp = self.w_1(x)
        #print(f"w_1 shape: {tmp.shape}")
        tmp = self.dropout(self.gelu(tmp))
        #print(f"action shape: {tmp.shape}")
        tmp = self.w_2(tmp)
        #print(f"w_2 shape: {tmp.shape}")
    
        qmin_, qmax_, scale_ = quantize.scaling(tmp, self.bits)
        print(f"scale at ffn:output: {scale_}")
        output = quantize.apply(tmp, scale_, qmin_, qmax_)
      
        return output
    
class addNorm(nn.Module):
    def __init__(self, size, dropout, eps=1e-6):
        super(addNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Layer normalization
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized_x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
        # Residual connection
        return x + self.dropout(sublayer(normalized_x))
    
class transformerBlock(nn.Module):
    #Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, bits):
        super().__init__()
        self.att = multiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, bits=bits)
        self.ffn = feedForwardNetwork(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout, bits=bits)
        self.input_sublayer = addNorm(size=hidden, dropout=dropout)
        self.output_sublayer = addNorm(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.att.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.ffn)

        return self.dropout(x)
    
class myBERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, bits=8):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.bits = bits
        self.feed_forward_hidden = hidden * 4

        self.embedding = myEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout, bits=bits)
        self.transformer_blocks = nn.ModuleList([transformerBlock(hidden, attn_heads, hidden * 4, dropout, bits) for _ in range(n_layers)])
        self.fc = nn.Linear(hidden, 2)  #fully connected layer

    def forward(self, x, attention_mask=None, labels=None):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        logits = self.fc(x[:, 0, :])  
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() 
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))  
        
        return loss, logits