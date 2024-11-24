import random
import numpy as np

from .apply_qat import apply_QAT
from torch import nn

class randQATBERT(nn.Module):
    def __init__(self, model, rate=0.5):
        super(randQATBERT, self).__init__()
        self.bert = model
        self.rate = rate

        #randomly selecting 4bit quantizing layers
        #랜덤하게 n%의 레이어를 8비트로 양자화함
        #if 4비트로 표현시 loss가 너무 커질거같으면 8비트로 하자
        #사실 attention은 8비트로 표현하는게 제일 좋긴한데 뭐 일단은..
        #그냥 정말 아예 random으로 해버릴까 검수과정없이.. 어차피 속도로 성능판별하기 애매한데 지금 코드도 좀 더러워보이고..

        elay = self.bert.bert.encoder.layer #encoder layer
        n = int(len(elay) * self.rate)  
        print(f"{n} layers gonna be quantized in 8bit")

        sampled = np.random.choice(elay, n, replace=False)

        for layer in elay:
            if layer in sampled:
                layer.attention.self = apply_QAT(layer.attention.self, precision = 8, mode = 'attention')
                layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = 8, mode = 'ffn')
                layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 8, mode = 'ffn')
                layer.output.dense = apply_QAT(layer.output.dense, precision = 8, mode = 'ffn') 
            else: 
                p = 4 # precision. 
                #if 4bit range is not enough, set precision to 8
                if self.getrange(layer.attention.self.query.weight) > 16: p = 8
                layer.attention.self = apply_QAT(layer.attention.self, precision = p, mode = 'attention')
                if self.getrange(layer.attention.output.dense.weight) > 16: p = 8
                layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = p, mode = 'ffn')
                if self.getrange(layer.intermediate.dense.weight) > 16: p = 8
                layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = p, mode = 'ffn')
                if self.getrange(layer.output.dense.weight) > 16: p = 8
                layer.output.dense = apply_QAT(layer.output.dense, precision = p, mode = 'ffn')
        
        e = np.random.choice([4, 8]) #embedding bit selection
        if e != 4 and self.getrange(self.bert.bert.embeddings.word_embeddings.weight) > 16: e = 8
        self.bert.bert.embeddings.word_embeddings = apply_QAT(self.bert.bert.embeddings.word_embeddings, precision = e, mode = 'embedding')

    def getrange(self, weights, threshold=0.99):  
        weights = weights.detach().cpu().numpy()
        weights = np.array(weights) 
        #print("weights:", weights)

        mean = np.mean(weights)
        stddev = np.std(weights)
        zscore = np.percentile(weights, threshold)

        lb = mean - zscore * stddev #lower bound
        ub = mean + zscore * stddev #upper bound

        return abs(ub - lb) #return coverage
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
