import numpy as np

from .apply_qat import apply_QAT
from torch import nn

class weightQATBERT(nn.Module):
    def __init__(self, model, rate=0.5):
        super(weightQATBERT, self).__init__()
        self.bert = model
        self.rate = rate

        #weight distribution based selection. 
        #가중치 분포의 threshold%를 커버하는데 필요한 range의 크기 순서대로 
        # n개는 8bit로, 나머지는 4bit로 처리하는 방식
        # embedding의 경우 weightrange의 임계값의 range와 비교해서 8 or 4
       
        weightrange = self.sortrange()
        #print(weightrange)
        n = int(len(weightrange) * self.rate)
        print(f"top {n} layers uses 8bit QAT")

        for j, layer in enumerate(self.bert.bert.encoder.layer): 
            for i in range(len(weightrange)): 
                layid, laytype, _ = weightrange[i]

                if i < n: #upper half.
                    if layid == j: #layer의 데이터를 가지고 QAT하세요
                        if laytype == 'att':
                            #print(f"upper half 'att'")
                            layer.attention.self = apply_QAT(layer.attention.self, precision = 8, mode = 'attention')
                            layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = 8, mode = 'ffn')                                 
                        if laytype == 'ffn':
                            #print(f"upper half 'ffn'")
                            layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 8, mode = 'ffn')
                            layer.output.dense = apply_QAT(layer.output.dense, precision = 8, mode = 'ffn')
                else:
                    if layid == j:
                        if laytype == 'att':
                            #print(f"lower half 'att'")
                            layer.attention.self = apply_QAT(layer.attention.self, precision = 4, mode = 'attention')
                            layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = 4, mode = 'ffn')
                        if laytype == 'ffn':
                            #print(f"lower half 'ffn'")
                            layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 4, mode = 'ffn')
                            layer.output.dense = apply_QAT(layer.output.dense, precision = 4, mode = 'ffn')
        
        
        e_range = self.getrange(self.bert.bert.embeddings.word_embeddings.weight) #embedding range
        _, _, b = weightrange[n] 
        
        if e_range >= b: #e_range is in upper bound
            ep = 8 #embedding precision
        else:
            ep = 4
        #print(f"e_range = {e_range}, bound = {b}, so embedding precision is: {ep}")
        self.bert.bert.embeddings.word_embeddings = apply_QAT(self.bert.bert.embeddings.word_embeddings, precision = ep, mode = 'embedding')
                    
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
        
    def sortrange(self):
        ranges=  []

        for i, layer in enumerate(self.bert.bert.encoder.layer):  
            ranges.append((i, 'att', self.getrange(layer.attention.self.query.weight)))
            ranges.append((i, 'ffn', self.getrange(layer.intermediate.dense.weight)))
        
        ranges.sort(key=lambda x: x[2], reverse=True) #내림차순 정렬

        return ranges

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
