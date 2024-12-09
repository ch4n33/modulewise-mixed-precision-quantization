
from .apply_qat import apply_QAT
from torch import nn
from .range_tracker import RangeTracker

class MixedQATBERT(nn.Module):
    def __init__(self, model, attention_bits=8, ffn_bits=4, range_tracker=None, activation_tracker=None):
        super(MixedQATBERT, self).__init__()
        if (range_tracker is None) or (not isinstance(range_tracker, RangeTracker)):
            raise ValueError("range_tracker should be an instance of RangeTracker")
            
        self.bert = model
        self.range_tracker = range_tracker

        self.activation_tracker = activation_tracker
        # 각 레이어마다 attention과 FFN에 다른 quantization 적용
        for layer in self.bert.bert.encoder.layer:
            layer.attention.self = apply_QAT(layer.attention.self, precision = attention_bits, mode = 'attention', range_tracker = range_tracker, activation_tracker = activation_tracker)

            layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = 8, mode = 'ffn', range_tracker = range_tracker, activation_tracker = activation_tracker)
            layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 4, mode = 'ffn', range_tracker = range_tracker, activation_tracker = activation_tracker)
            layer.output.dense = apply_QAT(layer.output.dense, precision = ffn_bits, mode = 'ffn', range_tracker = range_tracker, activation_tracker = activation_tracker)
        # apply qat to embedding layers
        self.bert.bert.embeddings.word_embeddings = apply_QAT(self.bert.bert.embeddings.word_embeddings, precision = 8, mode = 'embedding', range_tracker = range_tracker, activation_tracker = activation_tracker)


    # def forward(self, input_ids, attention_mask=None):
    #     return self.bert(input_ids, attention_mask)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BertForSequenceClassification의 forward에 모든 인수를 전달
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
