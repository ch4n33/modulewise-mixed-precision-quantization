from .apply_qat import apply_QAT
from torch import nn

class fixedlayerQATBERT(nn.Module):
    def __init__(self, model, attention_bits=8, ffn_bits=4):
        super(fixedlayerQATBERT, self).__init__()
        self.bert = model

        #fixed precision for specific layer. 
        #각 레이어마다 attention은 8bit로 FFN은 4비트로 quantize
        
        for layer in self.bert.bert.encoder.layer:
            layer.attention.self = apply_QAT(layer.attention.self, precision = attention_bits, mode = 'attention')

            layer.attention.output.dense = apply_QAT(layer.attention.output.dense, precision = 8, mode = 'ffn')
            layer.intermediate.dense = apply_QAT(layer.intermediate.dense, precision = 4, mode = 'ffn')
            layer.output.dense = apply_QAT(layer.output.dense, precision = ffn_bits, mode = 'ffn')
        # apply qat to embedding layers
        self.bert.bert.embeddings.word_embeddings = apply_QAT(self.bert.bert.embeddings.word_embeddings, precision = 8, mode = 'embedding')

    # def forward(self, input_ids, attention_mask=None):
    #     return self.bert(input_ids, attention_mask)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BertForSequenceClassification의 forward에 모든 인수를 전달
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
