from transformers import AutoTokenizer

class mytokenizer:
    def __init__(self, dataset, model="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.dataset_name = dataset

    def select(self, examples): #selecting tokenizer
        if self.dataset_name == "mrpc":
            return self.encode_mrpc(examples)
        elif self.dataset_name == "sst2":
            return self.encode_sst(examples)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    def encode_mrpc(self, examples):
        return self.tokenizer(
            examples['sentence1'],
            examples['sentence2'],
            truncation=True,
            padding='max_length',
            max_length=128
        )

    def encode_sst(self, examples):
        return self.tokenizer(
            examples['sentence'],  
            truncation=True,
            padding='max_length',
            max_length=128
        )