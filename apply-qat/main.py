

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values



from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(
    "/root/MRPC/", do_lower_case=True)


# Print the original sentence.
print(' Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))





max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)





# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])






from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))





from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )









from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "/root/MRPC/", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

from torch import nn
import copy 

class QuantizedBert(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedBert, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.ao.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.ao.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        # input_ids = self.quant(input_ids)
        # if attention_mask is not None:
        #     attention_mask = self.quant(attention_mask)
        # if token_type_ids is not None:
        #     token_type_ids = self.quant(token_type_ids.to(torch.int8))
        x = self.model_fp32(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, labels=labels)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

class myQuantStub(torch.nn.Module):
    def __init__(self):
        super(myQuantStub, self).__init__()
        self.scale = 32.0 / (2**4)  # 4비트 스케일
        self.zero_point = 16

    def forward(self, x):
        # 양자화된 값을 IntTensor로 변환
        x_int = torch.clamp(torch.round(x / self.scale) + self.zero_point, -16, 15).to(torch.int)
        return x_int

    def dequantize(self, x_int):
        # 비양자화
        return (x_int - self.zero_point) * self.scale

from transformers.modeling_outputs import SequenceClassifierOutput
class Quantized4BitBert(torch.nn.Module):
    def __init__(self, model):
        super(Quantized4BitBert, self).__init__()
        self.model_fp32 = model
        self.quant = myQuantStub()  # 4비트 양자화
        self.dequant = myQuantStub()  # 비양자화

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        # 양자화
        # input_ids = self.quant(input_ids)
        
        # 원래 모델의 forward 메서드 호출
        outputs = self.model_fp32(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 양자화 해제 후 logits를 float 타입으로 변환하고, requires_grad를 설정
        logits = self.dequant(outputs.logits).float().detach().requires_grad_(True)

        # 손실 계산
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model_fp32.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
# Tell pytorch to run this model on the GPU.
quantized_model = QuantizedBert(model)
quantized_4bit_model = Quantized4BitBert(model)
# ⑥ quantization configuration을 지정합니다. (ex. symmetric quantization, asymmetric quantization)
# Select quantization schemes from 
# https://pytorch.org/docs/stable/quantization-support.html
quantization_config = torch.ao.quantization.get_default_qconfig("x86")
# Custom quantization configurations
# quantization_config = torch.ao.quantization.default_qconfig
# quantization_config = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
quantized_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
quantized_4bit_model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

torch.ao.quantization.prepare_qat(quantized_model, inplace=True)
torch.ao.quantization.prepare_qat(quantized_4bit_model, inplace=True)
quantized_model.cuda()
quantized_4bit_model.cuda()
model.cuda()




# # Get all of the model's parameters as a list of tuples.
# params = list(quantized_4bit_model.named_parameters())

# print('The BERT model has {:} different named parameters.\n'.format(len(params)))

# print('==== Embedding Layer ====\n')

# for p in params[0:5]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# print('\n==== First Transformer ====\n')

# for p in params[5:21]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# print('\n==== Output Layer ====\n')

# for p in params[-4:]:
#     print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    



# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )




from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


from modules.train import train_model

# print("Training BERT on CoLA in float32...")
# training_stats = train_model(epochs, model, train_dataloader, validation_dataloader)


import pandas as pd

# # Display floats with two decimal places.
# pd.set_option('display.precision', 2)

# # Create a DataFrame from our training statistics.
# df_stats = pd.DataFrame(data=training_stats)

# # Use the 'epoch' as the row index.
# df_stats = df_stats.set_index('epoch')

# # A hack to force the column headers to wrap.
# #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# # save table to file(timestamp).csv
# df_stats.to_csv('table_fp32.csv')


# optimizer = AdamW(quantized_model.parameters(),
#                     lr = 3e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                     eps = 1e-6 # args.adam_epsilon  - default is 1e-8.
#                     )
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps
#                                             )
# print("Training BERT on CoLA in int8...")
# training_stats = train_model(epochs, quantized_model, train_dataloader, validation_dataloader, optimizer, scheduler)
# pd.set_option('display.precision', 2)

# df_stats = pd.DataFrame(data=training_stats)
# df_stats = df_stats.set_index('epoch')
# df_stats.to_csv('table_int8.csv')
# exit()


# print("Training BERT on CoLA in int4...")
# training_stats = train_model(epochs, quantized_4bit_model, train_dataloader, validation_dataloader)
# pd.set_option('display.precision', 2)

# df_stats = pd.DataFrame(data=training_stats)
# df_stats = df_stats.set_index('epoch')
# df_stats.to_csv('table_int4.csv')


from modules.mixed_qat import MixedQATBERT

# 모델 로드 및 준비
bert_model = BertForSequenceClassification.from_pretrained(
    "/root/MRPC/", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# from modules.range_tracker import HistogramRangeTracker
# range_tracker = HistogramRangeTracker(coverage=0.99)
from modules.range_tracker import MinMaxRangeTracker, EMAActivationRangeTracker
range_tracker = MinMaxRangeTracker()
activation_tracker = 'EMAActivationRangeTracker'
print("Original BERT model:")
print(bert_model)
mixed_qat_model = MixedQATBERT(bert_model, attention_bits=8, ffn_bits=4, range_tracker=range_tracker, activation_tracker=activation_tracker)
print("Mixed QAT BERT model:")
print(mixed_qat_model)


# 이후 학습 과정에서 각 레이어별 QAT가 적용됩니다.


mixed_qat_model.cuda()
optimizer = AdamW(mixed_qat_model.parameters(),
                    lr = 1e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps
                                            )
# 이후 학습 과정에서 각 레이어별 QAT가 적용됩니다.
print("Training Mixed QAT BERT on CoLA...")
training_stats = train_model(epochs, mixed_qat_model, train_dataloader, validation_dataloader, optimizer, scheduler)
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('table_int_mix-8-4.csv')


# bert_model.cuda()
# optimizer = AdamW(bert_model.parameters(),
#                     lr = 1e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
#                     weight_decay = 0.005
#                     )
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps
#                                             )
# # 이후 학습 과정에서 각 레이어별 QAT가 적용됩니다.
# print("Training Mixed QAT BERT on CoLA...")
# training_stats = train_model(epochs, bert_model, train_dataloader, validation_dataloader, optimizer, scheduler)
# pd.set_option('display.precision', 2)
# df_stats = pd.DataFrame(data=training_stats)
# df_stats = df_stats.set_index('epoch')
# df_stats.to_csv('table_fp32.csv')