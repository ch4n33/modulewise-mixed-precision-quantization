import torch
import pandas as pd

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from modules.fixlayer_qat import fixedlayerQATBERT 
from modules.weight_qat import weightQATBERT
from modules.rand_qat import randQATBERT
from modules.train import train_model

############################# 실행 환경 설정 #############################
if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


############################# 데이터 준비 및 전처리 #############################
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

sentences = df.sentence.values
labels = df.label.values

print('Loading BERT tokenizer...')
#tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=True) #xa option
tokenizer = BertTokenizer.from_pretrained("/root/MRPC/", do_lower_case=True) #jongchan option

max_len = 0
for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset)) 
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 32

train_dataloader = DataLoader(
            train_dataset, 
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size
        )

############################# BERT model 호출 및 준비 #############################
epochs = 5

bert_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False,)
#model = BertForSequenceClassification.from_pretrained("/root/MRPC/", num_labels = 2, output_attentions = False, output_hidden_states = False,)
'''
mixed_qat_model = predifQATBERT(bert_model, attention_bits=8, ffn_bits=4) #레이어마다 att에 8비트, ffn에 4비트 양자화

mixed_qat_model = weightQATBERT(bert_model, rate=0.7) #가중치 분포가 넓은 상위 rate%개의 sub-module을 8비트, 나머지 4비트 양자화
'''
mixed_qat_model = randQATBERT(bert_model, rate=0.7) #랜덤하게 rate%의 레이어를 전체 8비트 양자화.




mixed_qat_model.cuda()

total_steps = len(train_dataloader) * epochs

optimizer = AdamW(mixed_qat_model.parameters(),
                    lr = 1e-4, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps
                                            )

############################# Training and Validation #############################
print("Training Mixed QAT BERT on CoLA...")
training_stats = train_model(epochs, mixed_qat_model, train_dataloader, validation_dataloader, optimizer, scheduler)


############################# main #############################
pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('table_int_mix-8-4.csv')
