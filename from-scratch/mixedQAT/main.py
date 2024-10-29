import os
import torch
import pandas as pd

from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer

from modules.predefined_qat import predefQATBERT
from modules.weight_qat import weightQATBERT
from modules.rand_qat import randQATBERT
from modules.train import train_model
from modules.tokenizer import mytokenizer




############################# 실행 환경 설정 #############################
if torch.cuda.is_available():       
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


############################# 데이터 준비 및 전처리 #############################

dataset_name = "mrpc" # mrpc sst2 cola qqp
dataset = load_dataset("glue", dataset_name) #using GLUE data
print(f"Training Mixed QAT BERT on {dataset_name} dataset")

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

t = mytokenizer(dataset=dataset_name)
encoded_dataset = dataset.map(t.select, batched=True) 

input_ids = []
attention_masks = []

def to_tensor_dataset(split_dataset):
    input_ids = torch.tensor(split_dataset['input_ids'])
    attention_masks = torch.tensor(split_dataset['attention_mask'])
    labels = torch.tensor(split_dataset['label'])
    return TensorDataset(input_ids, attention_masks, labels)

train_dataset = to_tensor_dataset(encoded_dataset['train'])
val_dataset = to_tensor_dataset(encoded_dataset['validation'])
test_dataset = to_tensor_dataset(encoded_dataset['test'])

batch_size = 32

train_dataloader = DataLoader(
    train_dataset, 
    sampler=RandomSampler(train_dataset), 
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset, 
    sampler=SequentialSampler(val_dataset), 
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset, 
    sampler=SequentialSampler(test_dataset), 
    batch_size=batch_size
)

print(f'{len(train_dataset)} training samples')
print(f'{len(val_dataset)} validation samples')
print(f'{len(test_dataset)} test samples')
############################# BERT model 호출 및 준비 #############################
epochs = 4

bert_model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 2, output_attentions = False, output_hidden_states = False,)
'''
mixed_qat_model = predefQATBERT(bert_model, attention_bits=8, ffn_bits=4) #레이어마다 att에 8비트, ffn에 4비트 양자화
mode = 'predefined'
'''

mixed_qat_model = weightQATBERT(bert_model, rate=0.7) #가중치 분포가 넓은 상위 rate%개의 sub-module을 8비트, 나머지 4비트 양자화
mode = 'weight'

'''
mixed_qat_model = randQATBERT(bert_model, rate=0.7) #랜덤하게 rate%의 레이어를 전체 8비트 양자화.
mode = 'random'
'''

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

############################# Training, Validation, Inference #############################

training_stats = train_model(epochs, mixed_qat_model, train_dataloader, validation_dataloader, test_dataloader, optimizer, scheduler)

pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')

results_dir = './results'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

df_stats.to_csv(f'{results_dir}/{mode}.csv')

