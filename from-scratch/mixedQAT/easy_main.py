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
print("*** choose dataset (mrpc sst2 cola qqp) ***")
dataset_name = input()
#dataset_name = "mrpc" 
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

print("*** choose Model mode from options below ***")
print("* predefined : Quantize to a precision specified as 8 bits for the ATT layer and 4 bits for the FFN layer. ")
print("* weight : Determine which submodules to quantize to 8 bits based on the size of the normal distribution. ")
print("* random : Select a layer to randomly quantize to 8 bits ")
mode = input()

if mode == 'predefined':
    print("*** You have chosen the predefined QAT BERT. ***")
    mixed_qat_model = predefQATBERT(bert_model, attention_bits=8, ffn_bits=4)
elif mode == 'weight':
    print("*** You have chosen the weight-based QAT BERT. ***")
    print("*** What percentage of submodules will be quantized to 8 bits? (Ex. 0.5)  ***")
    rate = float(input())
    mixed_qat_model = weightQATBERT(bert_model, rate=rate)
elif mode == 'random':
    print("*** You have chosen the random QAT BERT. ***")
    print("*** What percentage of submodules will be quantized to 8 bits? (Ex. 0.5)  ***")
    rate = float(input())
    mixed_qat_model = randQATBERT(bert_model, rate=rate)
else:
    print("*** Invalid choice. Please select 'predefined', 'weight', or 'random'. ***")


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

