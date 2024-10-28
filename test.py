import time
import datetime
import torch
import random
import pandas as pd
import numpy as np


from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

from model import myBERT

############################# 실행 환경 설정 #############################
if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"using %s" % device)

############################# 데이터 준비 및 전처리 #############################
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Get the lists of sentences and their labels.
sentences = df.sentence.values
labels = df.label.values

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=True)

max_len = 128

# Tokenize all of the sentences and map the tokens to thier word IDs.
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
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size 
        )

############################# BERT model 호출 및 준비 #############################
'''
model = BertForSequenceClassification.from_pretrained( 
    "google-bert/bert-base-uncased", 
    num_labels = 2,
    output_attentions = False, 
    output_hidden_states = False,
)
quantization_config = torch.ao.quantization.get_default_qconfig("x86")

model.cuda()
'''
print("Loading BERT model...")
model = myBERT(vocab_size = tokenizer.vocab_size)
model.cuda()

epochs = 5
total_steps = len(train_dataloader) * epochs

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

############################# Training & Validation #############################
def train_model(epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler):
    seed_val = 42 

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=True)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # forward pass
            output = model(b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)

            loss, logits = output
            '''
            loss = output.loss
            logits = output.logits
            '''
            total_train_loss += loss.item()

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            
            for name, param in model.named_parameters(): #tracking gradients of each parameter
                if param.grad is not None:
                    print(f"Layer: {name} | Gradient Norm: {param.grad.norm().item()}")
            


            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)
        
        print("--------------------------------------")
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Total training time: {:}".format(training_time))
        print("--------------------------------------") 
        # -------- start validation
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0 

        for batch in tqdm(validation_dataloader, desc="Validating", leave=True):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        
                output = model(b_input_ids, 
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss, logits = output
                '''
                loss = output.loss
                logits = output.logits
                '''
            

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()   

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("--------------------------------------")
        print("  *Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation time: {:}".format(validation_time))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time,
            }
        )

    print("======== Training complete! ========")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats

############################# main #############################
print("Training BERT model...")
training_stats = train_model(epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler)

pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
df_stats.to_csv('result_CoLA.csv')
