import time
import torch
import random
import numpy as np

from tqdm import tqdm

from .util import format_time, flat_accuracy


def train_model(epochs, model, train_dataloader, validation_dataloader, optimizer, scheduler):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_stats = []

    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train() #set model as training mode

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=True)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            #forward pass
            output = model(b_input_ids, 
               token_type_ids=None, 
               attention_mask=b_input_mask, 
               labels=b_labels)
            
            if output is None:
                print("output is None. Check the model's forward method.")
            if output.loss is None:
                print("output.loss is None. Check the model's forward method.")
            else:
                loss = output.loss
                logits = output.logits


            loss = output.loss
            logits = output.logits

            total_train_loss += loss.item()


            #backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            
        training_time = format_time(time.time() - t0)

        print("--------------------------------------")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Total training time: {:}".format(training_time))
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
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                loss = output.loss
                logits = output.logits
                
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
                'Validation Time': validation_time
            }
        )

    print("======== Training complete! ========")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return training_stats
