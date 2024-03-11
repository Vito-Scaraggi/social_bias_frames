from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
from tqdm import tqdm
import time
import datetime

class SBICModel:
    def __init__(self, num_labels=4, model_name = 'bert-base-uncased', device='cuda'):
        self.model = BertForSequenceClassification.from_pretrained(
                        model_name, 
                        num_labels = num_labels, 
                        output_attentions = False, 
                        output_hidden_states = False,
                        problem_type = 'single_label_classification'
                    )
        self.model.to(device)
        self.device = device
    
    def load_best_model(self):
        self.model = torch.load('bert_model_sbic')

    def train(self, train_dataset, val_dataset, epochs, batch_size, lr = 5e-5):
        
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

        optimizer = AdamW(self.model.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
        )

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        
        training_stats = []
        best_eval_accuracy = 0

        for epoch in range(0, epochs):
            
            t_start = time.time()
            data_iterator = tqdm(
                train_dataloader,
                leave=True,
                unit="batch",
                postfix={
                    "epo": epoch,
                    "lss": "%.8f" % 0.0,
                },
                disable=False,
            ) 
           
            total_train_loss = 0
            self.model.train()
            
            for _, batch in enumerate(data_iterator):
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                self.model.zero_grad()
                output = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)                        
                loss = output.loss
                logits = output.logits
                total_train_loss += loss.item()
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.8f" % float(loss.item()),
                )

                optimizer.step()
                scheduler.step()

            t_end = time.time()
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            training_time = str(datetime.timedelta(seconds=int(round(t_end - t_start))))
                                
            print("\nAverage training loss: {0:.2f}".format(avg_train_loss))
            print("\nRunning Validation...")

            self.model.eval()

            # Tracking variables 
            total_correct = 0
            total_eval_loss = 0
            v_start = time.time()

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                with torch.no_grad():        
                    output= self.model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                loss = output.loss
                total_eval_loss += loss.item()

                # Move logits and labels to CPU if we are using GPU
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                pred_flat = np.argmax(logits, axis=1).flatten()
                labels_flat = label_ids.flatten()
                total_correct += np.sum(pred_flat == labels_flat)

            v_end = time.time()
            # Report the final accuracy for this validation run.
            val_accuracy = total_correct / len(val_dataset)
            
            print("\nValidation accuracy: {0:.2f}".format(val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            print("\nValidation Loss: {0:.2f}".format(avg_val_loss))

            val_time = str(datetime.timedelta(seconds=int(round(v_end - v_start))))

            #save the best model
            if val_accuracy > best_eval_accuracy:
                print("\nNew Best Model Found!")
                torch.save(self.model, 'bert_model_sbic')
                best_eval_accuracy = val_accuracy

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.':  val_accuracy,
                    'Training Time': training_time,
                    'Valid. Time': val_time
                }
            )

        print("\nTraining complete!")
        return training_stats
    
    def predict(self, test_dataset):
        
        test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = 1024 # Evaluate with this batch size.
        )

        print("")
        print("Running Test...")

        self.model.eval()

        # Tracking variables
        preds_all = []
        labels_all = []

        # Evaluate data for one epoch
        for batch in test_dataloader:

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():        
                output= self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

            # Move logits and labels to CPU if we are using GPU
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            preds_all.extend(pred_flat)
            labels_flat = label_ids.flatten()
            labels_all.extend(labels_flat)
            
        return preds_all, labels_all