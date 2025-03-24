"""
Email Classification System for Financial Requests
Copyright (C) 2024 atoshveer

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
__author__ = "Atosh Veer <veeratosh@gmail.com>"
__license__ = "GPL-3.0"

# @@AUTHOR:Atosh Veer@@ 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch

# Load and split dataset
dataset = load_dataset('csv', data_files={'full': 'emails.csv'})
train_test = dataset['full'].train_test_split(test_size=0.2, shuffle=True)

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_test['train']['label']),
    y=train_test['train']['label']
)
weights = torch.tensor(class_weights, dtype=torch.float)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_test['train'].map(tokenize_function, batched=True)
valid_dataset = train_test['test'].map(tokenize_function, batched=True)

# Custom trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs',
    logging_steps=10,
    remove_unused_columns=False
)

# Initialize trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

# Train and save model
print("Starting training...")
trainer.train()
print("Training complete!")
trainer.save_model("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
print("Model saved to ./fine-tuned-model")