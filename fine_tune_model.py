"""
Email Classification System for Financial Requests
Copyright (C) 2024
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
import logging
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="fine_tune_model.log",
)

def clean_dataset(df):
    """Handle missing values and clean the dataset."""
    # Remove rows with missing text or label
    df = df.dropna(subset=['text', 'label'])
    # Remove empty strings
    df = df[df['text'].str.strip().astype(bool)]
    # Remove rows with invalid labels (not 0 or 1)
    df = df[df['label'].isin([0, 1])]
    return df.reset_index(drop=True)

def prepare_datasets(df):
    """Prepare train/test datasets with proper formatting."""
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = (predictions == labels).mean()
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    try:
        logging.info("Loading dataset...")
        df = pd.read_csv('emails.csv')
        df = clean_dataset(df)

        # Convert labels to numeric
        unique_labels = sorted(df['label'].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        df['label'] = df['label'].map(label_map)

        train_test = prepare_datasets(df)

        num_classes = len(unique_labels)
        if num_classes < 2:
            raise ValueError(f"Need at least 2 classes, found {num_classes}")

        logging.info("Calculating class weights...")
        try:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_test['train']['label']),
                y=train_test['train']['label']
            )
            weights = torch.tensor(class_weights, dtype=torch.float)
        except Exception as e:
            logging.warning(f"Skipping class weight calculation: {str(e)}")
            weights = None

        logging.info("Initializing model...")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        train_dataset = train_test['train'].map(tokenize_function, batched=True)
        valid_dataset = train_test['test'].map(tokenize_function, batched=True)

        # Remove unnecessary columns
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']])
        valid_dataset = valid_dataset.remove_columns([col for col in valid_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']])

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        logging.info("Setting up training...")
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",  # Fixed the deprecation warning
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='./logs',
            logging_steps=10,
            remove_unused_columns=True,
            report_to="none"
        )

        # Custom trainer to handle class weights and unexpected arguments
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Handle unexpected arguments with **kwargs."""
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits

                # Expand dimensions if labels are 0-d tensors
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(model.device) if weights is not None else None)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )

        logging.info("Starting training...")
        trainer.train()

        logging.info("Evaluating model...")
        eval_results = trainer.evaluate()
        print("Evaluation results:", eval_results)

        output_dir = "./fine-tuned-model"
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save label mapping
        with open(os.path.join(output_dir, "label_map.json"), "w") as f:
            import json
            json.dump(label_map, f)

        print(f"Model successfully trained and saved to {output_dir}")

    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
