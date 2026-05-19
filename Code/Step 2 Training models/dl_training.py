import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from IPython.display import FileLink, display

MODEL_NAME = "microsoft/deberta-large" 

print(f"=== INITIALIZING PIPELINE FOR {MODEL_NAME} ===")

print("\nLoading datasets...")
train_df = pd.read_csv("/kaggle/input/datasets/aditya3kumarmishra/data-input/augmented_training.tsv", sep='\t')
val_df = pd.read_csv("/kaggle/input/datasets/aditya3kumarmishra/data-input/merged_val.tsv", sep='\t')
test_df = pd.read_csv("/kaggle/input/datasets/aditya3kumarmishra/data-input/merged_test.tsv", sep='\t')

label_cols = [
    'Self-direction: thought attained', 'Self-direction: thought constrained',
    'Self-direction: action attained', 'Self-direction: action constrained',
    'Stimulation attained', 'Stimulation constrained',
    'Hedonism attained', 'Hedonism constrained',
    'Achievement attained', 'Achievement constrained',
    'Power: dominance attained', 'Power: dominance constrained',
    'Power: resources attained', 'Power: resources constrained',
    'Face attained', 'Face constrained',
    'Security: personal attained', 'Security: personal constrained',
    'Security: societal attained', 'Security: societal constrained',
    'Tradition attained', 'Tradition constrained',
    'Conformity: rules attained', 'Conformity: rules constrained',
    'Conformity: interpersonal attained', 'Conformity: interpersonal constrained',
    'Humility attained', 'Humility constrained',
    'Benevolence: caring attained', 'Benevolence: caring constrained',
    'Benevolence: dependability attained', 'Benevolence: dependability constrained',
    'Universalism: concern attained', 'Universalism: concern constrained',
    'Universalism: nature attained', 'Universalism: nature constrained',
    'Universalism: tolerance attained', 'Universalism: tolerance constrained'
]

print("Upsampling all minority classes to 2000 minimum...")
TARGET_MINIMUM = 2000 
upsample_indices = []

for i, col_name in enumerate(label_cols):
    positive_indices = np.where(train_df[col_name] == 1)[0]
    current_count = len(positive_indices)
    
    if 0 < current_count < TARGET_MINIMUM:
        deficit = TARGET_MINIMUM - current_count
        chosen_indices = np.random.choice(positive_indices, size=deficit, replace=True)
        upsample_indices.extend(chosen_indices)

if upsample_indices:
    upsampled_df = train_df.iloc[upsample_indices]
    train_df = pd.concat([train_df, upsampled_df], ignore_index=True)

# Shuffle the data
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Teacher Dataset size after massive upsampling: {len(train_df)} rows")

def process_df(df):
    df['labels'] = df[label_cols].values.tolist()
    return Dataset.from_pandas(df[['Text', 'labels']])

train_dataset = process_df(train_df)
val_dataset = process_df(val_df)
test_dataset = process_df(test_df)

print(f"\nLoading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["Text"], truncation=True, max_length=128)

print("Tokenizing data...")
encoded_train = train_dataset.map(tokenize_function, batched=True)
encoded_val = val_dataset.map(tokenize_function, batched=True)
encoded_test = test_dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.array(labels).astype(int)
    predictions = (torch.sigmoid(torch.tensor(logits)).numpy() > 0.5).astype(int)
    
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": accuracy}

print(f"\nLoading {MODEL_NAME} architecture...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=38, 
    problem_type="multi_label_classification"
)


safe_name = MODEL_NAME.replace("/", "_")

training_args = TrainingArguments(
    output_dir=f"./{safe_name}_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2, 
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True, 
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

print("\nStarting Training...")
trainer.train()

print("\nTraining complete. Extracting Teacher Logits for Student Distillation...")
train_results = trainer.predict(encoded_train)
teacher_logits = train_results.predictions

npy_filename = f"{safe_name}_logits.npy"
np.save(npy_filename, teacher_logits)
print(f"Saved '{npy_filename}' successfully.")

print("\n" + "="*65)
print(f"FINAL TEST SET EVALUATION ({MODEL_NAME})")
print("="*65)

test_results = trainer.predict(encoded_test)
test_logits = test_results.predictions
test_labels = test_results.label_ids.astype(int)
test_preds = (torch.sigmoid(torch.tensor(test_logits)).numpy() > 0.5).astype(int)

final_macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
final_accuracy = accuracy_score(test_labels, test_preds)

print(f"OVERALL MACRO-F1: {final_macro_f1:.4f}")
print(f"OVERALL ACCURACY: {final_accuracy:.4f}\n")

print(f"{'Schwartz Value Status':<40} | {'F1-Score':<8} | {'Accuracy':<8}")
print("-" * 65)
per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)

for i, col_name in enumerate(label_cols):
    class_acc = accuracy_score(test_labels[:, i], test_preds[:, i])
    print(f"{col_name:<40} | {per_class_f1[i]:.4f}   | {class_acc:.4f}")
print("-" * 65)

print("\nGenerating download link for Teacher Logits...")
display(FileLink(npy_filename))

print("\nSaving raw PyTorch .pt file...")

pt_filename = f"{safe_name}_weights.pt"

torch.save(trainer.model.state_dict(), pt_filename)

print(f"PyTorch model saved as '{pt_filename}'.")
print("\nGenerating download link...")
display(FileLink(pt_filename))