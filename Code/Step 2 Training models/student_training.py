
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from IPython.display import FileLink, display

print("=== INITIALIZING END-TO-END DISTILLATION PIPELINE ===")

print("\nLoading datasets and Teacher Logits...")
train_df = pd.read_csv("/kaggle/input/datasets/cyskill/datainput3/augmented_training.tsv", sep='\t')
val_df = pd.read_csv("/kaggle/input/datasets/cyskill/datainput3/merged_val.tsv", sep='\t')
test_df = pd.read_csv("/kaggle/input/datasets/cyskill/datainput3/merged_test.tsv", sep='\t')

roberta_logits = np.load("/kaggle/input/datasets/cyskill/datainput3/roberta-large_logits.npy")
deberta_logits = np.load("/kaggle/input/datasets/cyskill/datainput3/microsoft_deberta-large_logits.npy")

print("Blending Teacher knowledge into Ensemble Target...")
ensemble_logits = (roberta_logits + deberta_logits) / 2.0

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

print("Upsampling minority classes (Text + Logits) to 2000 minimum...")
np.random.seed(42)
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
    upsampled_logits = ensemble_logits[upsample_indices]
    ensemble_logits = np.vstack((ensemble_logits, upsampled_logits))

shuffle_idx = np.random.permutation(len(train_df))
train_df = train_df.iloc[shuffle_idx].reset_index(drop=True)
ensemble_logits = ensemble_logits[shuffle_idx]

print(f"Dataset size after massive upsampling: {len(train_df)} rows")

def process_df(df):
    df['labels'] = df[label_cols].values.tolist()
    return Dataset.from_pandas(df[['Text', 'labels']])

train_dataset = process_df(train_df)
val_dataset = process_df(val_df)
test_dataset = process_df(test_df)

train_dataset = train_dataset.add_column("soft_labels", ensemble_logits.tolist())

STUDENT_MODEL = "roberta-base"
print(f"\nLoading tokenizer for Student ({STUDENT_MODEL})...")
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)

def student_tokenize(examples):
    return student_tokenizer(examples["Text"], truncation=True, max_length=128)

print("Tokenizing data...")
encoded_train = train_dataset.map(student_tokenize, batched=True)
encoded_val = val_dataset.map(student_tokenize, batched=True)
encoded_test = test_dataset.map(student_tokenize, batched=True)
encoded_train = encoded_train.remove_columns(["Text"])
encoded_val = encoded_val.remove_columns(["Text"])
encoded_test = encoded_test.remove_columns(["Text"])

import torch.nn.functional as F

class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha=0.8, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha 
        self.temperature = temperature
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_weights = torch.ones([38]).to(device) * 5.0 
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("soft_labels", None)
        
        outputs = model(**inputs)
        student_logits = outputs.logits

        hard_loss = self.bce_loss(student_logits, labels.float())
        
        if teacher_logits is not None:
            teacher_probs = torch.sigmoid(teacher_logits.float() / self.temperature)
            
            soft_loss = F.binary_cross_entropy_with_logits(
                student_logits / self.temperature, 
                teacher_probs
            ) * (self.temperature ** 2)
            
            loss = (self.alpha * hard_loss) + ((1. - self.alpha) * soft_loss)
        else:
            loss = hard_loss 

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.array(labels).astype(int)
    
    predictions = (torch.sigmoid(torch.tensor(logits)).numpy() > 0.3).astype(int)
    
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": accuracy}

print(f"\nLoading {STUDENT_MODEL} architecture...")
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL, num_labels=38, problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./student_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
    learning_rate=3e-5,
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True, 
    logging_steps=100,
    remove_unused_columns=False 
)

student_trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_val,
    processing_class=student_tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=student_tokenizer)
)

print("\nStarting Ensemble Distillation...")
student_trainer.train()

print("\nTraining complete. Saving PyTorch model weights...")
pt_filename = "valuescope_student_model.pt"
torch.save(student_trainer.model.state_dict(), pt_filename)
print(f"Saved '{pt_filename}' successfully.")

print("\n" + "="*65)
print("FINAL TEST SET EVALUATION (STUDENT MODEL)")
print("="*65)

test_results = student_trainer.predict(encoded_test)
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

print("\nGenerating download link for the .pt model file...")
display(FileLink(pt_filename))