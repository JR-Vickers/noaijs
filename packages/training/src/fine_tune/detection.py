from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load your text file as a dataset
dataset = load_dataset('text', data_files={'train': '../data/raw/mixed_corpus.txt'})

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('../../../models/distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('../../../models/distilbert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./distilbert-detection-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Train!
trainer.train()

# Save the model
trainer.save_model("./distilbert-detection-finetuned")
tokenizer.save_pretrained("./distilbert-detection-finetuned")
