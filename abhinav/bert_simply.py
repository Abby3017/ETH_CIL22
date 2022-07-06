import pandas as pd
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_metric
import logging

logging.basicConfig(level=logging.INFO, filename='bert_simple'+ str(time.time()) +'.log',
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

file_path_all = "/cluster/home/abkumar/dataset/twitter-datasets/train_all.csv"

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

def get_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df.loc[df['label'] == -1, 'label'] = 0
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    ds = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True)
    ds = ds.remove_columns(["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    return ds


if __name__ == "__main__":
    ds = get_data(file_path_all)
    ds_train = ds['train']
    ds_test = ds['test']
    train_dataloader = DataLoader(ds_train, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(ds_test, batch_size=8)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch -  {epoch} loss - {loss.item()}")
            logging.info(f"Epoch -  {epoch} loss - {loss.item()}")

    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    res = metric.compute()
    logging.info(res)
