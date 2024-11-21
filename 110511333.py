from tokenizer import Tokenizer
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import seaborn as sns


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=64, output_dim=3):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return self.softmax(output)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_data_loader(train_df, val_df, tokenizer, batch_size=32):
    train_token = tokenizer.batch_encode(train_df['text'].tolist())
    val_token = tokenizer.batch_encode(val_df['text'].tolist())

    train_token = torch.tensor(train_token, dtype=torch.long)
    val_token = torch.tensor(val_token, dtype=torch.long)
    train_label = torch.tensor(train_df['label'].values, dtype=torch.long)
    val_label = torch.tensor(val_df['label'].values, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(train_token, train_label)
    val_dataset = torch.utils.data.TensorDataset(val_token, val_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    device = get_device()
    print(f"Training on : {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=5)
    # Record the training process
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for token, label in train_loader:
            token, label = token.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(token)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(100*train_loss/len(train_loader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for token, label in val_loader:
                token, label = token.to(device), label.to(device)
                output = model(token)
                loss = criterion(output, label)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            
            val_accuracy = 100 * correct / total
            val_losses.append(100*val_loss/len(val_loader))
            val_accuracies.append(val_accuracy)
            print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {val_accuracy}%, LR: {optimizer.param_groups[0]['lr']}")
            scheduler.step(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    return model, train_losses, val_losses, val_accuracies

def plot_seaborn_metrics(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 構建 DataFrame
    data = {
        "Epoch": list(epochs) * 3,
        "Metric": ["Train Loss"] * len(epochs) + ["Val Loss"] * len(epochs) + ["Val Accuracy"] * len(epochs),
        "Value": train_losses + val_losses + val_accuracies
    }
    df = pd.DataFrame(data)

    # 繪製損失與準確率曲線
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Epoch", y="Value", hue="Metric", marker="o")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()    

def main(train_df, val_df, tokenizer):
    # Create data loaders
    train_loader, val_loader = create_data_loader(train_df, val_df, tokenizer)

    model = SentimentClassifier(vocab_size=tokenizer.get_vocab_size())

    trained_model, train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader)
    plot_seaborn_metrics(train_losses, val_losses, val_accuracies)    

    return trained_model

def prediction_model(model, test_df, tokenizer):
    device = get_device()
    model = model.to(device)
    model.eval()
    test_token = tokenizer.batch_encode(test_df['text'].tolist())
    test_token = torch.tensor(test_token, dtype=torch.long)
    test_dataset = torch.utils.data.TensorDataset(test_token)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = []
    with torch.no_grad():
        for token in test_loader:
            token = token[0].to(device)
            output = model(token)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

if __name__ == "__main__":
    # Test the Tokenizer class
    # example_text = ["Hello, I'm Charlie.", "I love traveling."]
    # tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=16)
    # tokenized_text = tokenizer.batch_encode(example_text)
    # for text, token_ids in zip(example_text, tokenized_text):
    #     print(f"\n[Text]: {text}")
    #     print(f"[Token_ids]: {token_ids}")
    #     decoded = " ".join([tokenizer.id_to_token.get(token_id, " ") for token_id in token_ids])
    #     print(f"[Decode_text]: {decoded}")
    labels = {'neutral': 0, 'positive': 1, 'negative': 2}
    # Use Pandas to load the training data in ./dataset/kaggle/train.jsonl
    # example data format: {"id": the index of data, "text": posts in string format, "label": sentiment label in string format(neutral, positive, negative)}
    # If 110511333.pth file is not found, train the model and save the model
    if os.path.exists('110511333.pth') == False:
        df = pd.read_json('./dataset/kaggle/train.jsonl', lines=True)
        df['label'] = df['label'].map(labels) # convert sentiment label to integer
        
        # Split the data into training and validation sets
        train_df = df.sample(frac=0.8, random_state=0)
        val_df = df.drop(train_df.index)
        
        # Load the Tokenizer class
        tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=64)

        # Train model
        model = main(train_df, val_df, tokenizer)

        # Save the model
        torch.save(model.state_dict(), '110511333.pth')
    else: # If 110511333.pth file is found, load the model
        tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=64)
        # Load the model
        model = SentimentClassifier(vocab_size=tokenizer.get_vocab_size())
        model.load_state_dict(torch.load('110511333.pth', weights_only = True))
    
    # Test the model
    # Use Pandas to load the test data in ./dataset/kaggle/test.jsonl
    # example data format: {"id": the index of data, "text": posts in string format}
    test_df = pd.read_json('./dataset/kaggle/test.jsonl', lines=True)
    # Load the Tokenizer class
    tokenizer = Tokenizer(vocab='token_to_index.pkl', max_length=64)
    # Make predictions
    predictions = prediction_model(model, test_df, tokenizer)
    # Save the predictions to a file
    reverse_lables = {v: k for k, v in labels.items()}
    test_df['label'] = predictions
    test_df['label'] = test_df['label'].map(reverse_lables)
    result_df = test_df.drop(columns=['text'])
    result_df.to_csv('110511333.csv', index=False)

    




