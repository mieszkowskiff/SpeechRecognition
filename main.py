import dataset
import torch
from torch.utils.data import DataLoader
from components import EncoderBlock
from sklearn.metrics import f1_score
import tqdm

class AudioClassifier(torch.nn.Module):
    def __init__(
            self, 
            n_classes, 
            d_embedding = 128, 
            n_encoder_blocks = 6,
            d_attention_hidden = 128,
            d_ffn_hidden = 128,
            ):
        super(AudioClassifier, self).__init__()
        self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(
            d_embedding = d_embedding,
            d_attention_hidden = d_attention_hidden,
            d_ffn_hidden = d_ffn_hidden,
            ) for _ in range(n_encoder_blocks)])
        self.fc = torch.nn.Linear(d_embedding, n_classes)

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    

train_dataset = dataset.AudioDataset(root_dir="./dataset/train", transform=None)
test_dataset = dataset.AudioDataset(root_dir="./dataset/valid", transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_f1_score(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels in test_loader:
            mel_spec = mel_spec.to(device)
            labels = labels.long().to(device)

            outputs = model(mel_spec)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')  # or 'weighted', 'micro'
    print(f"F1 Score on test dataset: {f1:.4f}")
    return f1


def main():
    model = AudioClassifier(
        n_classes=8,
        d_embedding=80,
        n_encoder_blocks=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        print(f"Epoch {epoch}/{10}")
        for mel_spec, labels in tqdm.tqdm(train_loader):
            mel_spec = mel_spec.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        evaluate_f1_score(model, test_loader, device)
        






if __name__ == "__main__":
    main()