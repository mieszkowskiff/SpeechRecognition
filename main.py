import dataset
import torch
from torch.utils.data import DataLoader
import components
import tqdm
import json
from datetime import datetime
import os
from split_dataset import class_list


class AudioClassifier(torch.nn.Module):
    def __init__(
            self, 
            n_classes, 
            d_embedding = 128, 
            n_encoder_blocks = 6,
            d_attention_hidden = 128,
            d_ffn_hidden = 128,
            n_heads = 8
            ):
        super(AudioClassifier, self).__init__()
        self.encoder_blocks = torch.nn.ModuleList([components.EncoderBlock(
            d_embedding = d_embedding,
            d_attention_hidden = d_attention_hidden,
            d_ffn_hidden = d_ffn_hidden,
            n_heads = n_heads
            ) for _ in range(n_encoder_blocks)])
        self.fc = torch.nn.Linear(d_embedding, n_classes)

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    




def main():
    epochs = 5
    batch_size = 32
    d_embedding = 80
    d_attention_hidden = 128
    d_ffn_hidden = 128
    n_encoder_blocks = 2
    n_heads = 8
    model_type = "Transformer"
    
    train_dataset = dataset.AudioDataset(root_dir="./dataset/train", transform=None)
    test_dataset = dataset.AudioDataset(root_dir="./dataset/valid", transform=None)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)



    model = AudioClassifier(
        n_classes = len(class_list),
        d_embedding = d_embedding,
        n_encoder_blocks = n_encoder_blocks,
        d_attention_hidden = d_attention_hidden,
        d_ffn_hidden = d_ffn_hidden,
        n_heads = n_heads,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    seed = 42
    torch.manual_seed(seed)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H:%M")
    os.mkdir(f"./models/{formatted_time}")
    with open(f"./models/{formatted_time}/config.json", "w") as f:
        json.dump({
            "classes": class_list,
            "d_embedding": d_embedding,
            "n_encoder_blocks": n_encoder_blocks,
            "d_attention_hidden": d_attention_hidden,
            "d_ffn_hidden": d_ffn_hidden,
            "n_heads": n_heads,
            "torch_seed": seed,
            "model_type": model_type
        }, f, indent=4)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        for mel_spec, labels in tqdm.tqdm(train_loader):
            mel_spec = mel_spec.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        f1_score = components.evaluate_f1_score(model, test_loader, device)
        print(f"F1 Score on valid dataset: {f1_score:.2f}")

        torch.save(model.state_dict(), f"./models/{formatted_time}/model_epoch_{epoch}_f1_{f1_score:.2f}.pth")
        torch.save(optimizer.state_dict(), f"./models/{formatted_time}/optimizer_epoch_{epoch}_f1_{f1_score:.2f}.pth")
        
        






if __name__ == "__main__":
    main()