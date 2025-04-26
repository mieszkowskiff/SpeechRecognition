import dataset
import torch
from torch.utils.data import DataLoader
import components
import tqdm
import json
from datetime import datetime
import os
#from split_dataset import class_list
from torchsummary import summary

class_list = class_list = ["background", "down", "go", "left", "no", "right", "stop", "unknown", "up", "yes"]
epochs = 15

config = {
    "model_parameters": {
        "d_embedding": 64,
        "d_attention_hidden": 128,
        "d_ffn_hidden": 256,
        "n_encoder_blocks": 6,
        "n_heads": 8,
        "model_type": "Transformer",
    },
    "dataset_parameters": {
        "n_fft": 400,
        "hop_length": 320,
        "n_mels": 64
    },
    "training_parameters": {
        "batch_size": 512,
    }

}

def main():

    config["torch_seed"] = 42
    torch.manual_seed(config["torch_seed"])
    config["classes"] = class_list

    assert config["model_parameters"]["d_embedding"] == config["dataset_parameters"]["n_mels"], "d_embedding must be equal to n_mels"

    
    train_dataset = dataset.AudioDataset(
        root_dir="./dataset/train",
        class_list=class_list,
        n_mels = config["dataset_parameters"]["n_mels"], 
        n_fft = config["dataset_parameters"]["n_fft"],
        hop_length = config["dataset_parameters"]["hop_length"]
    )
    
    test_dataset = dataset.AudioDataset(
        root_dir="./dataset/valid", 
        class_list=class_list,
        n_mels = config["dataset_parameters"]["n_mels"], 
        n_fft = config["dataset_parameters"]["n_fft"],
        hop_length = config["dataset_parameters"]["hop_length"]
    )

    train_loader = DataLoader(train_dataset, batch_size = config["training_parameters"]["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size = config["training_parameters"]["batch_size"], shuffle=False, num_workers=4)

    model = components.AudioClassifier(
        n_classes = len(class_list),
        d_embedding = config["model_parameters"]["d_embedding"],
        n_encoder_blocks = config["model_parameters"]["n_encoder_blocks"],
        d_attention_hidden = config["model_parameters"]["d_attention_hidden"],
        d_ffn_hidden = config["model_parameters"]["d_ffn_hidden"],
        n_heads = config["model_parameters"]["n_heads"],
    )
    #summary(model, (1, 32, 32))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M")
    config["model_parameters"]["n_params"] = sum(p.numel() for p in model.parameters())
    os.makedirs(f"./models/{formatted_time}", exist_ok=True)
    with open(f"./models/{formatted_time}/config.json", "w") as f:
        json.dump(config, f, indent=4)

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
        print(f"F1 Score on valid dataset: {f1_score:.3f}")

        torch.save(model.state_dict(), f"./models/{formatted_time}/model_epoch_{epoch}_f1_{f1_score:.3f}.pth")
        torch.save(optimizer.state_dict(), f"./models/{formatted_time}/optimizer_epoch_{epoch}_f1_{f1_score:.3f}.pth")
        
        

if __name__ == "__main__":
    main()