from torchvision import datasets, transforms
from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import torch
import tqdm
import sys

sys.path.append("..")
from utils.CNN_dataset import PreprocessedAudioDataset
sys.path.remove("..")
       
sys.path.append("..")
import model_init.model_10_20 as model_arch
sys.path.remove("..")

choose_model = "model_10_20"

#class_list = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
#class_list = ["background", "down", "go", "left", "no", "right", "stop", "up", "yes"]
class_list = ["background", "down", "go", "left", "no", "right", "stop", "unknown", "up", "yes"]
class_list = ["down", "up", "go", "stop", "right", "left", "no", "yes", "on", "off", "background", "unknown"]
  
config = {
    "model_parameters": {
        "d_embedding": 64,
        "d_attention_hidden": 64,
        "d_ffn_hidden": 128,
        "n_encoder_blocks": 4,
        "n_heads": 12,
        "model_type": "Transformer",
    },
    "dataset_parameters": {
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 80
    },
    "training_parameters": {
        "batch_size": 512,
    }

}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = PreprocessedAudioDataset(
        root_dir="../dataset_preprocessed/test",
        class_list=class_list
    )

    test_loader = DataLoader(test_dataset, batch_size = config["training_parameters"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset_size = len(test_dataset)

    model = model_arch.Network()
    model_path = "../models/" + choose_model + ".pth"
    conf_mat_name = "./conf_matrix/conf_mat_" + choose_model + ".png"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    #summary(model, (1, 32, 32))
    
    model.eval()
    correctly_predicted = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sounds, labels in tqdm.tqdm(test_loader):
            device_sounds = sounds.to(device)
            device_labels = labels.long().to(device)

            outputs = model(device_sounds)
            predictions = torch.argmax(outputs, dim=1)

            correctly_predicted += (predictions == device_labels).sum().item()

            # Save for confusion matrix
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(device_labels.cpu().numpy())

    print(f"Accuracy: {correctly_predicted / test_dataset_size:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_list, yticklabels=class_list)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save to path
    plt.savefig(conf_mat_name)
    plt.close()

if __name__ == "__main__":
    main()
    