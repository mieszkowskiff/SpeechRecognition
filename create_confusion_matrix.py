import components
import json
import torch
import glob
import dataset
import matplotlib.pyplot as plt
import tqdm


model_path = "./models/2025_04_27_14:10/"

model_epoch = 18

config = json.load(open(f"{model_path}config.json"))

model = components.AudioClassifier(
    n_classes = len(config["classes"]),
    d_embedding = config["model_parameters"]["d_embedding"],
    n_encoder_blocks = config["model_parameters"]["n_encoder_blocks"],
    d_attention_hidden = config["model_parameters"]["d_attention_hidden"],
    d_ffn_hidden = config["model_parameters"]["d_ffn_hidden"],
    n_heads = config["model_parameters"]["n_heads"],
    positional_encoding = config["model_parameters"]["positional_encoding"]
)

filename = glob.glob(f"{model_path}model_epoch_{model_epoch}_f1_*.pth")[0] 
model.load_state_dict(torch.load(filename))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_dataset = dataset.AudioDataset(
    root_dir="./dataset/test", 
    n_mels = config["dataset_parameters"]["n_mels"], 
    n_fft = config["dataset_parameters"]["n_fft"],
    hop_length = config["dataset_parameters"]["hop_length"]
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size = config["training_parameters"]["batch_size"], 
    shuffle=False, 
    num_workers=4
)

confusion_matrix = torch.zeros(len(config["classes"]), len(config["classes"])).to(device)

for image, label in tqdm.tqdm(test_loader):
    image = image.to(device)
    label = label.to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        for t, p in zip(label.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix.cpu(), interpolation='nearest', cmap=plt.cm.Blues)
conf_matrix_np = confusion_matrix.cpu().numpy()
for i in range(conf_matrix_np.shape[0]):
    for j in range(conf_matrix_np.shape[1]):
        plt.text(j, i, format(int(conf_matrix_np[i, j]), 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix_np[i, j] > conf_matrix_np.max() / 2 else "black")
plt.title('Confusion Matrix')
tick_marks = range(len(config["classes"]))
plt.xticks(tick_marks, config["classes"], rotation=45)
plt.yticks(tick_marks, config["classes"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(f"./images/confusion_matrix.png")

print(components.evaluate_f1_score(model, test_loader, device))


