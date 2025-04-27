import components
import json
import glob
import torch
import dataset
import spectrogram


observation_path = "./audio/sheila/fe1916ba_nohash_1.wav"

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

observation = dataset.get_observation(
    observation_path,
    n_mels = config["dataset_parameters"]["n_mels"], 
    n_fft = config["dataset_parameters"]["n_fft"],
    hop_length = config["dataset_parameters"]["hop_length"]
)

spectrogram.create_spectrogram(observation, filename="./images/0.png")



intermediate_outputs = {}

def get_activation(name):
    def hook(model, input, output):
        intermediate_outputs[name] = output
    return hook

for block_id in range(len(model.encoder_blocks)):
    model.encoder_blocks[block_id].register_forward_hook(get_activation(block_id))


with torch.no_grad():
    output = model(observation.unsqueeze(0).to(device))
    pred = torch.argmax(output, dim=1)
    predicted_class = config["classes"][pred.item()]
    print(f"Predicted class: {predicted_class}")


for key, value in intermediate_outputs.items():
    spectrogram.create_spectrogram(value[0].cpu(), filename=f"./images/{key + 1}.png")