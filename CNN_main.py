from torchsummary import summary
from torch.amp import autocast, GradScaler

import dataset
from torch.utils.data import DataLoader

import torch
import time
import tqdm
import copy
import shutil

from split_dataset import class_list
import CNN_components as components

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
        "hop_length": 320,
        "n_mels": 64
    },
    "training_parameters": {
        "batch_size": 32,
    }

}

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 32)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 32, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    )
        ]) 
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(128, 8)

    def forward(self, x):
        x = self.init_block(x)
        x = torch.nn.functional.relu(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)              # [B, 256, 1, 1]
        x = torch.flatten(x, 1)      # [B, 256]
        x = self.classifier(x)       # [B, 10]
        return x
    
def main():
    device = torch.device("cuda")
    #if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

        
    train_dataset = dataset.AudioDataset(root_dir=".\dataset\\train")
    test_dataset = dataset.AudioDataset(root_dir=".\dataset\\valid")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    '''
    for mel_spec, labels in train_loader:
        print(mel_spec.shape, labels.shape)
    '''
    # because of using AutoAugment we have to do different transformations on train and test
    # AutoAugment only for train
    '''
    train_transform = transforms.Compose([
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47889522, 0.47227842, 0.43047404], 
            std=[0.24205776, 0.23828046, 0.25874835]
        )
    ])

    train_dataset = datasets.ImageFolder(root = "./data/train", transform = train_transform)
    test_dataset = datasets.ImageFolder(root = "./data/valid", transform = test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 100, 
        shuffle = True, 
        pin_memory = True, 
        num_workers = 2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = 256, 
        shuffle = False, 
        pin_memory=True, 
        num_workers=2
    )
    '''

    test_dataset_size = len(test_dataset)

    model = Network()
    model.to(device)
    #summary(model, (3, 32, 32))

    criterion = torch.nn.CrossEntropyLoss()
    # removed weight_decay, regularization not needed
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    # monitor best acc model and save it to chechpoint
    best_acc = 0

    for epoch in range(20):
        print(f"Using device: {device}")
        start_time = time.time()

        model.train()
        scaler = GradScaler()
        total_loss = 0
        
        for images, labels in tqdm.tqdm(train_loader):
            torch.cuda.empty_cache()
            device_images, device_labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(device_images)
                loss = criterion(outputs, device_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        end_time = time.time()

        correctly_predicted = 0
        model.eval()

        with torch.no_grad():
            print(f"Using device: {device}")
            for images, labels in tqdm.tqdm(test_loader):
                device_images, device_labels = images.to(device), labels.long().to(device)

                outputs = model(device_images)
                preditctions = torch.argmax(outputs, dim = 1)
                correctly_predicted += (preditctions == device_labels).sum().item()
        
        print(f"Time for testing: {time.time() - end_time}")
        acc = correctly_predicted / test_dataset_size
        print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, Accuracy: {acc}, Time: {end_time - start_time}s")
        if(acc>best_acc):
            torch.save(copy.deepcopy(model.state_dict()), f"./models/checkpoint/model.pth")
            torch.save(optimizer.state_dict(), f"./models/checkpoint/optimizer.pth")
            best_acc = acc
    
    # at the end of the training, type the name of the model, it will move the best model instance 
    # from chechpoint to models directory and name the model and optimizer files accordingly to the name 
    filename = input("Enter the model name to save the model and optimizer: ")
    model_name = filename
    opt_name = filename + "_optim"
    shutil.move("./models/checkpoint/model.pth", f"./models/{model_name}.pth")
    shutil.move("./models/checkpoint/optimizer.pth", f"./models/{opt_name}.pth")
    
if __name__ == "__main__":
    main()