import torch
import time
import tqdm
import copy
import shutil
import sys

from torchsummary import summary
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import Subset

sys.path.append("..")
from utils.CNN_dataset import DynamicResolutionDataset
import utils.CNN_components as components
sys.path.remove("..")

class_list = ["down", "up", "go", "stop", "right", "left", "no", "yes", "on", "off", "background", "unknown"]

def save_txt(history, file_name):
    with open(file_name, "w") as f:
        for it in history:
            line = ' '.join(map(str, it))
            f.write(line + "\n")

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.init_block = components.InitBlock(out_channels = 64)
        self.blocks = torch.nn.ModuleList([
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 64,
                        out_channels = 64,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 64, 
                        internal_channels = 128,
                        out_channels = 128,
                        bypass = True,
                        max_pool = True,
                        batch_norm = True,
                        dropout = False
                    ),
            components.Module(
                        conv_blocks_number = 0,
                        in_channels = 128, 
                        internal_channels = 256,
                        out_channels = 256,
                        bypass = True,
                        max_pool = False,
                        batch_norm = True,
                        dropout = False
                    )
        ]) 
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(256, 12)

    def forward(self, x):
        x = self.init_block(x)
        for it in self.blocks:
            x = it(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #n_mels_grid = [256, 128, 64, 32]
    n_mels_grid = [32, 64, 128, 256]
    epochs = 20
    batch_size = 64
    for run in range(3):
        F1_history = [[0 for _ in range(epochs)] for _ in range(len(n_mels_grid))]
        loss_history = [[0 for _ in range(epochs)] for _ in range(len(n_mels_grid))]
        time_history = [[0 for _ in range(epochs)] for _ in range(len(n_mels_grid))]
        run_directory_path = "./run_" + str(run) + "/"

        for i in range(len(n_mels_grid)):
            train_dataset = DynamicResolutionDataset(
            root_dir="../dataset_high_res_preprocessed/train",
            class_list=class_list,
            target_n_mels=n_mels_grid[i]
        )
            #train_dataset = Subset(train_dataset, range(4096))
            
            test_dataset = DynamicResolutionDataset(
            root_dir="../dataset_high_res_preprocessed/valid",
            class_list=class_list,
            target_n_mels=n_mels_grid[i]
        )
            #test_dataset = Subset(test_dataset, range(4096))

            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

            model = Network()
            model.to(device)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
            best_F1 = 0

            for epoch in range(epochs):
                print(f"Using device: {device}")
                start_time = time.time()

                model.train()
                scaler = GradScaler()
                total_loss = 0
                
                for samples, labels in tqdm.tqdm(train_loader):
                    device_samples, device_labels = samples.to(device), labels.long().to(device)
                    optimizer.zero_grad()

                    with autocast(device_type='cuda'):
                        outputs = model(device_samples)
                        loss = criterion(outputs, device_labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    
                end_time = time.time()

                epoch_f1 = components.evaluate_f1_score(model=model, test_loader=test_loader, device=device)
                F1_history[i][epoch] = epoch_f1
                loss_history[i][epoch] = total_loss
                time_history[i][epoch] = end_time - start_time

                print(f"Time for testing: {time.time() - end_time}")
                print(f"Epoch {epoch + 1}, Training Loss: {total_loss}, F1_scrore: {epoch_f1}, Time: {end_time - start_time}s")
                print()
                if(epoch_f1>best_F1):
                    torch.save(copy.deepcopy(model.state_dict()), run_directory_path + "models/checkpoint/model.pth")
                    torch.save(optimizer.state_dict(), run_directory_path + "models/checkpoint/optimizer.pth")
                    best_F1 = epoch_f1
        
            # at the end of the training, type the name of the model, it will move the best model instance 
            # from chechpoint to models directory and name the model and optimizer files accordingly to the name 
            filename = "mels_" + str(n_mels_grid[i])
            model_name = filename + "_model"
            opt_name = filename + "_optim"
            shutil.move(run_directory_path + "models/checkpoint/model.pth", run_directory_path + "models/" + str(model_name) + ".pth")
            shutil.move(run_directory_path + "models/checkpoint/optimizer.pth", run_directory_path + "models/" + str(opt_name) + ".pth")

            print(f"History for n_mels = {n_mels_grid[i]}")
            print("F1_score:")
            print(F1_history[i])
            print("Loss:")
            print(loss_history[i])
            print("Time:")
            print(time_history[i])
            print()
        
        print("Saving history data to history.txt")
        save_txt(F1_history, run_directory_path + "F1_history.txt")
        save_txt(loss_history, run_directory_path + "loss_history.txt")
        save_txt(time_history, run_directory_path + "time_history.txt")
        print(f"Data gathered. Run no.{run} performed succesfully.")

if __name__ == "__main__":
    main()