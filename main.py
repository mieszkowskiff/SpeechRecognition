import dataset
from torch.utils.data import DataLoader


train_dataset = dataset.AudioDataset(root_dir="./dataset/train", transform=None)
test_dataset = dataset.AudioDataset(root_dir="./dataset/valid", transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


for mel_spec, labels in train_loader:
    print(mel_spec.shape, labels.shape)