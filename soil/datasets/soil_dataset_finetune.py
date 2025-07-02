# soil_dataset_finetune.py
from torch.utils.data import Dataset
from PIL import Image
import os

class SoilDatasetFinetune(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.group_to_moisture = {
            '0_10': 5.0/35.0,
            '10_20': 15.0/35.0,
            '20_30': 25.0/35.0,
            '30_40': 35.0/35.0
        }

        for depth_folder in os.listdir(root_dir):
            depth_path = os.path.join(root_dir, depth_folder)
            moisture_value = self.group_to_moisture.get(depth_folder)

            if os.path.isdir(depth_path) and moisture_value is not None:
                for type_folder in os.listdir(depth_path):
                    type_path = os.path.join(depth_path, type_folder)
                    if os.path.isdir(type_path):
                        for class_folder in os.listdir(type_path):
                            class_path = os.path.join(type_path, class_folder)
                            if os.path.isdir(class_path):
                                for img_file in os.listdir(class_path):
                                    if img_file.endswith(".jpg") or img_file.endswith(".JPG"):
                                        img_path = os.path.join(class_path, img_file)
                                        self.samples.append((img_path, moisture_value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label