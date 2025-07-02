# soil_dataset_ssl.py
from torch.utils.data import Dataset
from PIL import Image
import os


class SoilDatasetSSL(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for depth_folder in os.listdir(root_dir):
            depth_path = os.path.join(root_dir, depth_folder)
            if os.path.isdir(depth_path):
                for type_folder in os.listdir(depth_path):
                    type_path = os.path.join(depth_path, type_folder)
                    if os.path.isdir(type_path):
                        for class_folder in os.listdir(type_path):
                            class_path = os.path.join(type_path, class_folder)
                            if os.path.isdir(class_path):
                                for img_file in os.listdir(class_path):
                                    if img_file.endswith(".jpg") or img_file.endswith(".JPG"):
                                        img_path = os.path.join(class_path, img_file)
                                        self.samples.append(img_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            return self.transform(image)
        else:
            return image