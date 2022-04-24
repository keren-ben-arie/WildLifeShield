# importing the libraries
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

FILE_PATHS = []  # file_paths for targets
TARGETS = []  # targets
LABELS = []
ANIMALS_IMAGES_MOCK = ['caged_animals_demo', 'uncaged_animals_demo']
ANIMALS_IMAGES = ['caged_animals', 'uncaged_animals']

IMAGE_COUNT = 0


class AnimalsDataset(Dataset):
    def __init__(self) -> None:
        self.data = None
        self.root_dir = "C:\\Users\\User\\PycharmProjects\\CageClassifier\\animals_images"
        self.load_directories()
        self.create_datasets()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.data.iloc[idx]["path"])
        img = Image.open(img_name).convert('RGB')
        label = np.array(self.data.iloc[idx]["label"])
        transform_image = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image = transform_image(img)
        sample = {"data": image, "label": label}
        return sample

    def load_directories(self):
        MAP = {"uncaged_animals": 0, "caged_animals": 1}
        for type in ANIMALS_IMAGES:
            animal_dir = "C:\\Users\\User\\PycharmProjects\\CageClassifier\\animals_images\\" + type
            for file in os.listdir(animal_dir):
                FILE_PATHS.append(os.path.join(animal_dir, file))
                TARGETS.append(ANIMALS_IMAGES.index(type))
                LABELS.append(MAP[type])

    def create_datasets(self):
        map_data = {"path": FILE_PATHS, "label": LABELS}
        self.data = pd.DataFrame(map_data)
