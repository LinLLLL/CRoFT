import os
import pickle
import numpy as np
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet46(DatasetBase):
    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, f"preprocessed-seed_{cfg.SEED}.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        all_classnames = list(self.read_classnames(text_file).values())
        self.base_classes = np.random.choice(all_classnames, int(len(all_classnames) * 0.4), replace=False).tolist()
        self.new_classes = [tmp for tmp in all_classnames if tmp not in self.base_classes]
        print('-----------------------------base_classes---------------------------')
        print(self.base_classes)
        print('-----------------------------new_classes---------------------------')
        print(self.new_classes)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)
        train = self.reset_data(train, ood=False)
        test_id = self.reset_data(test, ood=False)
        val = self.generate_fewshot_dataset(test_id, num_shots=4)
        test_ood = self.reset_data(test, ood=True)
        test_ood = self.generate_fewshot_dataset(test_ood, num_shots=1000)

        # text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = self.read_classnames(text_file)

        self.imageneta_dir = os.path.join(root, 'imagenet-adversarial', "imagenet-a")
        imagenet_a = self.read_data_variation(classnames, self.imageneta_dir)
        self.imagenetr_dir = os.path.join(root, 'imagenet-rendition', "imagenet-r")
        imagenet_r = self.read_data_variation(classnames, self.imagenetr_dir)
        self.imagenets_dir = os.path.join(root, 'imagenet-sketch', "images")
        imagenet_s = self.read_data_variation(classnames, self.imagenets_dir)
        self.imagenetv2_dir = os.path.join(root, 'imagenetv2', "imagenetv2-matched-frequency-format-val")
        imagenet_v2 = self.read_data_v2(classnames, self.imagenetv2_dir)
        imagenet_v2 = self.reset_data(imagenet_v2, ood=False)
        imagenet_s = self.reset_data(imagenet_s, ood=False)
        imagenet_a = self.reset_data(imagenet_a, ood=False)
        imagenet_r = self.reset_data(imagenet_r, ood=False)
        # val = self.generate_fewshot_dataset(imagenet_r, num_shots=4)

        super().__init__(train_x=train, val=val, test=test_id, ood_test=test_ood)

    def read_data_variation(self, classnames, image_dir):
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders]
        items = []
        label_dict = {}
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
        return items

    def read_data_v2(self, classnames, image_dir):
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                label = self.base_classes.index(classname) if classname in self.base_classes else -1 ##
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    def reset_data(self, data_source, ood=False):
        items = []
        for item in data_source:
            impath, label, classname = item.impath, item.label, item.classname
            label = self.base_classes.index(classname) if classname in self.base_classes else label ##
            item = Datum(impath=impath, label=label, classname=classname)
            if ood:
                if classname in self.new_classes:
                    items.append(item)
            else:
                if classname not in self.new_classes:
                    items.append(item)

        return items
