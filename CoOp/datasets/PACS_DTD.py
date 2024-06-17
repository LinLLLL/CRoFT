import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json

from collections import OrderedDict
from dassl.utils import listdir_nohidden, mkdir_if_missing
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class PACS_DTD(DatasetBase):
    dataset_dir = "PACS"
    oodset_dir = "dtd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, cfg.TEST_ENV)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        train_x = self.generate_fewshot_dataset_balance(train, num_shots=num_shots) if num_shots > 4 \
            else self.generate_fewshot_dataset(train, num_shots=num_shots)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val, num_shots=cfg.DATASET.NUM_SHOTS)

        all_domain = ['art_painting', 'cartoon', 'photo', 'sketch']
        self.base_class = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person', 'human']


        self.oodset_dir = os.path.join(root, self.oodset_dir)
        self.ood_image_dir = os.path.join(self.oodset_dir, "images")
        self.ood_split_path = os.path.join(self.oodset_dir, "split_zhou_DescribableTextures.json")
        self.split_fewshot_dir = os.path.join(self.oodset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.ood_split_path):
            ood_train, ood_val, ood_test = self.ood_read_split(self.ood_split_path, self.ood_image_dir)
        else:
            ood_train, ood_val, ood_test = self.read_and_split_data(self.ood_image_dir)
            OxfordPets.save_split(ood_train, ood_val, ood_test, self.ood_split_path, self.ood_image_dir)

        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    ood_train, ood_val = data["train"], data["val"]
            else:
                ood_train = self.generate_fewshot_dataset(ood_train, num_shots=num_shots)
                ood_val = self.generate_fewshot_dataset(ood_val, num_shots=min(num_shots, 4))
                data = {"train": ood_train, "val": ood_val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        ood_train, ood_val, ood_test = OxfordPets.subsample_classes(ood_train, ood_val, ood_test, subsample=subsample)
        ood_test = ood_train + ood_val + ood_test
        ood_test = self.generate_fewshot_dataset(ood_test, num_shots=100)
        val = self.generate_fewshot_dataset(train, num_shots=500)
        test = self.generate_fewshot_dataset(test, num_shots=500)

        super().__init__(train_x=train_x, val=val, test=test, ood_test=ood_test, alldomain=all_domain)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items, flag=None):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                if 'art_painting' in impath:
                    domain_label = 0
                    domainname = 'art_painting'
                elif 'cartoon' in impath:
                    domain_label = 1
                    domainname = 'cartoon'
                elif 'photo' in impath:
                    domain_label = 2
                    domainname = 'photo'
                else:
                    domain_label = 3
                    domainname = 'sketch'
                item = Datum(impath=impath, label=int(label), domain=domain_label, classname=classname,
                             domainname=domainname)
                # item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"], flag='test')

        return train, val, test

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.oodset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.ood_image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = Datum(impath=impath, label=label, classname=classname)
                for n in self.base_class:
                    if n not in classname:
                        items.append(item)

        return items

    def ood_read_split(self, filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, classname, impath.split("/")[-1])
                # print(impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train: n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val:], label, category))

        return train, val, test






