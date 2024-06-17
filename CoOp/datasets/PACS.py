import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from dassl.utils import read_json, write_json


@DATASET_REGISTRY.register()
class PACS(DatasetBase):
    dataset_dir = "PACS"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, cfg.TEST_ENV)
        
        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        # train_samples = self.generate_train_samples_dataset(test, num_samples=20)
        train = self.generate_fewshot_dataset_balance(train, num_shots=num_shots) if num_shots > 4 else self.generate_fewshot_dataset(train, num_shots=num_shots)
        # val = self.generate_fewshot_dataset_balance(val, num_shots=num_shots) if num_shots > 4 else self.generate_fewshot_dataset(val, num_shots=num_shots)
        val = self.generate_fewshot_dataset(val, num_shots=num_shots)

        all_domain = ['art_painting', 'cartoon', 'photo', 'sketch']

        super().__init__(train_x=train, val=val, test=test, alldomain=all_domain)

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
                item = Datum(impath=impath, label=int(label), domain=domain_label, classname=classname, domainname=domainname)
                #item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"], flag='test')

        return train, val, test

