import time
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from contextlib import contextmanager
import argparse
import inspect
import pandas as pd


@contextmanager
def timer(name: str):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class Feature(metaclass=ABCMeta):
    prefix = ""
    suffix = ""
    dir = "."

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f"{self.name}_train.ftr"
        self.test_path = Path(self.dir) / f"{self.name}_test.ftr"

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = "_" + self.suffix if self.suffix else ""
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractclassmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(self.train_path)
        self.test.to_feather(self.test_path)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def get_features(namespace: dict):
    for _, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace: dict, overwrite: bool):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f"{f.train_path} and {f.test_path} are already exists")
        else:
            f.run().save()
