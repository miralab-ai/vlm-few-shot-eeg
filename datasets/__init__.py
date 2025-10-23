from .oxford_pets import OxfordPets
from .eeg import EEG


dataset_list = {
                "oxford_pets": OxfordPets,
                "eeg": EEG,
                }


def build_dataset(dataset, root_path, shots):
    return dataset_list[dataset](root_path, shots)