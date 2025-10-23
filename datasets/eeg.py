import os
import random
from collections import defaultdict
from .utils import Datum, DatasetBase, write_json
from .oxford_pets import OxfordPets

# Single Prompts

#template = ['{}']
#template = ['a brain EEG signal feature map showing {}.']
#template = ['a brain EEG signal featured image of {}.']
#template = ['an EEG connectivity-based featured image of {}.']
#template = ['a multi-quadrant EEG analysis visualization of {}, showing left top covariance, right top correntropy coefficient, left bottom cross-power spectral density and right bottom cohentropy coefficient.']

# Multiple prompts

template = ['{}','a brain EEG signal feature map showing {}.']
#template = ['{}','a brain EEG signal featured image of {}.']
#template = ['{}','an EEG connectivity-based featured image of {}.']
'''
template = ['{}',
'a brain EEG signal feature map showing {}.',
'a brain EEG signal featured image of {}.']
'''
'''
template = ['{}',
'a brain EEG signal feature map showing {}.',
'a brain EEG signal featured image of {}.',
'an EEG connectivity-based featured image of {}.']
'''
'''
template = ['{}',
'a brain EEG signal feature map showing {}.',
'a brain EEG signal featured image of {}.',
'an EEG connectivity-based featured image of {}.',
'a multi-quadrant EEG analysis visualization of {}, showing left top covariance, right top correntropy coefficient, left bottom cross-power spectral density and right bottom cohentropy coefficient.']
'''
'''
template = ['a brain EEG signal featured image of {}.',
'an EEG connectivity-based featured image of {}.',
'a multi-quadrant EEG analysis visualization of {}, showing left top covariance, right top correntropy coefficient, left bottom cross-power spectral density and right bottom cohentropy coefficient.']
'''
'''
template = ['{}',
'a brain EEG signal featured image of {}.',
'an EEG connectivity-based featured image of {}.',
'a multi-quadrant EEG analysis visualization of {}, showing left top covariance, right top correntropy coefficient, left bottom cross-power spectral density and right bottom cohentropy coefficient.']
'''

class EEG(DatasetBase):
    dataset_dir = 'eeg'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_eeg_16shot.json')  # Changed filename to indicate shots
        self.template = template

        if not os.path.exists(self.split_path):

            # Set a random seed based on current timeS
            random.seed()  # This uses system time as seed

            train, val, test = self._create_split(num_shots = 16)  # Change shot number here
            write_json({'train': train, 'val': val, 'test': test}, self.split_path)
            print(f"Created new split file: {self.split_path}")

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)

    def _create_split(self, num_shots):
        """Creates train/val/test splits for few-shot classification."""
        def _collect_images(class_dir, label):
            images = []
            class_names = {
                0: "normal brain activity",
                1: "abnormal brain activity"
            }
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    impath = os.path.join(os.path.basename(class_dir), img_name)
                    item = (impath, label, class_names[label])
                    images.append(item)
            return images

        # Collect all images
        class_0_images = _collect_images(os.path.join(self.image_dir, '0'), 0)
        class_1_images = _collect_images(os.path.join(self.image_dir, '1'), 1)

        # Shuffle images
        random.shuffle(class_0_images)
        random.shuffle(class_1_images)

        # Take exactly num_shots (16) samples for training from each class
        train_0 = class_0_images[:num_shots]
        train_1 = class_1_images[:num_shots]

        # Split remaining images
        remaining_0 = class_0_images[num_shots:]
        remaining_1 = class_1_images[num_shots:]

        # Split remaining images 20% val and 80% test
        val_size_0 = int(len(remaining_0) * 0.2)
        val_size_1 = int(len(remaining_1) * 0.2)

        val_0 = remaining_0[:val_size_0]
        test_0 = remaining_0[val_size_0:]
        val_1 = remaining_1[:val_size_1]
        test_1 = remaining_1[val_size_1:]

        # Combine splits
        train = train_0 + train_1
        val = val_0 + val_1
        test = test_0 + test_1

        # Shuffle combined splits
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return train, val, test