import os
import random
import clip
import yaml
import torch
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
from main import run_tip_adapter, run_tip_adapter_F, get_arguments
from datetime import datetime


def main():
    # Configuration
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config)), f"Config file '{args.config}' does not exist."
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Create cache directory
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")


    # Load CLIP model
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Set random seeds
    random.seed(1)
    torch.manual_seed(1)
    
    # Prepare dataset
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    # Calculate sizes
    total_images = len(dataset.train_x) + len(dataset.val) + len(dataset.test)
    train_images = len(dataset.train_x)
    val_images = len(dataset.val) 
    test_images = len(dataset.test)

    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total Images: {total_images}")
    print(f"Training Set: {train_images} images ({train_images/total_images*100:.2f}%)")
    print(f"Validation Set: {val_images} images ({val_images/total_images*100:.2f}%)")
    print(f"Test Set: {test_images} images ({test_images/total_images*100:.2f}%)")


    # Count images per class in training set
    class_counts = {}
    for item in dataset.train_x:
        class_name = item.classname
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print("\nTraining Set Class Distribution:")
    print("-" * 50)
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count} images")
    
    print(f"\nNumber of Classes: {len(class_counts)}")

    # Create data loaders
    val_loader = build_data_loader(
        data_source=dataset.val,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False
    )
    
    test_loader = build_data_loader(
        data_source=dataset.test,
        batch_size=64,
        is_train=False,
        tfm=preprocess,
        shuffle=False
    )

    # tfm = preprocess (uses CLIP's default preprocessing)
    train_loader_cache = build_data_loader(
        data_source=dataset.train_x,
        batch_size=2,
        tfm=preprocess,
        is_train=True,
        shuffle=False
    )
    
    train_loader_F = build_data_loader(
        data_source=dataset.train_x,
        batch_size=2,
        tfm=preprocess,
        is_train=True,
        shuffle=True
    )
    
    # Get CLIP's text features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Build cache model
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Load validation features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # Run basic Tip-Adapter
    print("\nRunning Tip-Adapter...")
    run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, 
                   test_features, test_labels, clip_weights)
    
    # Variable for store the start time 
    start_time = datetime.now()

    # Run Tip-Adapter-F (Fine-tuned version)
    print("\nRunning Tip-Adapter-F...")
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels,
                     test_features, test_labels, clip_weights, clip_model, train_loader_F)
    
    # Variable for end time and overall time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTests completed. Total duration for fine-tuning: {duration}")

if __name__ == '__main__':
    main()