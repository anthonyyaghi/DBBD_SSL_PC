import os
import argparse
import numpy as np
import random
import shutil
from collections import Counter
from tqdm import tqdm

def get_all_scenes(folder):
    """List all scenes in a folder"""
    scenes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    return sorted(scenes)

def get_scene_labels(scene_folder):
    """Load segment20.npy to get semantic labels in a scene"""
    label_path = os.path.join(scene_folder, 'segment20.npy')
    labels = np.load(label_path)
    return labels

def calculate_global_class_distribution(scene_paths):
    """Calculate the global class point distribution over all scenes"""
    global_counter = Counter()
    for scene_path in tqdm(scene_paths, desc="Calculating global distribution"):
        labels = get_scene_labels(scene_path)
        global_counter.update(labels)
    return global_counter

def greedy_balanced_selection(scene_paths, target_ratio=0.5, seed=42):
    """
    Greedy selection of scenes to maintain class distribution
    """
    random.seed(seed)
    np.random.seed(seed)

    total_scenes = len(scene_paths)
    target_scenes = int(total_scenes * target_ratio)

    # Calculate global class distribution
    global_distribution = calculate_global_class_distribution(scene_paths)
    total_points = sum(global_distribution.values())

    # Desired class percentages
    target_class_ratios = {cls: cnt / total_points for cls, cnt in global_distribution.items()}

    # Prepare scene label distributions
    scene_class_distributions = {}
    for scene_path in tqdm(scene_paths, desc="Preparing scene class profiles"):
        labels = get_scene_labels(scene_path)
        scene_counter = Counter(labels)
        scene_class_distributions[scene_path] = scene_counter

    selected_scenes = []
    selected_counter = Counter()

    remaining_scenes = set(scene_paths)

    # Greedy loop to select scenes
    for _ in tqdm(range(target_scenes), desc="Greedy scene selection"):
        best_scene = None
        best_score = float('inf')

        for scene in remaining_scenes:
            temp_counter = selected_counter + scene_class_distributions[scene]
            temp_total = sum(temp_counter.values())
            temp_ratios = {cls: temp_counter[cls] / temp_total for cls in target_class_ratios.keys()}

            # Calculate deviation from global distribution
            score = sum(abs(temp_ratios.get(cls, 0) - target_class_ratios[cls]) for cls in target_class_ratios.keys())

            if score < best_score:
                best_score = score
                best_scene = scene

        # Update selection
        if best_scene:
            selected_scenes.append(best_scene)
            selected_counter.update(scene_class_distributions[best_scene])
            remaining_scenes.remove(best_scene)

    print(f"Selected {len(selected_scenes)} scenes out of {total_scenes}")
    return selected_scenes

def copy_selected_scenes(selected_scenes, output_folder):
    """Copy selected scenes to a new dataset folder"""
    os.makedirs(output_folder, exist_ok=True)
    for scene_path in tqdm(selected_scenes, desc=f"Copying to {output_folder}"):
        scene_name = os.path.basename(scene_path)
        dest_path = os.path.join(output_folder, scene_name)
        shutil.copytree(scene_path, dest_path, dirs_exist_ok=True)

def update_tasks_folder(selected_scenes_per_split, output_root):
    """Update the tasks folder to match new dataset split"""
    tasks_scenes_folder = os.path.join(output_root, "tasks", "scenes")
    os.makedirs(tasks_scenes_folder, exist_ok=True)

    for split, scenes in selected_scenes_per_split.items():
        out_txt = os.path.join(tasks_scenes_folder, f"{split}.txt")
        with open(out_txt, "w") as f:
            for scene_path in scenes:
                scene_name = os.path.basename(scene_path)
                f.write(f"{scene_name}\n")
        print(f"Updated scene list: {out_txt}")

def process_split(original_root, split, output_root, ratio, seed):
    print(f"\nProcessing {split} split")
    input_folder = os.path.join(original_root, split)
    output_folder = os.path.join(output_root, split)

    scenes = get_all_scenes(input_folder)
    scenes_full_paths = [os.path.join(input_folder, s) for s in scenes]

    selected_scenes = greedy_balanced_selection(scenes_full_paths, ratio, seed)
    
    if VISUALIZE:
        def plot_class_distribution(original_counter, selected_counter, title="Class Distribution Comparison"):
            """Plot class distributions before and after split"""
            classes = sorted(original_counter.keys())
            original_total = sum(original_counter.values())
            selected_total = sum(selected_counter.values())

            original_ratios = [original_counter[c] / original_total for c in classes]
            selected_ratios = [selected_counter.get(c, 0) / selected_total for c in classes]

            x = np.arange(len(classes))
            width = 0.35

            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, original_ratios, width, label='Original', color='gray')
            plt.bar(x + width/2, selected_ratios, width, label='Selected (split)', color='green')

            plt.xlabel('Class ID')
            plt.ylabel('Proportion of Points')
            plt.title(title)
            plt.xticks(x, classes)
            plt.legend()
            plt.grid(True, axis='y')
            plt.show()
        
        # After greedy selection is done
        selected_counter = Counter()
        for scene in selected_scenes:
            selected_counter.update(get_scene_labels(scene))

        # Original global distribution (already computed)
        global_counter = calculate_global_class_distribution(scenes_full_paths)

        # Plot
        plot_class_distribution(global_counter, selected_counter, title=f"{split.upper()} Class Distribution")
    
    copy_selected_scenes(selected_scenes, output_folder)

    return selected_scenes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and downsample .npy datasets.")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the input dataset directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the new dataset.")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of the dataset to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--visualize-distribution", action="store_true", help="Visualize the class distribution before and after splitting. Default is False.")

    args = parser.parse_args()
    
    ORIGINAL_ROOT = args.root_dir
    OUTPUT_ROOT = args.output_dir
    RATIO = args.ratio
    SEED = args.seed
    
    VISUALIZE = args.visualize_distribution
    
    if VISUALIZE:
        import matplotlib.pyplot as plt

    splits = ["train", "val"]
    selected_scenes_per_split = {}

    for split in splits:
        selected_scenes = process_split(ORIGINAL_ROOT, split, OUTPUT_ROOT, RATIO, SEED)
        selected_scenes_per_split[split] = selected_scenes

    # Fix tasks folder
    update_tasks_folder(selected_scenes_per_split, OUTPUT_ROOT)

    print("\nAll splits processed using global greedy class-proportion-aware strategy.")
