import os
import shutil
import pandas as pd

base_data_dir = "data/all_data"
master_images_dir = os.path.join(base_data_dir, "all_images")
master_labels_filepath = os.path.join(base_data_dir, "combined_labels.csv")

os.makedirs(master_images_dir, exist_ok=True)

all_subdirectories = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d)) and d.startswith("run_v")]

master_labels_data = []
global_image_counter = 0

for subdir_name in sorted(all_subdirectories): # Sort to ensure consistent order
    subdir_path = os.path.join(base_data_dir, subdir_name)
    subdir_images_path = os.path.join(subdir_path, "images")
    subdir_labels_path = os.path.join(subdir_path, "labels.csv")

    if not os.path.exists(subdir_labels_path):
        print(f"Warning: labels.csv not found in {subdir_name}. Skipping.")
        continue

    # Load labels from the current subdirectory
    current_subdir_labels = pd.read_csv(subdir_labels_path)

    for index, row in current_subdir_labels.iterrows():
        original_filename = row['image_filename']
        steering_angle = row['steering_angle']

        original_image_path = os.path.join(subdir_images_path, original_filename)

        # Generate a unique filename (e.g., using a global counter)
        # Or you can use: new_filename = f"{subdir_name}_{original_filename}"
        new_filename = f"image_{global_image_counter:06d}.png" # 06d for 6 digits, adjust as needed for total samples
        new_image_path = os.path.join(master_images_dir, new_filename)

        # Copy the image to the master directory
        if os.path.exists(original_image_path):
            shutil.copy(original_image_path, new_image_path)
            master_labels_data.append({'image_filename': new_filename, 'steering_angle': steering_angle})
            global_image_counter += 1
        else:
            print(f"Warning: Image {original_image_path} not found. Skipping.")

# Save the master labels file
master_labels_df = pd.DataFrame(master_labels_data)
master_labels_df.to_csv(master_labels_filepath, index=False)

print(f"Consolidation complete. Total unique images: {global_image_counter}")
print(f"Master labels saved to: {master_labels_filepath}")
print(f"All images saved to: {master_images_dir}")
