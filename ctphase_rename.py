import os
import shutil

# Simulated directory paths
source_folder_path = '/umbc/rs/nasa-access/data/viirs_data/CLDPROP_L2_VIIRS_SNPP/2017/031M'
destination_folder_path = '/umbc/rs/nasa-access/xin/cloud-phase-prediction/data/ctphase/CLDPROP_L2_VIIRS_SNPP/2017/031'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder_path):
    os.makedirs(destination_folder_path)

# Now, listing files in the directory and renaming them according to the specifications
filenames = os.listdir(source_folder_path)
i = 0
# Copying and renaming logic
for filename in filenames:
    i += 1
    print(i)
    # Split the filename on "."
    parts = filename.split(".")

    # Construct the new filename according to the given specifications
    new_filename = "CLDPROP_L2_VIIRS_SNPP" + "." + parts[1] + "." + parts[2] + ".nc"

    # Full paths for source and destination
    source_file = os.path.join(source_folder_path, filename)
    destination_file = os.path.join(destination_folder_path, new_filename)

    # Copy file to the new folder with the new filename
    shutil.copy(source_file, destination_file)
    print(f"Copied '{filename}' to '{new_filename}' in '{destination_folder_path}'")

# List the files in the destination directory after renaming
renamed_files = os.listdir(destination_folder_path)
renamed_files

