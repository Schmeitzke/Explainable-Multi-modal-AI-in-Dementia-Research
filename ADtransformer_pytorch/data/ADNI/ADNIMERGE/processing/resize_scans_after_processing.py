import glob
import nibabel as nib
import os
import numpy as np
from nibabel.processing import resample_to_output
import multiprocessing
from tqdm import tqdm

TARGET_SHAPE = (144, 144, 144)
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)

def load_nifti_file(path: str) -> nib.Nifti1Image:
    return nib.load(path)

def resize_nifti_file(nifti_file: nib.Nifti1Image,
                        target_shape: tuple = TARGET_SHAPE,
                        target_voxel_size: tuple = TARGET_VOXEL_SIZE) -> nib.Nifti1Image:

    resampled_nii = resample_to_output(nifti_file, voxel_sizes=target_voxel_size, order=3)

    resampled_data = resampled_nii.get_fdata(dtype=np.float32)
    resampled_affine = resampled_nii.affine
    current_shape = np.array(resampled_data.shape)
    target_shape_arr = np.array(target_shape)

    spatial_dims = min(len(current_shape), len(target_shape))
    deltas = target_shape_arr[:spatial_dims] - current_shape[:spatial_dims]

    cropping = [[0, 0] for _ in range(spatial_dims)]
    padding = [[0, 0] for _ in range(spatial_dims)]
    origin_shift_voxels = np.zeros(spatial_dims)

    for i in range(spatial_dims):
        if deltas[i] < 0:
            crop_total = abs(deltas[i])
            cropping[i][0] = crop_total // 2
            cropping[i][1] = crop_total - cropping[i][0]
            origin_shift_voxels[i] = cropping[i][0]
        elif deltas[i] > 0:
            pad_total = deltas[i]
            padding[i][0] = pad_total // 2
            padding[i][1] = pad_total - padding[i][0]
            origin_shift_voxels[i] = -padding[i][0]

    slicer = tuple(slice(c[0], current_shape[i] - c[1]) for i, c in enumerate(cropping[:spatial_dims]))
    if len(current_shape) > spatial_dims:
        slicer += tuple(slice(None) for _ in range(len(current_shape) - spatial_dims))
    cropped_data = resampled_data[slicer]

    if len(current_shape) > spatial_dims:
        padding.extend([[0, 0]] * (len(current_shape) - spatial_dims))
    final_data = np.pad(cropped_data, padding, mode='constant', constant_values=0)
    final_data = final_data.astype(np.float32)

    final_affine = resampled_affine.copy()
    spatial_affine_part = final_affine[:spatial_dims, :spatial_dims]
    world_shift = spatial_affine_part @ origin_shift_voxels[:spatial_dims]
    final_affine[:spatial_dims, 3] += world_shift

    final_nii = nib.Nifti1Image(final_data, final_affine, header=resampled_nii.header)
    return final_nii

def save_nifti_file(nifti_file: nib.Nifti1Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nifti_file, path)

def process_file(file_path, output_path):
    try:
        nifti_img = load_nifti_file(file_path)
        resized_nifti_img = resize_nifti_file(nifti_img, TARGET_SHAPE, TARGET_VOXEL_SIZE)
        base_filename = os.path.basename(file_path)
        output_filename = os.path.join(output_path, base_filename)
        save_nifti_file(resized_nifti_img, output_filename)
        return True, base_filename
    except Exception as e:
        base_filename = os.path.basename(file_path)
        return False, base_filename, str(e)

def main():
    base_data_path = ""

    tasks = {
        "FLAIR": {
            "input": os.path.join(base_data_path, "FLAIR"),
            "output": os.path.join(base_data_path, "FLAIR_resized")
        },
        "MPRAGE": {
            "input": os.path.join(base_data_path, "MPRAGE"),
            "output": os.path.join(base_data_path, "MPRAGE_resized")
        },
        "T1": {
            "input": os.path.join(base_data_path, "T1"),
            "output": os.path.join(base_data_path, "T1_resized")
        }
    }

    num_processes = multiprocessing.cpu_count()

    for data_type, paths in tasks.items():
        input_path = paths["input"]
        output_path_resized = paths["output"]
        os.makedirs(output_path_resized, exist_ok=True)

        print(f"\nProcessing {data_type} files from: {input_path}")
        nifti_files = glob.glob(os.path.join(input_path, '*_3d.nii.gz'))
        print(f"Found {len(nifti_files)} files.")

        with multiprocessing.Pool(processes=num_processes) as pool:
            futures = [pool.apply_async(process_file, args=(file_path, output_path_resized)) for file_path in nifti_files]
            for future in tqdm(futures, desc=f"Processing {data_type}", unit="file"):
                result = future.get()
                if not result[0]:
                    if len(result) > 2:
                        print(f"ERROR processing file {result[1]}: {result[2]}")
                    else:
                        print(f"ERROR processing file {result[1]}: Unknown error")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()