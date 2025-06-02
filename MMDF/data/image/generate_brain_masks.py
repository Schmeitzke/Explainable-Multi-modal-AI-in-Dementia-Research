import logging
import multiprocessing
from pathlib import Path
import os
from functools import partial
from typing import List, Dict, Tuple

import tqdm
import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import scipy.ndimage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("generate_brain_masks")

def run_bet_2d_mask(
    input_2d_nifti: str,
    output_mask_nifti: str,
) -> bool:
    """Creates a brain mask from a 3-channel 2D NIfTI slice using Otsu thresholding and connected components."""
    if not os.path.exists(input_2d_nifti):
        logger.error(f"Input file for mask generation not found: {input_2d_nifti}")
        return False

    if os.path.exists(output_mask_nifti):
        logger.debug(f"Final mask file {output_mask_nifti} already exists. Skipping generation.")
        return True

    try:
        logger.debug(f"Loading 3-channel NIfTI: {input_2d_nifti}")
        img_3channel = nib.load(input_2d_nifti)
        data_3channel = img_3channel.get_fdata()
        original_affine = img_3channel.affine
        original_header = img_3channel.header

        if data_3channel.ndim not in [3, 4] or (data_3channel.ndim == 3 and data_3channel.shape[-1] != 3) or (data_3channel.ndim == 4 and data_3channel.shape[-1] != 3):
            if data_3channel.ndim == 2:
                logger.warning(f"Input {input_2d_nifti} seems to be 2D already. Using as is for masking.")
                single_channel_data = data_3channel.astype(np.float32)
                if original_affine.shape == (4, 4):
                    new_affine = original_affine
                else:
                    new_affine = np.eye(4)
                    new_affine[:2, :2] = np.eye(2)
                    new_affine[:2, 3] = np.array([0,0])
            else:
                logger.error(f"Input NIfTI {input_2d_nifti} is not a 3-channel or 2D image as expected (shape: {data_3channel.shape}). Cannot create mask.")
                return False
        else:
            if data_3channel.ndim == 4 and data_3channel.shape[2] == 1:
                single_channel_data = data_3channel[:,:,0,0].astype(np.float32)
            else:
                single_channel_data = data_3channel[:,:,0].astype(np.float32)

            new_affine = np.eye(4)
            try:
                new_affine[0,0] = original_affine[0,0]
                new_affine[1,1] = original_affine[1,1]
                new_affine[0,3] = original_affine[0,3]
                new_affine[1,3] = original_affine[1,3]
                if original_affine.shape[0] > 2 and original_affine.shape[1] > 2:
                    new_affine[2,2] = original_affine[2,2]
                    new_affine[2,3] = original_affine[2,3]
            except IndexError:
                logger.warning(f"Could not fully populate affine from original {original_affine.shape} for {input_2d_nifti}. Using defaults.")
            new_affine[3,3] = 1.0

        if np.all(single_channel_data == 0):
             logger.warning(f"Input slice data for {input_2d_nifti} is all zeros. Creating an empty mask.")
             empty_mask_data = np.zeros_like(single_channel_data, dtype=np.uint8)
             mask_img = nib.Nifti1Image(empty_mask_data, new_affine, header=original_header)
             nib.save(mask_img, output_mask_nifti)
             return True

        try:
            min_val = single_channel_data.min()
            max_val = single_channel_data.max()
            if max_val > min_val:
                otsu_data = single_channel_data
                if min_val < 0:
                    otsu_data = single_channel_data - min_val
                threshold = threshold_otsu(otsu_data)
                if min_val < 0:
                    threshold += min_val
            elif max_val > 0:
                threshold = max_val * 0.5
            else:
                logger.warning(f"Image {input_2d_nifti} is constant and non-positive. Creating empty mask.")
                threshold = np.inf

            binary_mask = single_channel_data > threshold
        except Exception as otsu_err:
            logger.error(f"Otsu thresholding failed for {input_2d_nifti}: {otsu_err}. Falling back to simple >0 threshold.", exc_info=True)
            binary_mask = single_channel_data > 0

        if not np.any(binary_mask):
            logger.warning(f"Thresholding resulted in an empty mask for {input_2d_nifti}. Saving empty mask.")
            mask_data = np.zeros_like(single_channel_data, dtype=np.uint8)
        else:
            labeled_mask, num_labels = scipy.ndimage.label(binary_mask)
            if num_labels < 1:
                logger.warning(f"Connected components labeling found no components despite non-empty binary mask for {input_2d_nifti}.")
                mask_data = binary_mask.astype(np.uint8)
            elif num_labels == 1:
                logger.debug(f"Only one component found for {input_2d_nifti}. Filling holes.")
                mask_data = scipy.ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
            else:
                logger.debug(f"Found {num_labels} components for {input_2d_nifti}. Keeping largest.")
                component_sizes = np.bincount(labeled_mask.ravel())
                if len(component_sizes) > 1:
                    largest_label = np.argmax(component_sizes[1:]) + 1
                    largest_component_mask = (labeled_mask == largest_label)
                    mask_data = scipy.ndimage.binary_fill_holes(largest_component_mask).astype(np.uint8)
                    logger.debug(f"Kept component {largest_label} with size {component_sizes[largest_label]}.")
                else:
                    logger.warning(f"np.bincount only found background label for {input_2d_nifti}. Using binary mask directly.")
                    mask_data = binary_mask.astype(np.uint8)

        mask_img = nib.Nifti1Image(mask_data, new_affine, header=original_header)
        mask_img.header.set_data_dtype(mask_data.dtype)
        nib.save(mask_img, output_mask_nifti)
        logger.debug(f"Saved final mask to {output_mask_nifti}")
        return True

    except Exception as e:
        logger.error(f"Error generating mask for {input_2d_nifti}: {e}", exc_info=True)
        if os.path.exists(output_mask_nifti):
            try:
                os.remove(output_mask_nifti)
            except OSError:
                 logger.warning(f"Could not remove potentially incomplete output file {output_mask_nifti}")
        return False

def process_single_slice_for_mask(
    args_tuple: Tuple[str, str],
) -> Dict[str, str]:
    """Processes a single 2D NIfTI slice to generate its brain mask."""
    input_file_path, output_mask_path = args_tuple
    status = {'input': input_file_path, 'output': output_mask_path, 'status': 'failed_init'}
    
    try:
        logger.info(f"Generating mask for: {Path(input_file_path).name}")
        if os.path.exists(output_mask_path):
            logger.info(f"Output mask {output_mask_path} already exists. Skipping.")
            status['status'] = 'skipped_exists'
            return status

        if not run_bet_2d_mask(input_file_path, output_mask_path):
            status['status'] = 'failed_mask_generation'
            return status

        logger.info(f"SUCCESS: {Path(input_file_path).name} -> {Path(output_mask_path).name}")
        status['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Unhandled exception during mask generation for {input_file_path}: {e}", exc_info=True)
        status['status'] = 'failed_exception'
    
    return status

def find_processed_slice_files(base_dirs: List[str]) -> List[Path]:
    """Finds all files matching '*_coronal_...x...x3.nii.gz' in the subdirectories (T1, FLAIR, MPRAGE) of the base_dirs."""
    all_files: List[Path] = []
    for base_dir_str in base_dirs:
        base_path = Path(base_dir_str)
        if not base_path.is_dir():
            logger.warning(f"Provided base path {base_dir_str} is not a directory or does not exist. Skipping.")
            continue
            
        for subdir_name in ["T1", "FLAIR", "MPRAGE"]:
            subdir_path = base_path / subdir_name
            if not subdir_path.is_dir():
                logger.warning(f"Subdirectory {subdir_name} not found in {base_dir_str}. Skipping.")
                continue
                
            found_files = list(subdir_path.glob("*_coronal_*x*x3.nii.gz"))
            if found_files:
                logger.info(f"Found {len(found_files)} processed slices in {subdir_path}.")
                all_files.extend(found_files)
            else:
                logger.info(f"No processed slices found in {subdir_path}.")
                
    return all_files

def main():         
    current_script_path = Path(__file__).resolve()
    calculated_project_root = current_script_path.parents[3] 

    hardcoded_input_base_dir_relative = "MMDF/data/image/processed_coronal_slices"
    
    script_input_base_dir = str(calculated_project_root / hardcoded_input_base_dir_relative)
    script_num_workers = None
    script_log_level = "INFO"

    logger.setLevel(getattr(logging, script_log_level.upper()))

    logger.info(f"SCRIPT CONFIGURATION:")
    logger.info(f"  Input Base Directory (contains T1, FLAIR, MPRAGE subdirs): {script_input_base_dir}")
    logger.info(f"  Num Workers: {'Default (CPU count - 1 or 1)' if script_num_workers is None else script_num_workers}")
    logger.info(f"  Log Level: {script_log_level}")

    input_files_to_process = find_processed_slice_files([script_input_base_dir])
    if not input_files_to_process:
        logger.info("No processed slice files found. Nothing to do. Exiting.")
        return

    tasks_for_processing: List[Tuple[str, str]] = []
    for file_path_obj in input_files_to_process:
        output_mask_name = file_path_obj.name.replace(".nii.gz", "_mask.nii.gz")
        output_mask_path = str(file_path_obj.parent / output_mask_name)
        tasks_for_processing.append((str(file_path_obj), output_mask_path))

    if not tasks_for_processing:
        logger.error("No tasks could be prepared for mask generation. Exiting.")
        return
        
    logger.info(f"Prepared {len(tasks_for_processing)} file(s) for mask generation.")
    
    num_workers_resolved = script_num_workers
    if num_workers_resolved is None:
        cpu_count = os.cpu_count()
        num_workers_resolved = max(1, cpu_count - 1) if cpu_count and cpu_count > 1 else 1
    num_workers_resolved = max(1, num_workers_resolved)
    
    logger.info(f"Using {num_workers_resolved} worker process(es).")
    
    process_func_with_fixed_args = partial(
        process_single_slice_for_mask,
    )
    
    all_results: List[Dict[str, str]] = []
    if num_workers_resolved > 1 and len(tasks_for_processing) > 1:
        logger.info(f"Starting parallel mask generation with {num_workers_resolved} workers.")
        pool = None
        try:
            pool = multiprocessing.Pool(processes=num_workers_resolved)
            all_results = list(tqdm(
                pool.imap_unordered(process_func_with_fixed_args, tasks_for_processing),
                total=len(tasks_for_processing),
                desc="Generating Masks"
            ))
        finally:
            if pool:
                pool.close()
                pool.join()
    else: 
        logger.info("Running mask generation in single-process mode.")
        for task_args_tuple in tqdm(tasks_for_processing, desc="Generating Masks"):
            all_results.append(process_func_with_fixed_args(task_args_tuple))

    success_count = sum(1 for r in all_results if r.get('status') == 'success')
    skipped_count = sum(1 for r in all_results if r.get('status') == 'skipped_exists')
    failed_count = len(all_results) - success_count - skipped_count
    
    logger.info("--- Mask Generation Summary ---")
    logger.info(f"Total slice files considered: {len(tasks_for_processing)}")
    logger.info(f"Successfully generated masks: {success_count}")
    logger.info(f"Skipped (mask already existed): {skipped_count}")
    logger.info(f"Failed to generate masks: {failed_count}")

    if failed_count > 0:
        logger.warning("Details for failed mask generations:")
        for result_item in all_results:
            if result_item.get('status') not in ['success', 'skipped_exists']:
                logger.warning(f"  Input Slice: {result_item.get('input', 'N/A')}")
                logger.warning(f"  Attempted Mask Output: {result_item.get('output', 'N/A')}")
                logger.warning(f"  Status: {result_item.get('status', 'N/A')}")
    logger.info("--- End of Summary ---")

if __name__ == "__main__":
    main() 