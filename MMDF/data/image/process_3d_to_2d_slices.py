import logging
import multiprocessing
from pathlib import Path
import os
import shlex
import subprocess
from functools import partial
from typing import List, Dict, Any, Optional, Tuple
import time

import nibabel as nib
import numpy as np
from tqdm import tqdm

DEBUG_ACTIVE = False 

try:
    import sys
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from ADtransformer_pytorch.data.ADNI.ADNIMERGE.processing.fsl import FSLProcessor
except ImportError as e:
    print(f"Error importing FSLProcessor: {e}")
    print("Please ensure that the script is placed correctly relative to the 'data' directory")
    print("or that the project root is in your PYTHONPATH.")
    print("Falling back to a basic FSL command runner if FSLProcessor cannot be imported.")
    class FSLProcessor:
        def __init__(self, fsl_output_type: str = "NIFTI_GZ", n_jobs: int = 1, verbose: bool = False):
            self.fsl_dir_wsl = os.environ.get("FSLDIR_WSL", "") 
            self.fsl_output_type = fsl_output_type
            self.verbose = verbose
            
            wsl_fslinfo_path = self._win_to_wsl_path(os.path.join(self.fsl_dir_wsl, 'bin', 'fslinfo'))
            try:
                subprocess.run(["wsl", "test", "-f", wsl_fslinfo_path], check=True, capture_output=True)
                if self.verbose:
                    print(f"Fallback FSLProcessor: fslinfo found at {wsl_fslinfo_path} in WSL.")
            except (subprocess.CalledProcessError, FileNotFoundError):
                 print(f"Warning: Fallback FSLProcessor could not verify fslinfo at {wsl_fslinfo_path} in WSL. FSL commands may fail.")


        def _win_to_wsl_path(self, win_path: str) -> str:
            import re
            if not win_path: return ""
            path = str(win_path).replace('\\\\', '/')
            path = path.replace('\\', '/')
            if re.match(r'^[A-Za-z]:', path):
                drive_letter = path[0].lower()
                path = f"/mnt/{drive_letter}{path[2:]}"
            return path

        def _run_wsl_command(self, core_command: str, check: bool = True, timeout: Optional[int] = 300) -> subprocess.CompletedProcess:
            quoted_fsl_dir = shlex.quote(self.fsl_dir_wsl)
            quoted_fsl_output_type = shlex.quote(self.fsl_output_type)
            full_command = f"export FSLDIR={quoted_fsl_dir}; export FSLOUTPUTTYPE={quoted_fsl_output_type}; . $FSLDIR/etc/fslconf/fsl.sh; {core_command}"
            
            try:
                result = subprocess.run(
                    ["wsl", "bash", "-c", full_command],
                    check=check, text=True, capture_output=True, timeout=timeout
                )
                if self.verbose:
                    print(f"WSL Command: {full_command}")
                    if result.stdout: print(f"WSL Stdout: {result.stdout}")
                    if result.stderr: print(f"WSL Stderr: {result.stderr}")
                if result.returncode != 0 and check:
                    raise subprocess.CalledProcessError(result.returncode, full_command, output=result.stdout, stderr=result.stderr)
                return result
            except subprocess.CalledProcessError as e:
                print(f"Command execution failed: {str(e)}\nCmd: {e.cmd}\nStderr: {e.stderr}")
                if check: raise
                return subprocess.CompletedProcess(e.cmd, e.returncode, e.stdout, e.stderr)
            except FileNotFoundError:
                print("Error: 'wsl' command not found. Is WSL installed and configured?")
                raise
            except subprocess.TimeoutExpired as e:
                print(f"Command timed out: {full_command}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                if check: raise
                return subprocess.CompletedProcess(e.cmd, -1, e.stdout, e.stderr)
            except Exception as e:
                print(f"Error running WSL command '{full_command}': {str(e)}")
                if check: raise
                return subprocess.CompletedProcess(full_command, -1, "", str(e))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("process_3d_to_2d")

def get_nifti_info(fsl_processor: FSLProcessor, nifti_file_path: str) -> Optional[Dict[str, Any]]:
    """Gets dimensions (dim1, dim2, dim3) and voxel sizes (pixdim1, pixdim2, pixdim3) of a NIfTI file using fslhd."""
    wsl_nifti_path = fsl_processor._win_to_wsl_path(nifti_file_path)
    fslhd_path = f"{fsl_processor.fsl_dir_wsl}/bin/fslhd" 
    command = f"{shlex.quote(fslhd_path)} {shlex.quote(wsl_nifti_path)}"
    
    info = {}
    try:
        result = fsl_processor._run_wsl_command(command, check=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts: continue
            key = parts[0]
            if key in ["dim1", "dim2", "dim3", "pixdim1", "pixdim2", "pixdim3"]:
                try:
                    value_str = parts[-1]
                    info[key] = float(value_str) if "pixdim" in key else int(value_str)
                except ValueError:
                    logger.warning(f"Could not parse value for {key} from line: '{line}' in {nifti_file_path}")
        
        required_dims = ["dim1", "dim2", "dim3"]
        if not all(d in info for d in required_dims):
            logger.error(f"Could not parse all required dimensions from fslhd output for {nifti_file_path}")
            logger.debug(f"fslhd output for {nifti_file_path}:\n{result.stdout}")
            logger.debug(f"Parsed info: {info}")
            return None
        return info
    except Exception as e:
        logger.error(f"Error getting NIfTI dimensions for {nifti_file_path} using fslhd: {e}")
        return None

def extract_coronal_slice_with_fslroi(
    fsl_processor: FSLProcessor,
    input_3d_nifti: str,
    output_slice_nifti: str,
    slice_index: int,
    original_dims: Dict[str, Any]
) -> bool:
    """Extracts a specific coronal slice (Y-axis) using fslroi."""
    wsl_input_path = fsl_processor._win_to_wsl_path(input_3d_nifti)
    wsl_output_path = fsl_processor._win_to_wsl_path(output_slice_nifti)
    fslroi_path = f"{fsl_processor.fsl_dir_wsl}/bin/fslroi"

    dim1 = int(original_dims['dim1'])
    dim3 = int(original_dims['dim3'])

    command = (f"{shlex.quote(fslroi_path)} {shlex.quote(wsl_input_path)} {shlex.quote(wsl_output_path)} "
               f"0 {dim1} {slice_index} 1 0 {dim3}")

    try:
        fsl_processor._run_wsl_command(command, check=True)
        if not os.path.exists(output_slice_nifti): 
            logger.error(f"Output slice {output_slice_nifti} not created by fslroi (checked on Windows path).")
            return False
        return True
    except Exception as e:
        logger.error(f"Error extracting coronal slice from {input_3d_nifti} with fslroi: {e}")
        return False

def resize_slice_with_flirt_applyisoxfm(
    fsl_processor: FSLProcessor,
    input_slice_nifti: str,
    output_resized_nifti: str,
    target_size: Tuple[int, int] = (72, 72)
) -> Tuple[bool, List[str]]:
    """Resizes a 2D NIfTI slice to target_size using FSL FLIRT with -applyisoxfm."""
    wsl_input_path = fsl_processor._win_to_wsl_path(input_slice_nifti)
    wsl_output_path = fsl_processor._win_to_wsl_path(output_resized_nifti)
    
    dummy_ref_dir = Path(output_resized_nifti).parent
    base_name_for_temp = Path(input_slice_nifti).stem.replace('_slice','').replace('_temp','')
    dummy_ref_name = f"dummy_{base_name_for_temp}_{target_size[0]}x{target_size[1]}ref_{int(time.time()*1000)}.nii.gz"
    dummy_ref_path = str(dummy_ref_dir / dummy_ref_name)
    wsl_dummy_ref_path = fsl_processor._win_to_wsl_path(dummy_ref_path)

    temp_files_to_clean_by_flirt = [dummy_ref_path] 

    input_slice_info = get_nifti_info(fsl_processor, input_slice_nifti)
    if not input_slice_info or not all(k in input_slice_info for k in ['dim1', 'pixdim1', 'dim3', 'pixdim3']):
        logger.error(f"Could not get complete NIfTI info for input slice {input_slice_nifti} to calculate dynamic scale.")
        return False, temp_files_to_clean_by_flirt

    fov_x_in = input_slice_info['dim1'] * input_slice_info['pixdim1']
    fov_z_in = input_slice_info['dim3'] * input_slice_info['pixdim3']
    max_fov_in = max(fov_x_in, fov_z_in)
    
    if target_size[0] <= 0:
        logger.error(f"Target size dimension ({target_size[0]}) is invalid for dynamic scale calculation.")
        return False, temp_files_to_clean_by_flirt
        
    dynamic_output_voxel_scale = max_fov_in / target_size[0]
    dynamic_output_voxel_scale = max(0.1, dynamic_output_voxel_scale) 
    logger.info(f"Calculated dynamic output voxel scale for FLIRT: {dynamic_output_voxel_scale:.4f} mm (based on max_input_fov={max_fov_in:.2f}mm / target_dim={target_size[0]})")

    if not os.path.exists(dummy_ref_path):
        fslcreatehd_path = f"{fsl_processor.fsl_dir_wsl}/bin/fslcreatehd"
        create_ref_cmd = (
            f"{shlex.quote(fslcreatehd_path)} "
            f"{target_size[0]} 1 {target_size[1]} 1 "
            f"{dynamic_output_voxel_scale:.6f} {dynamic_output_voxel_scale:.6f} {dynamic_output_voxel_scale:.6f} "
            f"1.0 0 0 0 16 "
            f"{shlex.quote(wsl_dummy_ref_path)}"
        )
        try:
            logger.debug(f"Creating dummy reference for FLIRT: {dummy_ref_path}")
            fsl_processor._run_wsl_command(create_ref_cmd, check=True)
            if not os.path.exists(dummy_ref_path):
                logger.error(f"Failed to create dummy reference image: {dummy_ref_path} (Windows path check).")
                return False, temp_files_to_clean_by_flirt
        except Exception as e:
            logger.error(f"Error creating dummy reference for FLIRT using fslcreatehd: {e}")
            return False, temp_files_to_clean_by_flirt

    flirt_path = f"{fsl_processor.fsl_dir_wsl}/bin/flirt"
    resize_cmd = (
        f"{shlex.quote(flirt_path)} "
        f"-in {shlex.quote(wsl_input_path)} "
        f"-ref {shlex.quote(wsl_dummy_ref_path)} "
        f"-out {shlex.quote(wsl_output_path)} "
        f"-applyisoxfm {dynamic_output_voxel_scale:.6f} "
        f"-interp sinc"
    )
    try:
        logger.debug(f"Executing FLIRT command for {Path(input_slice_nifti).name}: {resize_cmd}")
        fsl_processor._run_wsl_command(resize_cmd, check=True)
        
        if not os.path.exists(output_resized_nifti):
            logger.error(f"Output resized slice {output_resized_nifti} not created by flirt (Windows path check).")
            return False, temp_files_to_clean_by_flirt
                
        try:
            output_img_nib = nib.load(output_resized_nifti)
            output_data_nib = output_img_nib.get_fdata()
            if np.all(output_data_nib == 0):
                logger.warning(f"  WARNING: OUTPUT Slice ({Path(output_resized_nifti).name}) data is ALL ZEROS according to nibabel.")
            elif (np.max(output_data_nib) - np.min(output_data_nib)) < 1e-9:
                logger.warning(f"  WARNING: OUTPUT Slice ({Path(output_resized_nifti).name}) data is effectively flat (min~max: {np.min(output_data_nib):.4f}~{np.max(output_data_nib):.4f}). Might appear black.")
        except Exception as e_nib:
            logger.warning(f"  OUTPUT Slice ({Path(output_resized_nifti).name}) nibabel: Failed to load/read data - {e_nib}")
                    
        return True, temp_files_to_clean_by_flirt
    except Exception as e:
        logger.error(f"Error resizing slice {input_slice_nifti} with FLIRT: {e}")
        return False, temp_files_to_clean_by_flirt

def crop_nifti_slice_to_bbox(
    fsl_processor: FSLProcessor,
    input_slice_nifti: str,
    output_cropped_slice_nifti: str,
    threshold_factor: float = 0.05
) -> Tuple[bool, Optional[str]]:
    """Crops a 2D NIfTI slice to a bounding box determined by an intensity threshold."""
    try:
        logger.debug(f"Cropping slice {Path(input_slice_nifti).name} to bounding box.")
        img = nib.load(input_slice_nifti)
        data = img.get_fdata()
        original_affine = img.affine.copy()
        original_header = img.header.copy()

        if data.ndim > 2:
            squeezable_dims = [i for i, size in enumerate(data.shape) if size == 1]
            if len(squeezable_dims) > 0:
                data = np.squeeze(data, axis=tuple(squeezable_dims))
            else:
                logger.error(f"Data from {input_slice_nifti} has shape {data.shape} with no singleton dims to squeeze to 2D.")
                return False, None
        
        if data.ndim != 2:
            logger.error(f"Data from {input_slice_nifti} is not 2D after attempting squeeze; shape: {data.shape}. Cannot crop.")
            return False, None

        if np.all(data == 0) or np.max(data) == 0:
            logger.warning(f"Input slice {Path(input_slice_nifti).name} is all zeros or max is zero. Skipping crop, using original.")
            try:
                import shutil
                shutil.copy(input_slice_nifti, output_cropped_slice_nifti)
                logger.debug(f"Copied original slice to {Path(output_cropped_slice_nifti).name} as it was empty/zero.")
                return True, output_cropped_slice_nifti
            except Exception as e_copy:
                logger.error(f"Failed to copy empty/zero slice {input_slice_nifti} to {output_cropped_slice_nifti}: {e_copy}")
                return False, None

        threshold = threshold_factor * np.max(data)
        mask = data > threshold
        
        if not np.any(mask):
            logger.warning(f"No data above threshold {threshold:.2f} found in {Path(input_slice_nifti).name}. Crop might be empty or use original.")
            min_intensity_present = np.min(data[data > 0]) if np.any(data > 0) else 0
            if min_intensity_present > 0:
                threshold = min_intensity_present
                mask = data > threshold
                if not np.any(mask):
                    logger.warning(f"Still no data above fallback threshold for {Path(input_slice_nifti).name}. Copying original.")
                    shutil.copy(input_slice_nifti, output_cropped_slice_nifti)
                    return True, output_cropped_slice_nifti
            else:
                shutil.copy(input_slice_nifti, output_cropped_slice_nifti)
                return True, output_cropped_slice_nifti

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped_data_2d = data[rmin:rmax+1, cmin:cmax+1]
        
        cropped_data_3d = np.expand_dims(cropped_data_2d, axis=1)

        new_affine = original_affine.copy()
        translation_offset_voxels = np.array([rmin, 0, cmin])
        
        world_space_offset = original_affine[:3, :3] @ translation_offset_voxels
        new_affine[:3, 3] = original_affine[:3, 3] + world_space_offset

        cropped_img = nib.Nifti1Image(cropped_data_3d.astype(np.float32), new_affine, header=original_header)
        cropped_img.header.set_data_shape(cropped_data_3d.shape)
        cropped_img.header.set_data_dtype(np.float32)

        nib.save(cropped_img, output_cropped_slice_nifti)
        logger.debug(f"Saved cropped slice: {Path(output_cropped_slice_nifti).name} with shape {cropped_data_3d.shape}")
        
        cropped_info_fslhd = get_nifti_info(fsl_processor, output_cropped_slice_nifti)
        logger.info(f"  CROPPED Slice ({Path(output_cropped_slice_nifti).name}) fslhd: {cropped_info_fslhd}")
        logger.info(f"  CROPPED Slice ({Path(output_cropped_slice_nifti).name}) nibabel: min={np.min(cropped_data_3d):.4f}, max={np.max(cropped_data_3d):.4f}, mean={np.mean(cropped_data_3d):.4f}")
        return True, output_cropped_slice_nifti

    except Exception as e:
        logger.error(f"Error cropping slice {input_slice_nifti}: {e}", exc_info=True)
        return False, None

def convert_to_3channel_and_save(
    input_2d_nifti: str,
    output_3channel_nifti: str,
    target_dims: Tuple[int, int] = (72, 72)
) -> bool:
    """Loads a 2D NIfTI, normalizes, converts to 3-channel, and saves as NIfTI."""
    try:
        logger.debug(f"Converting {Path(input_2d_nifti).name} to 3-channel NIfTI.")
        img = nib.load(input_2d_nifti)
        data = img.get_fdata()

        if data.ndim == 3:
            data = np.squeeze(data)
        
        if data.ndim != 2:
            logger.error(f"Data from {input_2d_nifti} is not 2D after squeeze; shape: {data.shape}. Cannot convert to 3-channel.")
            return False

        if data.shape[0] != target_dims[0] or data.shape[1] != target_dims[1]:
            logger.warning(f"FLIRT output slice {Path(input_2d_nifti).name} has shape {data.shape}, but target is {target_dims}. Will crop/pad.")
            corrected_data = np.zeros(target_dims, dtype=data.dtype)
            src_rows, src_cols = data.shape
            trg_rows, trg_cols = target_dims
            copy_rows = min(src_rows, trg_rows)
            copy_cols = min(src_cols, trg_cols)
            corrected_data[:copy_rows, :copy_cols] = data[:copy_rows, :copy_cols]
            data = corrected_data
            
        min_val = np.min(data)
        max_val = np.max(data)
        if (max_val - min_val) > 1e-6:
            normalized_data = (data - min_val) / (max_val - min_val)
        else:
            normalized_data = np.zeros_like(data)

        data_3channel = np.stack([normalized_data, normalized_data, normalized_data], axis=-1)
        
        new_affine = img.affine.copy() 
        
        new_img = nib.Nifti1Image(data_3channel.astype(np.float32), new_affine, header=img.header)
        
        new_img.header.set_data_shape(data_3channel.shape)
        new_img.header.set_data_dtype(np.float32)
        
        new_img.header['intent_code'] = 1007 
        new_img.header['intent_name'] = b'RGB-COLOR'

        nib.save(new_img, output_3channel_nifti)
        logger.debug(f"Saved 3-channel NIfTI: {Path(output_3channel_nifti).name}")
        return True
    except Exception as e:
        logger.error(f"Error converting {input_2d_nifti} to 3-channel and saving: {e}", exc_info=True)
        return False

def process_single_scan(
    args_tuple: Tuple[str, str],
    fsl_processor: FSLProcessor,
    target_size_2d: Tuple[int, int]
) -> Dict[str, str]:
    """Processes a single 3D NIfTI scan to a 2D 3-channel slice."""
    input_file_path, output_dir = args_tuple
    status = {'input': input_file_path, 'output': '', 'status': 'failed_init'}
    
    file_stem_for_temps = Path(input_file_path).stem.replace('_3d', '')
    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000) % 100000}"
    
    temp_slice_file = str(Path(output_dir) / f"{file_stem_for_temps}_temp_slice_{unique_suffix}.nii.gz")
    temp_cropped_slice_file = str(Path(output_dir) / f"{file_stem_for_temps}_temp_cropped_slice_{unique_suffix}.nii.gz")
    temp_resized_slice_file = str(Path(output_dir) / f"{file_stem_for_temps}_temp_resized_{unique_suffix}.nii.gz")
    
    output_file_name = f"{file_stem_for_temps}_coronal_{target_size_2d[0]}x{target_size_2d[1]}x3.nii.gz"
    final_output_path = str(Path(output_dir) / output_file_name)
    status['output'] = final_output_path

    created_temp_files_this_run = [] 

    try:
        logger.info(f"Starting processing for: {Path(input_file_path).name}")
        if os.path.exists(final_output_path):
            logger.info(f"Output file {final_output_path} already exists. Skipping.")
            status['status'] = 'skipped_exists'
            return status

        original_info = get_nifti_info(fsl_processor, input_file_path)
        if not original_info or 'dim2' not in original_info or 'dim1' not in original_info or 'dim3' not in original_info :
            status['status'] = 'failed_get_dims'
            logger.error(f"Critical dimensions missing from NIfTI info for {input_file_path}. Parsed: {original_info}")
            return status
        
        middle_coronal_index = int(original_info['dim2'] // 2) 
        logger.debug(f"Input: {Path(input_file_path).name}, Info: {original_info}, Middle Coronal (dim2) Index: {middle_coronal_index}")

        if not extract_coronal_slice_with_fslroi(fsl_processor, input_file_path, temp_slice_file, middle_coronal_index, original_info):
            status['status'] = 'failed_extract_slice'
            return status
        created_temp_files_this_run.append(temp_slice_file)
        logger.debug(f"Extracted coronal slice to {Path(temp_slice_file).name}")

        crop_success, path_to_use_for_resize = crop_nifti_slice_to_bbox(
            fsl_processor, temp_slice_file, temp_cropped_slice_file
        )
        if not crop_success:
            status['status'] = 'failed_crop_slice'
            return status
        
        if path_to_use_for_resize and os.path.exists(path_to_use_for_resize):
             created_temp_files_this_run.append(path_to_use_for_resize)
             logger.debug(f"Using {Path(path_to_use_for_resize).name} for subsequent resize step.")
        else:
            logger.error(f"Cropping reported success but output path {path_to_use_for_resize} is invalid. Critical error.")
            status['status'] = 'failed_crop_slice_invalid_path'
            return status

        resize_success, flirt_created_temps = resize_slice_with_flirt_applyisoxfm(
            fsl_processor, path_to_use_for_resize, temp_resized_slice_file, target_size_2d
        )
        created_temp_files_this_run.extend(flirt_created_temps)
        
        if not resize_success:
            status['status'] = 'failed_resize_slice'
            return status
        if os.path.exists(temp_resized_slice_file):
             created_temp_files_this_run.append(temp_resized_slice_file)
        logger.debug(f"Resized slice to {Path(temp_resized_slice_file).name}")

        if not convert_to_3channel_and_save(temp_resized_slice_file, final_output_path, target_size_2d):
            status['status'] = 'failed_convert_3channel'
            return status
        
        logger.info(f"SUCCESS: {Path(input_file_path).name} -> {Path(final_output_path).name}")
        status['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Unhandled exception during processing of {input_file_path}: {e}", exc_info=True)
        status['status'] = 'failed_exception'
    finally:
        if DEBUG_ACTIVE:
            logger.info(f"DEBUG_ACTIVE: Keeping intermediate files for {Path(input_file_path).name}:")
            for temp_file_path in created_temp_files_this_run:
                if os.path.exists(temp_file_path):
                    logger.info(f"  - Kept: {Path(temp_file_path).name} at {temp_file_path}")
                else:
                    logger.debug(f"  - Info: Temp file {Path(temp_file_path).name} was marked for cleanup but not found at time of cleanup.")
        else:
            logger.debug(f"Cleaning up temporary files for {Path(input_file_path).name}:")
            for temp_file_path in created_temp_files_this_run:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"  - Cleaned up: {Path(temp_file_path).name}")
                    except OSError as e_clean:
                        logger.warning(f"  - Could not clean up temporary file {Path(temp_file_path).name}: {e_clean}")
                else:
                    logger.debug(f"  - Info: Temp file {Path(temp_file_path).name} was marked for cleanup but not found at time of cleanup.")
    return status

def find_input_files(input_dirs: List[str]) -> List[Path]:
    """Finds all files matching '*_3d.nii.gz' in the provided list of input directories."""
    all_files: List[Path] = []
    for directory_str in input_dirs:
        input_path = Path(directory_str)
        if not input_path.is_dir():
            logger.warning(f"Provided input path {directory_str} is not a directory or does not exist. Skipping.")
            continue
        
        found_files = list(input_path.glob("*_3d.nii.gz"))
        if found_files:
            logger.info(f"Found {len(found_files)} '*_3d.nii.gz' files in {directory_str}.")
            all_files.extend(found_files)
        else:
            logger.info(f"No '*_3d.nii.gz' files found in {directory_str}.")
            
    return all_files

def main():    
    current_script_path = Path(__file__).resolve()
    calculated_project_root = current_script_path.parents[3] 

    hardcoded_input_dirs_relative = [
        "data/ADNI/ADNIMERGE/processed_images/T1",
        "data/ADNI/ADNIMERGE/processed_images/FLAIR",
        "data/ADNI/ADNIMERGE/processed_images/MPRAGE"
    ]
    hardcoded_output_parent_dir_relative = "MMDF/data/image/processed_coronal_slices"
    
    script_input_dirs = [str(calculated_project_root / rel_path) for rel_path in hardcoded_input_dirs_relative]
    script_output_parent_dir = str(calculated_project_root / hardcoded_output_parent_dir_relative)
    script_num_workers = None
    script_target_size = (144, 144)
    script_log_level = "INFO"

    logger.setLevel(getattr(logging, script_log_level.upper()))

    logger.info(f"SCRIPT HARDCODED CONFIGURATION:")
    logger.info(f"  Input Directories: {script_input_dirs}")
    logger.info(f"  Output Parent Directory: {script_output_parent_dir}")
    logger.info(f"  Num Workers: {'Default (CPU count - 1 or 1)' if script_num_workers is None else script_num_workers}")
    logger.info(f"  Target Size: {script_target_size}")
    logger.info(f"  Log Level: {script_log_level}")
    if DEBUG_ACTIVE:
        logger.warning("*****************************************************************")
        logger.warning("*** DEBUG_ACTIVE IS ENABLED:                                ***")
        logger.warning("*** - Will process only THE FIRST input scan found.         ***")
        logger.warning("*** - Intermediate files WILL BE KEPT in output directory.  ***")
        logger.warning("*** SET DEBUG_ACTIVE TO FALSE FOR NORMAL FULL PROCESSING.     ***")
        logger.warning("*****************************************************************")

    try:
        fsl_processor = FSLProcessor(verbose=(script_log_level.upper() == "DEBUG"))
        logger.info(f"Successfully initialized FSLProcessor. WSL FSLDIR appears to be: {fsl_processor.fsl_dir_wsl}")
    except NameError: 
        logger.critical("FSLProcessor class definition not found. This indicates a problem with the import or the fallback mechanism. Exiting.")
        return
    except Exception as e:
        logger.critical(f"Failed to initialize FSLProcessor: {e}. This might be due to WSL/FSL setup issues. Exiting.", exc_info=True)
        return

    input_files_to_process = find_input_files(script_input_dirs)
    if not input_files_to_process:
        logger.info("No input files matching '*_3d.nii.gz' found in the specified directories. Nothing to do. Exiting.")
        return

    Path(script_output_parent_dir).mkdir(parents=True, exist_ok=True)
    
    tasks_for_processing: List[Tuple[str, str]] = []
    for file_path_obj in input_files_to_process:
        input_parent_folder_name = file_path_obj.parent.name
        output_dir_for_this_file = Path(script_output_parent_dir) / input_parent_folder_name
        output_dir_for_this_file.mkdir(parents=True, exist_ok=True)
        tasks_for_processing.append((str(file_path_obj), str(output_dir_for_this_file)))

    if not tasks_for_processing:
        logger.error("No tasks could be prepared for processing. Exiting.")
        return

    if DEBUG_ACTIVE and tasks_for_processing:
        logger.info(f"DEBUG_ACTIVE: Reducing task list from {len(tasks_for_processing)} to 1.")
        tasks_for_processing = [tasks_for_processing[0]]
        
    logger.info(f"Prepared {len(tasks_for_processing)} file(s) for processing.")
    
    num_workers_resolved = script_num_workers
    if num_workers_resolved is None:
        cpu_count = os.cpu_count()
        num_workers_resolved = max(1, cpu_count - 1) if cpu_count and cpu_count > 1 else 1
    num_workers_resolved = max(1, num_workers_resolved)
    
    logger.info(f"Using {num_workers_resolved} worker process(es). Target 2D slice size: {script_target_size[0]}x{script_target_size[1]}.")
    
    process_func_with_fixed_args = partial(
        process_single_scan,
        fsl_processor=fsl_processor,
        target_size_2d=script_target_size
    )
    
    all_results: List[Dict[str, str]] = []
    if num_workers_resolved > 1 and len(tasks_for_processing) > 1:
        logger.info(f"Starting parallel processing with {num_workers_resolved} workers.")
        with multiprocessing.Pool(processes=num_workers_resolved) as pool:
            all_results = list(tqdm(
                pool.imap_unordered(process_func_with_fixed_args, tasks_for_processing),
                total=len(tasks_for_processing),
                desc="Processing Scans"
            ))
    else: 
        logger.info("Running in single-process mode.")
        for task_args_tuple in tqdm(tasks_for_processing, desc="Processing Scans"):
            all_results.append(process_func_with_fixed_args(task_args_tuple))

    success_count = sum(1 for r in all_results if r.get('status') == 'success')
    skipped_count = sum(1 for r in all_results if r.get('status') == 'skipped_exists')
    failed_count = len(all_results) - success_count - skipped_count
    
    logger.info("--- Processing Summary ---")
    logger.info(f"Total files considered for processing: {len(tasks_for_processing)}")
    logger.info(f"Successfully processed and saved: {success_count}")
    logger.info(f"Skipped (output already existed): {skipped_count}")
    logger.info(f"Failed to process: {failed_count}")

    if failed_count > 0:
        logger.warning("Details for failed files:")
        for result_item in all_results:
            if result_item.get('status') not in ['success', 'skipped_exists']:
                logger.warning(f"  Input: {result_item.get('input', 'N/A')}")
                logger.warning(f"  Attempted Output: {result_item.get('output', 'N/A')}")
                logger.warning(f"  Status: {result_item.get('status', 'N/A')}")
    logger.info("--- End of Summary ---")

if __name__ == "__main__":
    main() 