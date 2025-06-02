import os
import subprocess
import logging
import shlex
import multiprocessing
from enum import Enum
import re
from typing import Optional
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger("FSL")

class ScanType(Enum):
    MPRAGE = "MPRAGE"
    MT1_GRADWARP_N3M = "MT1_GradWarp_N3m"
    FLAIR = "FLAIR"
    UNKNOWN = "Unknown"

class FSLProcessor:
    
    def __init__(
        self, 
        fsl_output_type: str = "NIFTI_GZ",
        n_jobs: int = None,
        verbose: bool = False
    ):
        self.fsl_dir_wsl = ""
        self.n4_path_wsl = ""
        self.fsl_output_type = fsl_output_type
        self.n_jobs = n_jobs or max(1, multiprocessing.cpu_count() - 1)
        self.verbose = verbose
        
        self._validate_fsl_installation()
        self._validate_n4_installation()
        
        if verbose:
            logger.info(f"FSL Processor initialized with WSL FSL directory: {self.fsl_dir_wsl}")
            logger.info(f"Using N4BiasFieldCorrection from: {self.n4_path_wsl}")

    def _validate_fsl_installation(self):
        try:
            find_cmd = f"command -v {shlex.quote(self.fsl_dir_wsl + '/bin/fslinfo')}"
            result_find = self._run_wsl_command(find_cmd, check=False)

            if result_find.returncode != 0:
                logger.warning(f"Could not find FSL executable 'fslinfo' at path: {self.fsl_dir_wsl}/bin/fslinfo")
                raise RuntimeError(f"FSL validation failed: fslinfo not found at expected path")
            else:
                fslinfo_path = result_find.stdout.strip()
                if self.verbose:
                    logger.info(f"Found fslinfo executable at: {fslinfo_path}")
                check_exec_cmd = f"[ -x {shlex.quote(fslinfo_path)} ]"
                result_check = self._run_wsl_command(check_exec_cmd, check=False)
                if result_check.returncode != 0:
                     logger.warning(f"Found fslinfo but it does not appear to be executable within WSL.")
                     raise RuntimeError(f"FSL validation failed: fslinfo found but not executable.")

        except Exception as e:
            logger.error(f"Error during FSL installation validation: {str(e)}")
            raise RuntimeError(f"Failed to validate FSL installation: {str(e)}")

    def _validate_n4_installation(self):
        try:
            check_exist_cmd = f"[ -x {shlex.quote(self.n4_path_wsl)} ]"
            exist_result = self._run_wsl_command(check_exist_cmd, check=False)
            
            if exist_result.returncode != 0:
                find_cmd = f"which N4BiasFieldCorrection"
                result_find = self._run_wsl_command(find_cmd, check=False)
                
                if result_find.returncode != 0:
                    logger.warning(f"N4BiasFieldCorrection not found at {self.n4_path_wsl} or in PATH")
                    raise RuntimeError(f"N4BiasFieldCorrection validation failed: command not found")
                else:
                    self.n4_path_wsl = result_find.stdout.strip()
                    if self.verbose:
                        logger.info(f"Found N4BiasFieldCorrection at: {self.n4_path_wsl}")
                
        except Exception as e:
            logger.error(f"Error during N4BiasFieldCorrection validation: {str(e)}")
            raise RuntimeError(f"Failed to validate N4BiasFieldCorrection: {str(e)}")

    def _win_to_wsl_path(self, win_path: str) -> str:
        if not win_path:
            return ""
            
        path = win_path.replace('\\', '/')
        
        if re.match(r'^[A-Za-z]:', path):
            drive_letter = path[0].lower()
            path = f"/mnt/{drive_letter}{path[2:]}"
            
        return path

    def _run_wsl_command(self, core_command: str, check: bool = True) -> subprocess.CompletedProcess:
        quoted_fsl_dir = shlex.quote(self.fsl_dir_wsl)
        quoted_fsl_output_type = shlex.quote(self.fsl_output_type)
        
        full_command = f"export FSLDIR={quoted_fsl_dir}; export FSLOUTPUTTYPE={quoted_fsl_output_type}; {core_command}"
                    
        try:
            result = subprocess.run(
                ["wsl", "bash", "-c", full_command],
                check=check,
                text=True,
                capture_output=True
            )
                            
            if result.returncode != 0 and check:
                logger.error(f"Command failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, full_command, output=result.stdout, stderr=result.stderr)
                    
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command execution failed: {str(e)}")
            logger.error(f"Error output: {e.stderr}")
            if check:
                raise
            return e
            
        except FileNotFoundError:
             logger.error("Error: 'wsl' command not found. Is WSL installed and configured?")
             raise
        except Exception as e:
            logger.error(f"Error running WSL command: {str(e)}")
            if check:
                raise
            return None

    def _detect_scan_type(self, input_path: str) -> ScanType:
        path_lower = input_path.lower()
        
        if "mprage" in path_lower or "mp-rage" in path_lower:
            return ScanType.MPRAGE
        elif "mt1" in path_lower and ("gradwarp" in path_lower or "n3m" in path_lower):
            return ScanType.MT1_GRADWARP_N3M
        elif "flair" in path_lower or "sagittal 3d flair" in path_lower:
            return ScanType.FLAIR
            
        if os.path.isdir(input_path):
            files = os.listdir(input_path)
            if any("mprage" in f.lower() or "mp-rage" in f.lower() for f in files):
                return ScanType.MPRAGE
            elif any("flair" in f.lower() for f in files):
                return ScanType.FLAIR
            else:
                logger.warning(f"Could not determine scan type for directory {input_path}, defaulting to MPRAGE")
                return ScanType.MPRAGE
        else:
            if "flair" in os.path.basename(input_path).lower():
                return ScanType.FLAIR
            elif "mprage" in os.path.basename(input_path).lower() or "mp-rage" in os.path.basename(input_path).lower():
                return ScanType.MPRAGE
            elif any(pattern in os.path.basename(input_path).lower() for pattern in ["mt1", "gradwarp", "n3m"]):
                return ScanType.MT1_GRADWARP_N3M
            
        logger.warning(f"Could not determine scan type for {input_path}")
        return ScanType.UNKNOWN

    def _validate_file(self, file_path: str, min_size_bytes: int = 200) -> bool:
        if not os.path.exists(file_path):
            logger.warning(f"File validation failed for {file_path}: File doesn't exist")
            return False
            
        try:
            file_size = os.path.getsize(file_path)
            if file_size < min_size_bytes:
                logger.warning(f"File validation failed for {file_path}: File too small ({file_size} bytes)")
                return False
        except Exception as e:
            logger.warning(f"Error checking file size for {file_path}: {e}")
            return False
            
        wsl_path = self._win_to_wsl_path(file_path)
        
        check_cmd = f"test -f {shlex.quote(wsl_path)} && [ $(stat -c%s {shlex.quote(wsl_path)}) -gt {min_size_bytes} ]"
        result = self._run_wsl_command(check_cmd, check=False)
        
        if result.returncode != 0:
            logger.warning(f"File validation failed for {file_path} in WSL: File doesn't exist or is too small")
            return False
            
        fslinfo_path = f"{self.fsl_dir_wsl}/bin/fslinfo"
        header_cmd = f"{shlex.quote(fslinfo_path)} {shlex.quote(wsl_path)} > /dev/null 2>&1"
        result = self._run_wsl_command(header_cmd, check=False)
        
        if result.returncode != 0:
            logger.warning(f"File validation failed for {file_path}: Could not read header with fslinfo")
            return False
            
        return True

    def _convert_dicom_to_nifti(self, input_dir: str, output_file: str) -> bool:        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output_basename = os.path.basename(output_file).replace('.nii.gz', '')
        output_dir = os.path.dirname(output_file)
        
        safe_output_basename = output_basename.replace(' ', '_')
        
        wsl_input_dir = self._win_to_wsl_path(input_dir)
        wsl_output_dir = self._win_to_wsl_path(output_dir)

        dcm2niix_wsl_path = f"{self.fsl_dir_wsl}/bin/dcm2niix"

        command = f"{shlex.quote(dcm2niix_wsl_path)} -z y -o {shlex.quote(wsl_output_dir)} -f {shlex.quote(safe_output_basename)} {shlex.quote(wsl_input_dir)}"
        try:
            self._run_wsl_command(command)
        except Exception as e:
            logger.error(f"Error converting DICOM to NIfTI: {e}")
            return False
        
        if not os.path.exists(output_file):
            potential_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
            
            if potential_files:
                expected_pattern = safe_output_basename
                matched_files = [f for f in potential_files if expected_pattern in f]
                
                if matched_files:
                    source_file = os.path.join(output_dir, matched_files[0])
                else:
                    source_file = os.path.join(output_dir, potential_files[0])
                
                os.rename(source_file, output_file)
            else:
                logger.error(f"DICOM conversion failed: no .nii.gz files found in {output_dir}")
                return False
        
        for f in os.listdir(output_dir):
            if f.endswith('.json') and (safe_output_basename in f or output_basename in f):
                try:
                    json_path = os.path.join(output_dir, f)
                    os.remove(json_path)
                except Exception as e:
                    logger.warning(f"Error removing JSON sidecar file {f}: {str(e)}")
        
        return True

    def _reorient_to_standard(self, input_file: str, output_file: str) -> bool:
        wsl_input = self._win_to_wsl_path(input_file)
        wsl_output = self._win_to_wsl_path(output_file)
        
        fslreorient2std_path = f"{self.fsl_dir_wsl}/bin/fslreorient2std"

        try:
            command = f"{shlex.quote(fslreorient2std_path)} {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(command)
        except Exception as e:
            logger.error(f"Error reorienting to standard: {e}")
            copy_cmd = f"cp {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(copy_cmd)
            
        if not self._validate_file(output_file):
            logger.warning(f"Reorientation failed for {input_file}, using input file")
            copy_cmd = f"cp {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(copy_cmd)
            if not self._validate_file(output_file):
                return False
        
        return True

    def _adjust_fov(self, input_file: str, output_file: str) -> bool:
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted")
            return False
            
        wsl_input = self._win_to_wsl_path(input_file)
        wsl_output = self._win_to_wsl_path(output_file)
        
        robustfov_path = f"{self.fsl_dir_wsl}/bin/robustfov"

        try:
            command = f"{shlex.quote(robustfov_path)} -i {shlex.quote(wsl_input)} -r {shlex.quote(wsl_output)}"
            self._run_wsl_command(command)
            
            if not self._validate_file(output_file):
                raise RuntimeError(f"robustfov created an invalid output file")
                
        except Exception as e:
            logger.warning(f"robustfov failed: {str(e)}. Copying input to output.")
            copy_cmd = f"cp {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(copy_cmd)
            
            if not self._validate_file(output_file):
                return False
        
        return True

    def _correct_bias_field(self, input_file: str, output_file: str) -> bool:        
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted")
            return False
            
        wsl_input = self._win_to_wsl_path(input_file)
        wsl_output = self._win_to_wsl_path(output_file)

        try:
            n4_command = f"{shlex.quote(self.n4_path_wsl)} -i {shlex.quote(wsl_input)} -o {shlex.quote(wsl_output)} -d 3 --convergence [20x20x20,0.001] --shrink-factor 4 --bspline-fitting [200]"
            
            self._run_wsl_command(n4_command)
            
            if not self._validate_file(output_file):
                raise RuntimeError(f"N4 bias correction created an invalid output file")
                
        except Exception as e:
            logger.warning(f"N4 bias correction failed: {str(e)}")
            copy_cmd = f"cp {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(copy_cmd)
            
            if not self._validate_file(output_file):
                return False
        
        return True

    def _extract_brain(self, input_file: str, output_file: str, fractional_intensity: float = 0.5) -> bool:        
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted")
            return False
            
        wsl_input = self._win_to_wsl_path(input_file)
        wsl_output = self._win_to_wsl_path(output_file)
        
        bet_executable_path = f"{self.fsl_dir_wsl}/bin/bet"

        try:
            command = f"{shlex.quote(bet_executable_path)} {shlex.quote(wsl_input)} {shlex.quote(wsl_output)} -f {fractional_intensity} -m"
            self._run_wsl_command(command)
            
            if not self._validate_file(output_file):
                raise RuntimeError(f"Brain extraction created an invalid output file")
                
        except Exception as e:
            logger.warning(f"Brain extraction failed: {str(e)}")
            copy_cmd = f"cp {shlex.quote(wsl_input)} {shlex.quote(wsl_output)}"
            self._run_wsl_command(copy_cmd)
            
            if not self._validate_file(output_file):
                return False
        
        return True

    def _extract_best_brain_slice(self, input_file: str, output_file: str, axis: int = 2) -> bool:
        input_file = os.path.normpath(input_file)
        output_file = os.path.normpath(output_file)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted")
            return False
        
        try:
            import nibabel as nib
            import numpy as np
            
            img = nib.load(input_file)
            data = img.get_fdata()
            
            threshold = np.max(data) * 0.01
            brain_mask = data > threshold
            
            if not np.any(brain_mask):
                logger.error(f"No brain tissue detected in {input_file} (all values below threshold)")
                return False
            
            shape = data.shape
            
            if axis == 0:
                middle_slice = shape[0] // 2
                total_slices = shape[0]
            elif axis == 1:
                middle_slice = shape[1] // 2
                total_slices = shape[1]
            else:
                middle_slice = shape[2] // 2
                total_slices = shape[2]
            
            range_size = max(int(total_slices * 0.4), 5)
            
            start_idx = max(0, middle_slice - range_size // 2)
            end_idx = min(total_slices, middle_slice + range_size // 2 + 1)
            
            slice_metrics = []
            for i in range(start_idx, end_idx):
                if axis == 0:
                    slice_data = data[i, :, :]
                    slice_mask = brain_mask[i, :, :]
                elif axis == 1:
                    slice_data = data[:, i, :]
                    slice_mask = brain_mask[:, i, :]
                else:
                    slice_data = data[:, :, i]
                    slice_mask = brain_mask[:, :, i]
                
                brain_voxels = np.sum(slice_mask)
                
                if brain_voxels > 0:
                    mean_intensity = np.mean(slice_data[slice_mask])
                else:
                    mean_intensity = 0
                
                combined_score = brain_voxels * mean_intensity
                
                slice_metrics.append({
                    'index': i,
                    'brain_voxels': brain_voxels,
                    'mean_intensity': mean_intensity,
                    'combined_score': combined_score
                })
            
            slice_metrics.sort(key=lambda x: x['combined_score'], reverse=True)
            
            if not slice_metrics or slice_metrics[0]['brain_voxels'] == 0:
                logger.warning(f"No slices with brain content found in the central region. Using middle slice as fallback.")
                best_slice_idx = middle_slice
            else:
                best_slice_idx = slice_metrics[0]['index']
            
            if axis == 0:
                slice_data = data[best_slice_idx, :, :]
            elif axis == 1:
                slice_data = data[:, best_slice_idx, :]
            else:
                slice_data = data[:, :, best_slice_idx]
                            
            if axis == 0:
                best_slice_data = data[best_slice_idx, :, :]
                best_slice_data = best_slice_data.reshape(1, best_slice_data.shape[0], best_slice_data.shape[1])
            elif axis == 1:
                best_slice_data = data[:, best_slice_idx, :]
                best_slice_data = best_slice_data.reshape(best_slice_data.shape[0], 1, best_slice_data.shape[1])
            else:
                best_slice_data = data[:, :, best_slice_idx]
                best_slice_data = best_slice_data.reshape(best_slice_data.shape[0], best_slice_data.shape[1], 1)
            
            new_img = nib.Nifti1Image(best_slice_data, img.affine, img.header)
            
            nib.save(new_img, output_file)
            
            if not os.path.exists(output_file):
                logger.error(f"Failed to create slice file at {output_file}")
                return False
                
            if not self._validate_file(output_file):
                logger.error(f"Created slice file is invalid: {output_file}")
                return False
            
            return True
            
        except ImportError as e:
            logger.error(f"Missing required Python package: {e}")
            raise RuntimeError(f"Nibabel and numpy are required for brain slice extraction")
        except Exception as e:
            logger.error(f"Error extracting best brain slice with nibabel: {e}")
            return False

    def _extract_middle_slice(self, input_file: str, output_file: str, axis: int = 2) -> bool:
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted")
            return False
        
        wsl_input = self._win_to_wsl_path(input_file)
        wsl_output = self._win_to_wsl_path(output_file)
        
        fslinfo_path = f"{self.fsl_dir_wsl}/bin/fslinfo"
        
        dim_command = f"{shlex.quote(fslinfo_path)} {shlex.quote(wsl_input)} | grep ^dim"
        dim_result = self._run_wsl_command(dim_command)
        
        dim_lines = dim_result.stdout.strip().split('\n')
        dims = []
        for line in dim_lines[:3]:
            dims.append(int(line.split()[1]))
        
        if len(dims) < 3:
            logger.error(f"Failed to get dimensions for {input_file}")
            return False
        
        middle_slice = dims[axis] // 2
        
        fslroi_path = f"{self.fsl_dir_wsl}/bin/fslroi"
        
        roi_params = []
        for i in range(3):
            if i == axis:
                roi_params.extend([middle_slice, 1])
            else:
                roi_params.extend([0, dims[i]])
                
        roi_command = f"{shlex.quote(fslroi_path)} {shlex.quote(wsl_input)} {shlex.quote(wsl_output)} {' '.join(map(str, roi_params))}"
        try:
            self._run_wsl_command(roi_command)
        except Exception as e:
            logger.error(f"Error extracting middle slice: {e}")
            return False
        
        if not self._validate_file(output_file):
            logger.error(f"Failed to extract valid middle slice from {input_file}")
            return False
            
        return True

    def _resize(self, input_file: str, output_file: str, size: int = 224) -> bool:
        if not self._validate_file(input_file):
            logger.error(f"Input file {input_file} is invalid or corrupted for resizing")
            return False

        input_file = os.path.normpath(input_file)
        output_file = os.path.normpath(output_file)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            import nibabel as nib
            import numpy as np
            from scipy import ndimage
            
            img = nib.load(input_file)
            data = img.get_fdata()
            
            current_shape = data.shape[:2]
            
            zoom_factors = [size / current_shape[0], size / current_shape[1], 1.0]
            
            resized_data = ndimage.zoom(data, zoom_factors, order=1)
            
            if resized_data.shape[0] != size or resized_data.shape[1] != size:
                logger.warning(f"Resized image has shape {resized_data.shape[:2]} instead of {size}x{size}. Adjusting...")
                temp = np.zeros((size, size, resized_data.shape[2]), dtype=resized_data.dtype)
                x_min = min(size, resized_data.shape[0])
                y_min = min(size, resized_data.shape[1])
                temp[:x_min, :y_min, :] = resized_data[:x_min, :y_min, :]
                resized_data = temp
            
            pixel_size = np.mean(img.header.get_zooms()[:2])
            new_affine = np.eye(4)
            new_affine[0, 0] = pixel_size
            new_affine[1, 1] = pixel_size
            new_affine[2, 2] = img.header.get_zooms()[2]
            new_affine[:3, 3] = -np.array([size/2 * pixel_size, size/2 * pixel_size, 0])
            
            new_img = nib.Nifti1Image(resized_data, new_affine, img.header)
            
            nib.save(new_img, output_file)
            
            if not os.path.exists(output_file):
                raise RuntimeError(f"Output file was not created: {output_file}")
                
            if not self._validate_file(output_file, min_size_bytes=100):
                data = nib.load(output_file).get_fdata()
                if np.all(data == 0):
                    logger.error(f"Resized output {output_file} is all zeros")
                    return False
                
                logger.warning(f"Resized file validation failed but file exists")
                
            return True
                
        except ImportError as e:
            logger.error(f"Missing required package: {e}. Need nibabel, numpy, and scipy for direct resizing.")
            
            return self._resize_with_flirt(input_file, output_file, size)
            
        except Exception as e:
            logger.error(f"Error during direct resizing to {size}x{size}: {e}")
            
            logger.info(f"Attempting fallback to FSL FLIRT for resizing")
            return self._resize_with_flirt(input_file, output_file, size)
    
    def _resize_with_flirt(self, input_file: str, output_file: str, size: int = 224) -> bool:
        """Fall back method to resize a 2D image slice using FSL FLIRT."""
        try:
            wsl_input = self._win_to_wsl_path(input_file)
            wsl_output = self._win_to_wsl_path(output_file)
            
            standard_ref_dir = os.path.dirname(output_file)
            standard_ref_file = os.path.normpath(os.path.join(standard_ref_dir, f"standard_{size}x{size}x1_ref.nii.gz"))
            wsl_standard_ref = self._win_to_wsl_path(standard_ref_file)

            if not os.path.exists(standard_ref_file):
                fslcreatehd_path = f"{self.fsl_dir_wsl}/bin/fslcreatehd"
                create_ref_cmd = (
                    f"{shlex.quote(fslcreatehd_path)} "
                    f"{size} {size} 1 1 1.0 1.0 1.0 1.0 0 0 0 16 " 
                    f"{shlex.quote(wsl_standard_ref)}"
                )
                self._run_wsl_command(create_ref_cmd, check=True)
                
                if not self._validate_file(standard_ref_file):
                    raise RuntimeError(f"Failed to create reference file")

            flirt_path = f"{self.fsl_dir_wsl}/bin/flirt"
            resize_cmd = (
                f"{shlex.quote(flirt_path)} "
                f"-in {shlex.quote(wsl_input)} "
                f"-ref {shlex.quote(wsl_standard_ref)} "
                f"-out {shlex.quote(wsl_output)} "
                f"-applyxfm -usesqform "
                f"-interp trilinear"
            )
            self._run_wsl_command(resize_cmd, check=True)
            
            return os.path.exists(output_file)
            
        except Exception as e:
            logger.error(f"FSL FLIRT resize fallback also failed: {e}")
            return False

    def _convert_3d_to_2d_image(self, input_file: str, output_file: str, axis: int = 2) -> bool:
        """Convert a 3D volume to a 2D image by extracting the slice with the most brain matter."""
        input_file = os.path.normpath(input_file)
        output_file = os.path.normpath(output_file)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        temp_dir = tempfile.gettempdir()
        temp_basename = os.path.basename(input_file).replace('.nii.gz', '')
        temp_slice_file = os.path.normpath(os.path.join(temp_dir, f"{temp_basename}_slice_temp_{os.getpid()}.nii.gz"))
        
        os.makedirs(os.path.dirname(temp_slice_file), exist_ok=True)
        
        if not self._extract_best_brain_slice(input_file, temp_slice_file, axis):
            logger.error(f"Failed to extract best brain slice from {input_file}")
            return False
        
        if not os.path.exists(temp_slice_file):
            logger.error(f"Temporary slice file was not created at {temp_slice_file}")
            return False
            
        result = self._resize(temp_slice_file, output_file)
        
        try:
            if os.path.exists(temp_slice_file):
                os.remove(temp_slice_file)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_slice_file}: {e}")
        
        return result

    def preprocess(
        self,
        input_path: str,
        output_file: str,
        output_3d_file: Optional[str] = None,
        scan_type: Optional[ScanType] = None,
        slice_axis: int = 2,
    ) -> bool:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if scan_type is None:
            scan_type = self._detect_scan_type(input_path)
                    
        temp_dir = os.path.dirname(output_file)
        temp_base = os.path.join(temp_dir, f"temp_{os.path.basename(output_file).replace('.nii.gz', '')}")
        
        converted_file = f"{temp_base}_converted.nii.gz"
        reoriented_file = f"{temp_base}_reoriented.nii.gz"
        fov_file = f"{temp_base}_fov.nii.gz"
        bias_corrected_file = f"{temp_base}_bias.nii.gz"
        brain_file = f"{temp_base}_brain.nii.gz"
        
        try:
            if os.path.isdir(input_path):
                if not self._convert_dicom_to_nifti(input_path, converted_file):
                    logger.error(f"DICOM conversion failed for {input_path}")
                    return False
                current_file = converted_file
            else:
                current_file = input_path
                
            if not self._reorient_to_standard(current_file, reoriented_file):
                logger.error(f"Reorientation failed for {current_file}")
                return False
            current_file = reoriented_file
                
            if not self._adjust_fov(current_file, fov_file):
                logger.error(f"Field of view adjustment failed for {current_file}")
                return False
            current_file = fov_file
                
            if not self._correct_bias_field(current_file, bias_corrected_file):
                logger.error(f"Bias correction failed for {current_file}")
                return False
            current_file = bias_corrected_file
                
            if scan_type == ScanType.MPRAGE:
                bet_f = 0.6
            elif scan_type == ScanType.MT1_GRADWARP_N3M:
                bet_f = 0.55
            else:
                bet_f = 0.45
                
            if not self._extract_brain(current_file, brain_file, bet_f):
                logger.error(f"Brain extraction failed for {current_file}")
                return False
            current_file = brain_file
            
            if output_3d_file:
                os.makedirs(os.path.dirname(output_3d_file), exist_ok=True)
                if os.path.exists(current_file):
                    try:
                        wsl_brain = self._win_to_wsl_path(current_file)
                        wsl_3d_output = self._win_to_wsl_path(output_3d_file)
                        copy_cmd = f"cp {shlex.quote(wsl_brain)} {shlex.quote(wsl_3d_output)}"
                        self._run_wsl_command(copy_cmd)
                    except Exception as e:
                        logger.warning(f"Failed to save 3D brain model to {output_3d_file}: {e}")
                
            if not self._convert_3d_to_2d_image(current_file, output_file, slice_axis):
                logger.error(f"Conversion to 2D image failed for {current_file}")
                return False
                
            if self._validate_file(output_file):
                for temp_file in [converted_file, reoriented_file, fov_file, bias_corrected_file]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.warning(f"Failed to clean up {temp_file}: {e}")
                            
                return True
            else:
                logger.error(f"Final validation failed for {output_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            return False
