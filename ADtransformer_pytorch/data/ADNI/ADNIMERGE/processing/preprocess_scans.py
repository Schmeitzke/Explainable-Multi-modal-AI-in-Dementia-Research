import argparse
import yaml
import logging
import multiprocessing
from pathlib import Path
import os
from functools import partial
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from data.ADNI.ADNIMERGE.processing.fsl import FSLProcessor, ScanType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("preprocess")

def find_scan_path(base_input_dir: str, participant_id: str, scan_type_enum: ScanType) -> Optional[str]:
    """Find the path to the original scan data for a participant."""
    try:
        base_path = Path(base_input_dir)

        if scan_type_enum == ScanType.MPRAGE:
            scan_type_folder = "MPRAGE"
            scan_specific_subfolders = ["MPRAGE", "MP-RAGE"]
        elif scan_type_enum == ScanType.FLAIR:
            scan_type_folder = "FLAIR"
            scan_specific_subfolders = ["Sagittal_3D_FLAIR"]
        elif scan_type_enum == ScanType.MT1_GRADWARP_N3M:
            scan_type_folder = "T1"
            scan_specific_subfolders = ["MT1__GradWarp__N3m"]
        else:
            return None

        if scan_type_enum == ScanType.MT1_GRADWARP_N3M:
            for scan_specific_subfolder in scan_specific_subfolders:
                pattern = f"{scan_type_folder}/ADNI/{participant_id}/{scan_specific_subfolder}/*/*/*.nii*"
                potential_files = list(base_path.glob(pattern))
                if potential_files:
                    return str(sorted(potential_files)[0])
        
        for scan_specific_subfolder in scan_specific_subfolders:
            participant_scan_base = base_path / scan_type_folder / "ADNI" / participant_id / scan_specific_subfolder
            
            if not participant_scan_base.is_dir():
                continue
                
            date_folders = sorted([d for d in participant_scan_base.iterdir() if d.is_dir()])
            if not date_folders:
                continue
            
            scan_path_date = date_folders[0]

            i_folders = sorted([i for i in scan_path_date.iterdir() if i.is_dir() and i.name.startswith("I")])
            if not i_folders:
                continue
                
            i_folder_path = i_folders[0]
            
            if any(f.suffix.lower() in ['.dcm', ''] for f in i_folder_path.iterdir() if f.is_file()):
                return str(i_folder_path)

        return None

    except Exception as e:
        logger.error(f"Error finding scan for {participant_id}, {scan_type_enum.value}: {e}")
        return None

def process_participant(
    participant_id: str, 
    config: Dict[str, Any], 
    preprocessor: FSLProcessor,
    scan_types_to_process: List[ScanType],
    slice_axis: int
) -> Dict[str, str]:
    results = {}
    raw_base_dir = config['data']['raw_base_dir']
    found_any_scan = False
    
    for scan_type_enum in scan_types_to_process:
        if scan_type_enum == ScanType.MT1_GRADWARP_N3M:
            output_dir = config['data'].get('t1_dir')
        elif scan_type_enum == ScanType.FLAIR:
            output_dir = config['data'].get('flair_dir')
        elif scan_type_enum == ScanType.MPRAGE:
            output_dir = config['data'].get('mprage_dir')
        else:
            results[scan_type_enum.value] = 'config_error'
            continue

        if not output_dir:
            results[scan_type_enum.value] = 'config_error'
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"{participant_id}_{scan_type_enum.value}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            results[scan_type_enum.value] = 'skipped_exists'
            found_any_scan = True
            continue
            
        input_path = find_scan_path(raw_base_dir, participant_id, scan_type_enum)
        if not input_path:
            results[scan_type_enum.value] = 'not_found'
            continue
        
        found_any_scan = True
        
        try:
            output_3d_filename = f"{participant_id}_{scan_type_enum.value}_3d.nii.gz"
            output_3d_path = os.path.join(output_dir, output_3d_filename)
            
            success = preprocessor.preprocess(
                input_path=input_path,
                output_file=output_path,
                output_3d_file=output_3d_path,
                scan_type=scan_type_enum,
                slice_axis=slice_axis
            )
            
            if success and os.path.exists(output_path):
                results[scan_type_enum.value] = 'success'
            else:
                results[scan_type_enum.value] = 'error_processing'
                
        except Exception as e:
            logger.error(f"Error processing {participant_id} {scan_type_enum.value}: {e}")
            results[scan_type_enum.value] = 'error'
    
    if not found_any_scan:
        scan_types_str = ", ".join([st.value for st in scan_types_to_process])
        logger.warning(f"No scan data found for participant {participant_id} across any requested scan types: {scan_types_str}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Preprocess MRI scans and extract 2D slices.")
    parser.add_argument("--config", type=str, default="", 
                        help="Path to config file")
    parser.add_argument("--scan_types", nargs='+', 
                        choices=['mprage', 't1', 'flair', 'all'], 
                        default=['all'], 
                        help="Scan types to process")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--slice_axis", type=int, default=2, choices=[0, 1, 2],
                        help="Axis for slice extraction: 0=sagittal, 1=coronal, 2=axial (default)")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    config_path = Path(args.config)
    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return
    
    for dir_key in ['mprage_dir', 'flair_dir', 't1_dir']:
        if dir_key in config['data']:
            os.makedirs(config['data'][dir_key], exist_ok=True)
    
    scan_types_requested = args.scan_types
    scan_type_map = {
        "mprage": ScanType.MPRAGE,
        "t1": ScanType.MT1_GRADWARP_N3M,
        "flair": ScanType.FLAIR
    }
    
    if 'all' in scan_types_requested:
        scan_types_to_process = list(scan_type_map.values())
    else:
        scan_types_to_process = [scan_type_map[st] for st in scan_types_requested if st in scan_type_map]

    if not scan_types_to_process:
        logger.error("No valid scan types selected for processing.")
        return

    try:
        n_jobs = args.num_workers if args.num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        if n_jobs <= 0:
            n_jobs = 1
            
        preprocessor = FSLProcessor(
            n_jobs=1,
            verbose=(log_level == logging.DEBUG)
        )
    except Exception as e:
        logger.error(f"Failed to initialize FSL processor: {e}")
        return

    raw_base_dir = config['data']['raw_base_dir']
    participant_ids_set = set()

    scan_folder_mapping = {
        ScanType.MPRAGE: ["MPRAGE", "MP-RAGE"],
        ScanType.FLAIR: ["FLAIR"],
        ScanType.MT1_GRADWARP_N3M: ["T1"]
    }
    
    for scan_enum, folder_names in scan_folder_mapping.items():
        if scan_enum in scan_types_to_process:
            for scan_folder_name in folder_names:
                adni_path = Path(raw_base_dir) / scan_folder_name / "ADNI"
                if adni_path.is_dir():
                    try:
                        found_participants = {p.name for p in adni_path.iterdir() if p.is_dir()}
                        participant_ids_set.update(found_participants)
                        logger.info(f"Found {len(found_participants)} participants in {scan_folder_name} folder")
                    except OSError as e:
                        logger.warning(f"Could not list participants in {adni_path}: {e}")

    participant_ids = sorted(list(participant_ids_set))

    if not participant_ids:
        logger.error(f"No participant directories found.")
        return
        
    logger.info(f"Found {len(participant_ids)} total unique participants")

    process_func = partial(
        process_participant,
        config=config,
        preprocessor=preprocessor,
        scan_types_to_process=scan_types_to_process,
        slice_axis=args.slice_axis
    )

    with multiprocessing.Pool(processes=n_jobs) as pool:
        all_results = list(tqdm(
            pool.imap_unordered(process_func, participant_ids),
            total=len(participant_ids),
            desc="Processing Participants"
        ))

    results_by_scan_type = {}
    for scan_type in [st.value for st in scan_types_to_process]:
        results_by_scan_type[scan_type] = {
            'success': 0,
            'skipped_exists': 0,
            'error': 0,
            'not_found': 0,
            'other': 0
        }
    
    for result in all_results:
        for scan_type, status in result.items():
            if scan_type in results_by_scan_type:
                if status in results_by_scan_type[scan_type]:
                    results_by_scan_type[scan_type][status] += 1
                else:
                    results_by_scan_type[scan_type]['other'] += 1

    logger.info("Processing complete!")
    for scan_type, counts in results_by_scan_type.items():
        logger.info(f"{scan_type} results: "
                   f"{counts['success']} succeeded, "
                   f"{counts['skipped_exists']} skipped (already exist), "
                   f"{counts['not_found']} not found, "
                   f"{counts['error'] + counts['other']} failed")

if __name__ == "__main__":
    main()
