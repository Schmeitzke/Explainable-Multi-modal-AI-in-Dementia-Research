import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import logging
import nibabel as nib
from sklearn.utils.class_weight import compute_class_weight
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _create_ptid_to_image_path_map(image_base_dirs_dict, scan_types):
    """
    Creates a map from PTID to a dictionary of scan_type: image_path.
    """
    ptid_map = {}
    logging.info(f"Creating PTID to image path map for scan types: {scan_types}")
    
    logged_sample_ptid_count = 0
    max_sample_ptids_to_log = 5

    for scan_type in scan_types:
        image_dir = image_base_dirs_dict.get(scan_type)
        if not image_dir:
            logging.warning(f"No image directory specified for scan type: {scan_type}")
            continue
        if not os.path.isdir(image_dir):
            logging.warning(f"Image directory not found for scan type {scan_type}: {image_dir}")
            continue

        logging.info(f"Scanning directory: {image_dir} for {scan_type} images...")
        image_files = glob.glob(os.path.join(image_dir, "*.nii")) + \
                      glob.glob(os.path.join(image_dir, "*.nii.gz"))
        
        logging.info(f"Found {len(image_files)} potential image files for {scan_type}.")

        for image_file_path in image_files:
            try:
                basename = os.path.basename(image_file_path)
                filename_parts = basename.split('_')
                if len(filename_parts) >= 3 and filename_parts[0].isdigit() and \
                   filename_parts[1].isalpha() and filename_parts[2].isdigit():
                    ptid = '_'.join(filename_parts[:3])
                else:
                    ptid = filename_parts[0]

                if logged_sample_ptid_count < max_sample_ptids_to_log:
                    logging.info(f"Sample extracted PTID from filename {basename} for scan {scan_type}: '{ptid}'")
                    logged_sample_ptid_count +=1

                if ptid not in ptid_map:
                    ptid_map[ptid] = {}
                if scan_type not in ptid_map[ptid]:
                    ptid_map[ptid][scan_type] = image_file_path
                else:
                    logging.debug(f"Duplicate image for PTID {ptid} and scan_type {scan_type}: {image_file_path}. Keeping first one: {ptid_map[ptid][scan_type]}")

            except Exception as e:
                logging.warning(f"Error processing image filename {image_file_path}: {e}")
    
    logging.info(f"Created PTID map with {len(ptid_map)} entries.")
    return ptid_map

def get_first_available_scan(row_dict, scan_priority):
    if not isinstance(row_dict, dict):
        return None
    for scan_type in scan_priority:
        if scan_type in row_dict and row_dict[scan_type] and os.path.exists(row_dict[scan_type]):
            return row_dict[scan_type]
    return None

class ADNIDataset(Dataset):
    """
    ADNI dataset that loads images on-demand rather than storing them in memory.
    This significantly reduces memory usage during training.
    """
    def __init__(self, features, labels, image_paths=None, subject_ids=None, config=None, feature_columns=None):
        self.features = features
        self.labels = labels
        self.subject_ids = subject_ids if subject_ids is not None else []
        self.config = config if config is not None else {}
        self.feature_columns = feature_columns
        self.image_paths = image_paths if image_paths is not None else {}
        
        self.use_images = False
        self.image_shape = None
        self.expected_channels = 1
        
        if config and "image" in config:
            img_cfg = config["image"]
            self.use_images = img_cfg.get("use_images", False)
            
            if self.use_images:
                self.image_shape = tuple(img_cfg.get("shape", [144, 144, 144]))
        
        if self.use_images and self.image_shape:
            self.dummy_image = torch.zeros((1, *self.image_shape), dtype=torch.float32)
        else:
            self.dummy_image = None
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'non_image': self.features[idx],
            'label': self.labels[idx]
        }
        
        if self.use_images:
            image_path = self.image_paths[idx] 
            loaded_image_tensor = None

            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                try:
                    loaded_image_tensor = self.load_image(image_path)
                except Exception as e:
                    logging.debug(f"Error loading image at {image_path} for subject {self.subject_ids[idx] if self.subject_ids and idx < len(self.subject_ids) else 'Unknown'}: {e}")
            
            if loaded_image_tensor is None:
                item['image'] = torch.zeros((1, *self.image_shape), dtype=torch.float32) if self.image_shape else None
                if image_path:
                    logging.warning(f"Failed to load image for subject {self.subject_ids[idx] if self.subject_ids and idx < len(self.subject_ids) else 'Unknown'} from {image_path}. Using dummy image.")
            else:
                item['image'] = loaded_image_tensor
        else:
            item['image'] = None
            
        return item

    def load_image(self, image_path):
        """Load and preprocess a single image file."""
        if not image_path or not os.path.exists(image_path):
            return None
            
        try:
            nib_img = nib.load(image_path)
            img_data = nib_img.get_fdata(dtype=np.float32)
            
            if img_data.shape != self.image_shape:
                logging.warning(f"Image shape mismatch: {image_path} has shape {img_data.shape}, expected {self.image_shape}")
                return None
                
            img_tensor = torch.from_numpy(img_data).unsqueeze(0)
            return img_tensor
            
        except Exception as e:
            logging.debug(f"Error loading {image_path}: {e}")
            return None

def process_genotype(genotype):
    """
    Processes genotype information into APOE4 allele count.
    This function is taken from make_dataset.py to ensure consistent processing.
    """
    if pd.isna(genotype):
        return -4
        
    genotype_str = str(genotype).strip()
    
    if '/' in genotype_str:
        alleles = genotype_str.split('/')
        return alleles.count('4')
        
    elif '.' in genotype_str:
        try:
            val = float(genotype_str)
            if val == 0.0: return 0
            if val == 1.0: return 1
            if val == 2.0: return 2
        except ValueError:
            pass
            
    try:
        val = int(genotype_str)
        if val in [0, 1, 2]:
            return val
    except ValueError:
        pass
        
    logging.warning(f"Unrecognized genotype format '{genotype}'. Assigning missing value.")
    return -4

def _process_csv(csv_path, ptid_map, cfg, is_train_data, scaler_instance=None, current_actual_scan_types=None):
    """
    Processes a single CSV file: reads data, handles features, and maps images.
    Scales features if a scaler is provided (for test data), or fits and scales (for train data).
    """
    logging.info(f"Processing CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logging.info(f"Successfully loaded {csv_path}. Shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading CSV {csv_path}: {e}")
        return None, None, None, None, None, None

    subject_id_col = cfg.get('data', {}).get('subject_id_column', 'PTID')
    label_col = 'DIAGNOSIS'
    class_names = cfg.get('data', {}).get('class_names', ['CN', 'MCI', 'AD'])
    perform_binary_classification = cfg.get('data', {}).get('perform_binary_classification_cn_ad', False)

    if perform_binary_classification:
        logging.info(f"Performing binary classification (CN vs AD) for {csv_path}.")
        df = df[df[label_col].isin([0, 2])].copy()
        df.loc[df[label_col] == 2, label_col] = 1
        class_names = cfg.get('data', {}).get('binary_class_names', ['CN', 'AD'])
        cfg['data']['class_names'] = class_names
        cfg['classifier']['output_dim'] = 2
        logging.info(f"Filtered for CN/AD and remapped labels. New class_names: {class_names}. Output_dim set to 2.")
        logging.info(f"Shape of df after filtering for binary classification: {df.shape}")
        if df.empty:
            logging.error(f"DataFrame is empty after filtering for CN/AD in {csv_path}. Cannot proceed.")
            return None, None, None, None, None, None

    current_image_paths_list = []
    if cfg.get('image', {}).get('use_images', False) and ptid_map:
        df['image_paths_dict'] = df[subject_id_col].map(lambda ptid: ptid_map.get(str(ptid)))
        
        found_in_map_count = df['image_paths_dict'].notna().sum()
        logging.info(f"{found_in_map_count} out of {len(df)} PTIDs from CSV found in the generated image PTID map.")

        df['selected_scan_path'] = df['image_paths_dict'].apply(
            lambda paths_dict: get_first_available_scan(paths_dict, current_actual_scan_types)
        )

        initial_rows = len(df)
        df = df[df['selected_scan_path'].notna()].copy()
        rows_dropped = initial_rows - len(df)
        logging.info(f"Filtered by single scan availability (one of {current_actual_scan_types}): Dropped {rows_dropped} rows. {len(df)} rows remaining.")
        
        if len(df) > 0:
            logging.info(f"Sample selected scan paths: {df['selected_scan_path'].head().tolist()}")
        current_image_paths_list = df['selected_scan_path'].tolist()
    else:
        current_image_paths_list = [None for _ in range(len(df))]

    if len(df) == 0:
        logging.error(f"No rows remaining in {csv_path} after image filtering. Cannot proceed.")
        if is_train_data:
             return None, None, None, [], scaler_instance, None
        else:
             return None, None, None, [], None

    balance_classes = cfg.get('data', {}).get('balance_classes', False)
    if balance_classes:
        logging.info(f"Applying class balancing (undersampling) for {csv_path} AFTER image filtering.")
        if label_col not in df.columns:
            logging.error(f"Label column '{label_col}' not found in {csv_path} for balancing.")
        elif df.empty:
            logging.warning(f"DataFrame is empty for {csv_path} before balancing. Skipping balancing.")
        else:
            class_counts = df[label_col].value_counts()
            if class_counts.empty:
                logging.warning(f"No class counts to balance in {csv_path} (all values might be NaN or df is empty).")
            else:
                min_class_count = class_counts.min()
                logging.info(f"Class counts before balancing (post-image filter): {class_counts.to_dict()}")
                logging.info(f"Minority class count for balancing: {min_class_count}")

                balanced_df_list = []
                for class_val in df[label_col].unique():
                    class_subset = df[df[label_col] == class_val]
                    if len(class_subset) > min_class_count:
                        balanced_df_list.append(class_subset.sample(min_class_count, random_state=42))
                    else:
                        balanced_df_list.append(class_subset)
                
                if balanced_df_list:
                    df = pd.concat(balanced_df_list).sample(frac=1, random_state=42).reset_index(drop=True)
                    logging.info(f"Class counts after balancing: {df[label_col].value_counts().to_dict()}")
                    logging.info(f"Shape of df after balancing: {df.shape}")
                    if 'selected_scan_path' in df.columns and cfg.get('image', {}).get('use_images', False) :
                        current_image_paths_list = df['selected_scan_path'].tolist()
                    else:
                        current_image_paths_list = [None for _ in range(len(df))]
                else:
                    logging.warning(f"Could not perform balancing for {csv_path}, balanced_df_list is empty (e.g. df might have become empty).")
        if df.empty:
            logging.error(f"DataFrame is empty after attempted balancing for {csv_path}. Cannot proceed.")
            if is_train_data: return None, None, None, [], scaler_instance, None
            else: return None, None, None, [], None
    
    subject_ids = df[subject_id_col].values.tolist() if subject_id_col in df.columns and not df.empty else []

    excluded_cols_for_X = [subject_id_col, label_col, 'image_paths_dict', 'has_all_scans', 'selected_scan_path']
    if 'GENOTYPE_APOE4_count' in df.columns and 'GENOTYPE' in df.columns:
        excluded_cols_for_X.append('GENOTYPE')
        logging.info("Excluding 'GENOTYPE' column as 'GENOTYPE_APOE4_count' is available.")
    
    potential_feature_columns = [col for col in df.columns if col not in excluded_cols_for_X]
    
    final_feature_columns = []
    X_df = pd.DataFrame()

    for col in potential_feature_columns:
        if df[col].dtype == 'object':
            try:
                converted_col = pd.to_numeric(df[col], errors='coerce')
                if converted_col.isna().sum() > len(df) / 2:
                    logging.warning(f"Column '{col}' is object type and mostly non-numeric after coercion. Excluding from features.")
                else:
                    X_df[col] = converted_col
                    final_feature_columns.append(col)
                    logging.info(f"Column '{col}' (object) converted to numeric and included in features.")
            except ValueError:
                logging.warning(f"Column '{col}' is object type and could not be converted to numeric. Excluding from features.")
        else:
            X_df[col] = df[col]
            final_feature_columns.append(col)
    
    logging.info(f"Selected feature columns for X: {final_feature_columns}")
    
    if not final_feature_columns:
        logging.error(f"No feature columns selected from {csv_path}. Cannot proceed.")
        if is_train_data:
            return None, None, None, [], scaler_instance, None
        else:
            return None, None, None, [], None


    X = X_df[final_feature_columns].copy()
    
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        logging.info(f"Converted {label_col} to numeric in {csv_path}.")
    
    df.dropna(subset=[label_col], inplace=True)
    X = X.loc[df.index]
    subject_ids = df.loc[df.index, subject_id_col].values.tolist()
    if cfg.get('image', {}).get('use_images', False) and ptid_map:
        current_image_paths_list = df.loc[df.index, 'selected_scan_path'].tolist() if 'selected_scan_path' in df.columns else [None for _ in range(len(df))]
    else:
        current_image_paths_list = [None for _ in range(len(df))]


    y = df[label_col].values
    logging.info(f"Target variable '{label_col}' processed. Min label: {y.min()}, Max label: {y.max()}")

    X = X.fillna(-4.0).astype(np.float32)
    logging.info(f"Filled NaNs in X with -4.0 (if any). X shape: {X.shape}")

    X_to_tensor = X 

    if is_train_data:
        cfg['non_image']['num_features'] = X_to_tensor.shape[1]
        logging.info(f"Set config['non_image']['num_features'] to {X_to_tensor.shape[1]}")
        if not perform_binary_classification:
            unique_train_labels = np.unique(y)
            cfg['classifier']['output_dim'] = len(unique_train_labels)
            logging.info(f"Set config['classifier']['output_dim'] to {len(unique_train_labels)} based on training labels.")
        else:
            unique_train_labels = np.unique(y)
            if len(unique_train_labels) != 2 and not df.empty:
                logging.warning(f"Binary classification expected 2 unique labels, but found {len(unique_train_labels)}: {unique_train_labels} in y for {csv_path}. Check data and filtering.")
            cfg['classifier']['output_dim'] = 2


    X_tensor = torch.tensor(X_to_tensor.values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    if is_train_data:
        return X_tensor, y_tensor, current_image_paths_list, subject_ids, final_feature_columns, None 
    else:
        return X_tensor, y_tensor, current_image_paths_list, subject_ids, final_feature_columns


def prepare_adni_dataset(config_path, train_csv_path, test_csv_path):
    """
    Prepare ADNI dataset for training/evaluation using separate train/test CSVs
    and mapping images from directories.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_base_dirs_dict = {}
    scan_types_from_config = config.get('data', {}).get('scan_types', [])
    use_images = config.get('image', {}).get('use_images', False)

    if use_images:
        logging.info("Image usage enabled. Constructing image base directories dictionary.")
        for scan_type in scan_types_from_config:
            key = f"processed_{scan_type}_dir_adnimerge"
            if key in config['data']:
                image_base_dirs_dict[scan_type] = config['data'][key]
                logging.info(f"Registered image directory for {scan_type}: {config['data'][key]}")
            else:
                logging.warning(f"Directory key '{key}' not found in config['data'] for scan type '{scan_type}'.")
        
        if not image_base_dirs_dict:
            logging.warning("No image directories found in config. Disabling image use.")
            use_images = False
            config['image']['use_images'] = False
    else:
        logging.info("Image usage is disabled in config.")

    ptid_to_image_map = {}
    actual_scan_types_for_mapping = []
    if use_images:
        actual_scan_types_for_mapping = [st for st in scan_types_from_config if st in image_base_dirs_dict]
        if actual_scan_types_for_mapping:
            logging.info(f"Attempting to map images for scan types: {actual_scan_types_for_mapping}")

            ptid_to_image_map = _create_ptid_to_image_path_map(
                image_base_dirs_dict,
                actual_scan_types_for_mapping
            )
            if not ptid_to_image_map:
                logging.warning("PTID to image map is empty. Image loading might fail or be skipped.")
        else:
            logging.warning("No scan types with valid directory configurations. Skipping image mapping.")
            use_images = False
            config['image']['use_images'] = False
            config['image']['in_channels'] = 0
            logging.info("Set config['image']['in_channels'] to 0")

    train_data_processed = _process_csv(
        train_csv_path, 
        ptid_to_image_map, 
        config, 
        is_train_data=True,
        current_actual_scan_types=actual_scan_types_for_mapping
    )
    if train_data_processed is None:
        logging.error("Training data processing failed. Cannot continue.")
        return None
        
    X_train, y_train, train_image_paths, train_subject_ids, feature_columns, _ = train_data_processed
    
    if X_train is None or y_train is None:
        logging.error(f"Training data processing for {train_csv_path} resulted in no data. Cannot continue.")
        return None

    test_data_processed = _process_csv(
        test_csv_path, 
        ptid_to_image_map, 
        config, 
        is_train_data=False, 
        current_actual_scan_types=actual_scan_types_for_mapping
    )
    if test_data_processed is None:
        logging.warning("Test data processing failed. Proceeding without test data if possible.")
        X_test, y_test, test_image_paths, test_subject_ids = None, None, [], []
    else:
        X_test, y_test, test_image_paths, test_subject_ids, _ = test_data_processed 

    if X_test is None or y_test is None:
        logging.warning(f"Test data processing for {test_csv_path} resulted in no data. Val_loader will be None.")

    train_dataset = ADNIDataset(
        X_train,
        y_train,
        image_paths=train_image_paths,
        subject_ids=train_subject_ids,
        config=config,
        feature_columns=feature_columns
    )
    logging.info(f"Training dataset created. Size: {len(train_dataset)}")

    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_dataset) > 0 else None

    val_loader = None
    if X_test is not None and y_test is not None and len(X_test) > 0 :
        test_dataset = ADNIDataset(
            X_test,
            y_test,
            image_paths=test_image_paths,
            subject_ids=test_subject_ids,
            config=config,
            feature_columns=feature_columns
        )
        logging.info(f"Test (validation) dataset created. Size: {len(test_dataset)}")
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None
    else:
        logging.info("No test data available to create validation loader.")

    class_weights = None
    if config.get('training', {}).get('use_class_weights', False) and y_train is not None and len(y_train) > 0:
        try:
            unique_classes_train = np.unique(y_train.numpy())
            if len(unique_classes_train) > 1:
                weights = compute_class_weight('balanced', classes=unique_classes_train, y=y_train.numpy())
                class_weights = torch.tensor(weights, dtype=torch.float32)
                logging.info(f"Computed class weights: {class_weights}")
            else:
                logging.warning(f"Only one class found in training data: {unique_classes_train}. Cannot compute class weights.")
        except ValueError as e:
            logging.warning(f"Could not compute class weights: {e}")
    
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving updated configuration: {e}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'feature_columns': feature_columns,
        'config': config,
        'class_weights': class_weights
    }
