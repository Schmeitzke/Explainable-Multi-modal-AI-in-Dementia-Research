import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np
import os
import glob
from typing import List, Tuple, Optional, Callable
import torchvision.transforms as T
import logging
from skimage.transform import resize
from pathlib import Path

import config

logger = logging.getLogger(__name__)

class AdniDataset(Dataset):
    """
    Loads ADNI clinical and MRI data.
    Connects CSV data with MRI scans using PTID.
    """
    def __init__(self,
                 csv_path: str,
                 image_base_dir: str,
                 image_modalities: List[str],
                 num_clinical_features: int,
                 image_size: Tuple[int, int],
                 num_classes: int,
                 transform: Optional[Callable] = None):
        super().__init__()
        self.image_base_dir = image_base_dir
        self.image_modalities = image_modalities
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            self.df = pd.DataFrame()
            self.image_paths = []
            self.clinical_data = []
            self.labels = []
            return

        if config.perform_binary_classification_cn_ad:
            print(f"Performing binary classification (CN vs AD) for {csv_path}.")
            initial_length = len(self.df)
            self.df = self.df[self.df['DIAGNOSIS'].isin([0, 2])].copy()
            self.df.loc[self.df['DIAGNOSIS'] == 2, 'DIAGNOSIS'] = 1
            print(f"Filtered for CN/AD and remapped labels. Dropped {initial_length - len(self.df)} samples.")
            print(f"Shape after binary classification filtering: {self.df.shape}")
            
            if self.df.empty:
                print(f"Error: DataFrame is empty after filtering for CN/AD in {csv_path}")
                self.image_paths = []
                self.clinical_data = []
                self.labels = []
                return

        self.ptid_to_image_path = self._create_image_path_map()

        self.df['has_image'] = self.df['PTID'].apply(lambda x: x in self.ptid_to_image_path)
        self.df_filtered = self.df[self.df['has_image']].reset_index(drop=True)
        
        if len(self.df_filtered) == 0 and len(self.df) > 0:
            print(f"Warning: No matching images found for PTIDs in {csv_path}. Dataset will be empty.")
        elif len(self.df_filtered) < len(self.df):
            print(f"Warning: Dropped {len(self.df) - len(self.df_filtered)} samples from {csv_path} due to missing images.")

        clinical_features_df = self.df_filtered.iloc[:, 2:2+num_clinical_features]
        
        clinical_features_numeric = clinical_features_df.apply(pd.to_numeric, errors='coerce')
        
        if clinical_features_numeric.isnull().any().any():
            print("Warning: Some clinical features contain non-numeric values that were converted to NaN.")
            print("Rows with NaN values:")
            nan_rows = clinical_features_numeric.isnull().any(axis=1)
            print(self.df_filtered.loc[nan_rows, ['PTID']])
            
            clinical_features_numeric = clinical_features_numeric.fillna(0.0)
            print("NaN values have been filled with 0.0")
        
        clinical_array = clinical_features_numeric.values.astype(np.float32)
        
        self.clinical_data = torch.from_numpy(clinical_array)
        
        self.image_paths = [self.ptid_to_image_path[ptid] for ptid in self.df_filtered['PTID']]
        
        raw_labels = torch.tensor(self.df_filtered['DIAGNOSIS'].values, dtype=torch.long)
        if raw_labels.min() == 1 and self.num_classes == (raw_labels.max()):
             self.labels = raw_labels - 1
        else:
             self.labels = raw_labels

        self.mask_paths = []
        self.missing_mask_indices = []

        for idx, img_path_str in enumerate(self.image_paths):
            try:
                img_path_obj = Path(img_path_str)
                img_filename = img_path_obj.name
                
                modality_dir_name = img_path_obj.parent.name
                base_processed_dir = img_path_obj.parent.parent

                mask_dir_name = f"{modality_dir_name}_mask"
                mask_filename = img_filename.replace('.nii.gz', '_mask.nii.gz')
                
                expected_mask_path = base_processed_dir / mask_dir_name / mask_filename
                
                if expected_mask_path.exists():
                    self.mask_paths.append(str(expected_mask_path))
                else:
                    logger.warning(f"Mask file not found for image {img_path_str} at expected path {str(expected_mask_path)}")
                    self.mask_paths.append(None)
                    self.missing_mask_indices.append(idx)
            except Exception as e:
                logger.error(f"Error processing image path {img_path_str} for mask lookup: {e}", exc_info=True)
                self.mask_paths.append(None)
                self.missing_mask_indices.append(idx)


        if self.missing_mask_indices:
            logger.warning(f"Found {len(self.missing_mask_indices)} missing mask files out of {len(self.image_paths)} total samples.")


    def _create_image_path_map(self) -> dict:
        ptid_map = {}
        for mod_dir_name in self.image_modalities:
            mod_full_path = os.path.join(self.image_base_dir, mod_dir_name)
            if not os.path.isdir(mod_full_path):
                print(f"Warning: Image modality directory not found: {mod_full_path}")
                continue
            
            for img_file in glob.glob(os.path.join(mod_full_path, "*.nii.gz")):
                basename = os.path.basename(img_file)
                parts = basename.split('_')
                if len(parts) >= 3 and parts[1] == 'S':
                    ptid = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    if ptid not in ptid_map:
                        ptid_map[ptid] = img_file
        return ptid_map

    def __len__(self):
        return len(self.df_filtered)

    def __getitem__(self, idx: int):
        clinical_features = self.clinical_data[idx]
        label = self.labels[idx]
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            img_obj = nib.load(img_path)
            mri_data = img_obj.get_fdata().astype(np.float32)

            if mri_data.shape == (self.image_size[0], self.image_size[1], 3):
                mri_data = np.transpose(mri_data, (2, 0, 1))
            elif mri_data.shape == (3, self.image_size[0], self.image_size[1]):
                pass
            else:
                print(f"Warning: Image {img_path} has unexpected shape {mri_data.shape}. Expected (H,W,3) or (3,H,W).")
                if mri_data.shape[0] == 3 and mri_data.shape[1] == self.image_size[0] and mri_data.shape[2] == self.image_size[1]:
                     pass
                elif mri_data.shape[2] == 3 and mri_data.shape[0] == self.image_size[0] and mri_data.shape[1] == self.image_size[1]:
                     mri_data = np.transpose(mri_data, (2, 0, 1))
                else:
                    print(f"Error: Could not reshape image {img_path} from {mri_data.shape} to (3, {self.image_size[0]}, {self.image_size[1]}). Using zeros.")
                    mri_data = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=np.float32)


            mri_tensor = torch.from_numpy(mri_data)
            
            if self.transform:
                min_val = mri_tensor.min()
                max_val = mri_tensor.max()
                if max_val > min_val:
                    mri_tensor = (mri_tensor - min_val) / (max_val - min_val)
                else:
                    mri_tensor = torch.zeros_like(mri_tensor)
                
                mri_tensor = self.transform(mri_tensor)
            else:
                min_val = mri_tensor.min()
                max_val = mri_tensor.max()
                if max_val > min_val:
                    mri_tensor = (mri_tensor - min_val) / (max_val - min_val)
                else:
                    mri_tensor = torch.zeros_like(mri_tensor)

        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Returning zeros.")
            mri_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]))
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}. Returning zeros.")
            mri_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]))
            
        mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        if mask_path is not None:
            try:
                mask_obj = nib.load(mask_path)
                mask_data = mask_obj.get_fdata().astype(np.float32)
                if mask_data.ndim > 2:
                    mask_data = np.squeeze(mask_data)
                if mask_data.ndim != 2:
                    raise ValueError(f"Loaded mask {mask_path} is not 2D after squeeze, shape: {mask_data.shape}")
                
                if mask_data.shape[0] != self.image_size[0] or mask_data.shape[1] != self.image_size[1]:
                    logger.warning(f"Mask {mask_path} shape {mask_data.shape} differs from target image size {self.image_size}. Resizing mask...")
                    mask_data = resize(mask_data, self.image_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)

                mask_data = (mask_data > 0.5).astype(np.float32) 
                mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)
            except FileNotFoundError:
                logger.error(f"Mask file {mask_path} not found during getitem (should have been caught earlier?). Using zeros.")
            except ValueError as ve:
                logger.error(f"Error processing mask {mask_path}: {ve}. Using zeros.")
            except Exception as e:
                logger.error(f"Error loading or processing mask {mask_path}: {e}. Using zeros.")
        
        return clinical_features, mri_tensor, mask_tensor, label


def get_dataloaders(train_csv_path: str = config.train_csv_path,
                    test_csv_path: str = config.test_csv_path,
                    image_base_dir: str = config.image_data_base_dir,
                    image_modalities: List[str] = config.image_modalities,
                    batch_size: int = config.batch_size,
                    num_clinical_features: int = config.num_clinical_features,
                    mri_image_size: Tuple[int, int] = config.mri_image_size,
                    num_classes: int = config.num_classes
                    ):
    """
    Returns training and validation DataLoaders using the AdniDataset.
    Supports both binary (CN vs AD) and multiclass (CN vs MCI vs AD) classification.
    """
    if config.perform_binary_classification_cn_ad:
        print(f"Using binary classification mode: {config.class_names}")
    else:
        print(f"Using multiclass classification mode: {config.class_names}")
    
    train_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((144, 144)),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=5, scale=(0.8, 1.2)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor()
    ])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((144, 144)),
        T.ToTensor()
    ])

    train_ds = AdniDataset(csv_path=train_csv_path,
                           image_base_dir=image_base_dir,
                           image_modalities=image_modalities,
                           num_clinical_features=num_clinical_features,
                           image_size=mri_image_size,
                           num_classes=num_classes,
                           transform=train_transform)
    
    val_ds = AdniDataset(csv_path=test_csv_path,
                         image_base_dir=image_base_dir,
                         image_modalities=image_modalities,
                         num_clinical_features=num_clinical_features,
                         image_size=mri_image_size,
                         num_classes=num_classes,
                         transform=val_transform)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check CSV paths, image paths, and PTID matching.")
    if len(val_ds) == 0:
        print("Warning: Validation dataset is empty. This might be intended if test_csv_path is for a final test set only.")

    print(f"Training dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    
    try:
        train_labels = [train_ds[i][3].item() for i in range(len(train_ds))]
        val_labels = [val_ds[i][3].item() for i in range(len(val_ds))]
        
        import collections
        train_class_counts = collections.Counter(train_labels)
        val_class_counts = collections.Counter(val_labels)
        
        print(f"Training class distribution: {dict(train_class_counts)}")
        print(f"Validation class distribution: {dict(val_class_counts)}")
    except Exception as e:
        print(f"Could not compute class distribution: {e}")

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
                            
    return train_loader, val_loader
