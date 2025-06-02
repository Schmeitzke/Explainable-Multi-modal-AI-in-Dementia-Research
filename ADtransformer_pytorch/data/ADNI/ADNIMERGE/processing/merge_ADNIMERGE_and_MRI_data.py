import pandas as pd

def merge_mri():
    adnimerge_file = ""
    all_images_file = ""
    output_file = ""
    output_image_ids_file = ""
    
    print("Reading ADNIMERGE_pruned dataset...")
    adnimerge = pd.read_csv(adnimerge_file, low_memory=False)
    
    print("Filtering participants to include only CN, AD, EMCI, LMCI, or SMC diagnoses...")
    valid_diagnoses = ['CN', 'AD', 'EMCI', 'LMCI', 'SMC'] 
    adnimerge = adnimerge[adnimerge['DX_bl'].isin(valid_diagnoses)]
    print(f"After diagnosis filtering: {len(adnimerge)} participants")
    
    print("Filtering to baseline visits only...")
    baseline_mask = (adnimerge['Years_bl'] == 0) & (adnimerge['Month_bl'] == 0)
    adnimerge = adnimerge[baseline_mask]
    print(f"After baseline filtering: {len(adnimerge)} participants")
    
    adnimerge_pruned = adnimerge[['RID', 'PTID', 'EXAMDATE']].copy()
    print(f"ADNIMERGE filtered dataset: {len(adnimerge_pruned)} patients")
    
    adnimerge_pruned['EXAMDATE'] = pd.to_datetime(adnimerge_pruned['EXAMDATE'], format='%d/%m/%Y')
    
    print("Reading All_Images dataset...")
    all_images = pd.read_csv(all_images_file, low_memory=False)
    
    print("Filtering for specific descriptions...")

    primary_description = "MPRAGE"
    secondary_description = "MT1; GradWarp; N3m"
    tertiary_description = "Sagittal 3D FLAIR"
    quaternary_description = "MP-RAGE"

    primary_mask = (all_images['description'] == primary_description)
    primary_mri = all_images[primary_mask].copy()
    
    secondary_mask = (all_images['description'] == secondary_description)
    secondary_mri = all_images[secondary_mask].copy()
    
    tertiary_mask = (all_images['description'] == tertiary_description)
    tertiary_mri = all_images[tertiary_mask].copy()
    
    quaternary_mask = (all_images['description'] == quaternary_description)
    quaternary_mri = all_images[quaternary_mask].copy()
            
    mri_columns = ['subject_id', 'image_id', 'image_date', 'description']
    primary_mri = primary_mri[mri_columns].copy()
    secondary_mri = secondary_mri[mri_columns].copy()
    tertiary_mri = tertiary_mri[mri_columns].copy()
    quaternary_mri = quaternary_mri[mri_columns].copy()
        
    primary_mri['image_date'] = pd.to_datetime(primary_mri['image_date'])
    secondary_mri['image_date'] = pd.to_datetime(secondary_mri['image_date'])
    tertiary_mri['image_date'] = pd.to_datetime(tertiary_mri['image_date'])
    quaternary_mri['image_date'] = pd.to_datetime(quaternary_mri['image_date'])
    
    print(f"Filtered primary MRI dataset: {len(primary_mri)} images")
    print(f"Filtered secondary MRI dataset: {len(secondary_mri)} images")
    print(f"Filtered tertiary MRI dataset: {len(tertiary_mri)} images")
    print(f"Filtered quaternary MRI dataset: {len(quaternary_mri)} images")
    
    print("Merging datasets...")
    
    merged_df = pd.DataFrame()
    
    print("Finding exact date matches for primary description...")
    merged_primary = pd.merge(
        adnimerge_pruned,
        primary_mri,
        left_on='PTID',
        right_on='subject_id',
        how='inner',
        suffixes=('', '_mri')
    )
    
    merged_primary['date_diff'] = (merged_primary['image_date'] - merged_primary['EXAMDATE']).dt.days
    
    exact_date_matches = merged_primary[merged_primary['date_diff'] == 0].copy()
    print(f"Found {len(exact_date_matches)} exact date matches with primary description")
    
    merged_df = pd.concat([merged_df, exact_date_matches])
    
    matched_ptids = exact_date_matches['PTID'].unique()
    remaining_patients = adnimerge_pruned[~adnimerge_pruned['PTID'].isin(matched_ptids)].copy()
    
    print(f"Finding closest matches for {len(remaining_patients)} remaining patients...")
    
    closest_matches = []
    
    for _, patient in remaining_patients.iterrows():
        primary_patient_mri = primary_mri[primary_mri['subject_id'] == patient['PTID']]
        secondary_patient_mri = secondary_mri[secondary_mri['subject_id'] == patient['PTID']]
        tertiary_patient_mri = tertiary_mri[tertiary_mri['subject_id'] == patient['PTID']]
        quaternary_patient_mri = quaternary_mri[quaternary_mri['subject_id'] == patient['PTID']]
        
        best_match = None
        
        if len(primary_patient_mri) > 0:
            primary_patient_mri = primary_patient_mri.copy()
            primary_patient_mri['date_diff'] = abs((primary_patient_mri['image_date'] - patient['EXAMDATE']).dt.days)
            within_threshold = primary_patient_mri[primary_patient_mri['date_diff'] <= 90]
            
            if len(within_threshold) > 0:
                best_match = within_threshold.loc[within_threshold['date_diff'].idxmin()].copy()
        
        if best_match is None and len(secondary_patient_mri) > 0:
            secondary_patient_mri = secondary_patient_mri.copy()
            secondary_patient_mri['date_diff'] = abs((secondary_patient_mri['image_date'] - patient['EXAMDATE']).dt.days)
            within_threshold = secondary_patient_mri[secondary_patient_mri['date_diff'] <= 90]
            
            if len(within_threshold) > 0:
                best_match = within_threshold.loc[within_threshold['date_diff'].idxmin()].copy()

        if best_match is None and len(tertiary_patient_mri) > 0:
            tertiary_patient_mri = tertiary_patient_mri.copy()
            tertiary_patient_mri['date_diff'] = abs((tertiary_patient_mri['image_date'] - patient['EXAMDATE']).dt.days)
            within_threshold = tertiary_patient_mri[tertiary_patient_mri['date_diff'] <= 90]
            
            if len(within_threshold) > 0:
                best_match = within_threshold.loc[within_threshold['date_diff'].idxmin()].copy()

        if best_match is None and len(quaternary_patient_mri) > 0:
            quaternary_patient_mri = quaternary_patient_mri.copy()
            quaternary_patient_mri['date_diff'] = abs((quaternary_patient_mri['image_date'] - patient['EXAMDATE']).dt.days)
            within_threshold = quaternary_patient_mri[quaternary_patient_mri['date_diff'] <= 90]
            
            if len(within_threshold) > 0:
                best_match = within_threshold.loc[within_threshold['date_diff'].idxmin()].copy()

        if best_match is not None:
            merged_row = patient.copy()
            for col in best_match.index:
                if col not in merged_row.index:
                    merged_row[col] = best_match[col]
            
            closest_matches.append(merged_row)
    
    if closest_matches:
        closest_matches_df = pd.DataFrame(closest_matches)
        print(f"Found {len(closest_matches_df)} closest matches within 3 months")
        
        merged_df = pd.concat([merged_df, closest_matches_df])
    else:
        print("No additional matches found within 3 months")
    
    result_columns = ['RID', 'PTID', 'EXAMDATE', 'image_date', 'date_diff', 'image_id', 'description']
    
    available_columns = [col for col in result_columns if col in merged_df.columns]
    merged_df = merged_df[available_columns].sort_values('RID')
    
    print(f"Columns in final dataset: {merged_df.columns.tolist()}")
    
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    print(f"Total matched patients: {len(merged_df)}")
    
    print("Generating text file with image IDs grouped by description...")
    
    descriptions = merged_df['description'].unique()
    
    with open(output_image_ids_file, 'w') as f:
        for description in descriptions:
            image_ids = merged_df[merged_df['description'] == description]['image_id'].tolist()
            
            f.write(f"{description}: {','.join(map(str, image_ids))}\n")
    
    print(f"Image IDs by description saved to {output_image_ids_file}")

if __name__ == "__main__":
    merge_mri()
