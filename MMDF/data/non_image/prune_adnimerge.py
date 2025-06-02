import pandas as pd

def prune_adnimerge():
    input_file = ""
    output_file = ""
    
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Original data shape: {df.shape}")
    
    columns_to_delete = [
        'RID', 'COLPROT', 'ORIGPROT', 'SITE', 'EXAMDATE',
        'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV',
        'FBB', 'CDRSB', 'ADAS11',
        'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
        'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
        'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan',
        'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal', 'FSVERSION', 'IMAGEUID',
        'DX', 'mPACCdigit', 'mPACCtrailsB', 'EXAMDATE_bl',
        'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl', 'RAVLT_learning_bl',
        'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl', 'LDELTOTAL_BL', 'DIGITSCOR_bl', 'TRABSCOR_bl',
        'FAQ_bl', 'mPACCdigit_bl', 'mPACCtrailsB_bl', 'FLDSTRENG_bl', 'FSVERSION_bl', 'IMAGEUID_bl', 'Ventricles_bl',
        'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'MOCA_bl',
        'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl', 'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl',
        'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl', 'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl',
        'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'FDG_bl', 'PIB_bl', 'AV45_bl', 'FBB_bl', 'Years_bl',
        'Month_bl', 'Month', 'M', 'update_stamp'
    ]

    df_pruned = df.drop(columns=columns_to_delete)
    print(f"After selecting columns, data shape: {df_pruned.shape}")
    
    baseline_mask = (df_pruned['VISCODE'] == 'bl')
    df_baseline = df_pruned[baseline_mask]
    print(f"After filtering to baseline only, data shape: {df_baseline.shape}")

    scan_mask = (df_baseline['FLDSTRENG'] == '1.5 Tesla MRI') | (df_baseline['FLDSTRENG'] == '3 Tesla MRI')
    df_baseline = df_baseline[scan_mask]
    print(f"After filtering to baseline only, data shape: {df_baseline.shape}")
    df_baseline = df_baseline.drop(columns=['FLDSTRENG'])

    df_baseline = df_baseline.drop(columns=['VISCODE'])
    
    df_baseline = df_baseline.rename(columns={'PTGENDER': 'GENDER'})

    df_baseline = df_baseline.rename(columns={'DX_bl': 'DIAGNOSIS'})

    df_baseline['DIAGNOSIS'] = df_baseline['DIAGNOSIS'].map({
        'CN': 0,
        'LMCI': 1,
        'EMCI': 1,
        'SMC' : 1,
        'AD': 2
    })

    df_baseline['GENDER'] = df_baseline['GENDER'].map({
        'Male': 1,
        'Female': 2
    })

    df_baseline = df_baseline.rename(columns={'APOE4': 'GENOTYPE'})

    df_baseline['AGE'] = df_baseline['AGE'].apply(lambda x: int(x) if isinstance(x, (int, float)) and not pd.isna(x) else pd.NA)
    df_baseline['DIAGNOSIS'] = df_baseline['DIAGNOSIS'].apply(lambda x: int(x) if isinstance(x, (int, float)) and not pd.isna(x) else pd.NA)

    df_baseline.to_csv(output_file, index=False)
    print(f"Pruned baseline data saved to {output_file}")

if __name__ == "__main__":
    prune_adnimerge()
