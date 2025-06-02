import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import config_data
import utils

class ClinicalDataProcessor:
    def __init__(self,
                 input_csv_path=config_data.INPUT_CSV_PATH,
                 ptid_col=config_data.PTID_COLUMN,
                 target_col=config_data.TARGET_COLUMN,
                 categorical_cols_ohe=config_data.CATEGORICAL_COLS_OHE,
                 knn_n_neighbors=config_data.KNN_N_NEIGHBORS):
        self.input_csv_path = input_csv_path
        self.ptid_col = ptid_col
        self.target_col = target_col
        self.categorical_cols_ohe = categorical_cols_ohe
        self.knn_n_neighbors = knn_n_neighbors
        self.numerical_cols = []
        self.imputer = None
        self.scaler = None
        self.one_hot_encoded_columns = []
        self.fitted_feature_cols_ordered = []

    def load_data(self):
        """Loads data from the input CSV."""
        print(f"Loading data from {self.input_csv_path}...")
        df = pd.read_csv(self.input_csv_path)
        print(f"Data loaded. Shape: {df.shape}")
        
        utils.verify_target_mapping(df, self.target_col)
        
        df.dropna(subset=[self.target_col], inplace=True)
        print(f"Shape after dropping rows with NaN in target '{self.target_col}': {df.shape}")
        
        return df

    def _identify_numerical_cols(self, df):
        """Identifies numerical columns, excluding PTID, target, and categorical_ohe columns."""
        exclude_cols = [self.ptid_col, self.target_col] + self.categorical_cols_ohe
        self.numerical_cols = [
            col for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        print(f"Identified {len(self.numerical_cols)} numerical columns for imputation/scaling.")


    def one_hot_encode(self, df: pd.DataFrame, fit_mode: bool = True):
        """Performs one-hot encoding on specified categorical columns."""
        print("Starting one-hot encoding...")
        for col in self.categorical_cols_ohe:
            if col in df.columns:
                df[col] = df[col].astype('category')
            else:
                print(f"Warning: Categorical column '{col}' not found in DataFrame.")
        
        df_encoded = pd.get_dummies(df, columns=self.categorical_cols_ohe, prefix=self.categorical_cols_ohe, dummy_na=False)
        
        if fit_mode:
            original_cols_set = set(df.columns)
            encoded_cols_set = set(df_encoded.columns)
            self.one_hot_encoded_columns = list(encoded_cols_set - original_cols_set)
            utils.save_encoded_columns(self.one_hot_encoded_columns, config_data.ENCODED_COLUMNS_PATH)
        else:
            expected_ohe_cols = utils.load_encoded_columns(config_data.ENCODED_COLUMNS_PATH)
            for col in expected_ohe_cols:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
        print(f"One-hot encoding completed. Shape: {df_encoded.shape}")
        return df_encoded

    def impute_missing_values(self, df_train_features: pd.DataFrame, df_test_features: pd.DataFrame = None):
        """
        Imputes missing values using KNNImputer.
        Fits on df_train_features and transforms both.
        Returns numpy arrays of imputed values.
        """
        print("Starting KNN imputation...")
        
        cols_to_impute = df_train_features.columns.tolist()
        
        print(f"Columns to impute count: {len(cols_to_impute)}")

        self.imputer = KNNImputer(n_neighbors=self.knn_n_neighbors, missing_values=np.nan)
        
        print("Fitting KNNImputer on training features...")
        self.imputer.fit(df_train_features)
        utils.save_object(self.imputer, config_data.IMPUTER_PATH)

        print("Transforming training features with KNNImputer...")
        df_train_imputed_values = self.imputer.transform(df_train_features)
        print("Training data imputation completed.")

        df_test_imputed_values = None
        if df_test_features is not None:
            print("Transforming test features with KNNImputer...")
            df_test_imputed_values = self.imputer.transform(df_test_features)
            print("Test data imputation completed.")
            
        return df_train_imputed_values, df_test_imputed_values


    def normalize_features(self, df_train_features: pd.DataFrame, df_test_features: pd.DataFrame = None):
        """
        Normalizes features using StandardScaler.
        Fits on df_train_features and transforms both.
        Returns numpy arrays of scaled values.
        """
        print("Starting feature normalization (StandardScaler)...")
        
        cols_to_scale = df_train_features.columns.tolist()
        print(f"Columns to scale count: {len(cols_to_scale)}")

        self.scaler = StandardScaler()
        
        print("Fitting StandardScaler on training features...")
        self.scaler.fit(df_train_features)
        utils.save_object(self.scaler, config_data.SCALER_PATH)

        print("Transforming training features with StandardScaler...")
        df_train_scaled_values = self.scaler.transform(df_train_features)
        print("Training data normalization completed.")

        df_test_scaled_values = None
        if df_test_features is not None:
            print("Transforming test features with StandardScaler...")
            df_test_scaled_values = self.scaler.transform(df_test_features)
            print("Test data normalization completed.")
            
        return df_train_scaled_values, df_test_scaled_values

    def process(self, test_size=0.2, random_state=42):
        """Main processing pipeline."""
        df = self.load_data()

        df[self.target_col] = df[self.target_col].astype(int)

        df_train, df_test = train_test_split(df, test_size=test_size,
                                             random_state=random_state, stratify=df[self.target_col])
        
        print(f"Train set shape: {df_train.shape}, Test set shape: {df_test.shape}")

        df_train_ohe = self.one_hot_encode(df_train.copy(), fit_mode=True)
        df_test_ohe = self.one_hot_encode(df_test.copy(), fit_mode=False)
        
        temp_numerical_cols_original = []
        exclude_for_num_id = [self.ptid_col, self.target_col] + self.categorical_cols_ohe
        for col in df_train.columns:
            if col not in exclude_for_num_id and pd.api.types.is_numeric_dtype(df_train[col]):
                temp_numerical_cols_original.append(col)
        
        self.fitted_feature_cols_ordered = temp_numerical_cols_original + self.one_hot_encoded_columns
        
        self.numerical_cols = temp_numerical_cols_original
        
        print(f"Canonical feature column order (count: {len(self.fitted_feature_cols_ordered)}): {self.fitted_feature_cols_ordered}")

        ptid_train = df_train_ohe[self.ptid_col]
        target_train = df_train_ohe[self.target_col]
        features_train = df_train_ohe.drop(columns=[self.ptid_col, self.target_col])

        ptid_test = df_test_ohe[self.ptid_col]
        target_test = df_test_ohe[self.target_col]
        features_test = df_test_ohe.drop(columns=[self.ptid_col, self.target_col])

        features_train_aligned = features_train.reindex(columns=self.fitted_feature_cols_ordered, fill_value=0)
        features_test_aligned = features_test.reindex(columns=self.fitted_feature_cols_ordered, fill_value=0)

        features_train_imputed_vals, features_test_imputed_vals = self.impute_missing_values(
            features_train_aligned,
            features_test_aligned
        )
        
        if hasattr(self.imputer, 'feature_names_in_') and list(self.imputer.feature_names_in_) != self.fitted_feature_cols_ordered:
            print("Warning: Imputer feature names do not match expected ordered feature columns!")
            print(f"Expected: {self.fitted_feature_cols_ordered}")
            print(f"Imputer saw: {list(self.imputer.feature_names_in_)}")


        features_train_imputed = pd.DataFrame(features_train_imputed_vals, columns=self.fitted_feature_cols_ordered, index=features_train_aligned.index)
        if features_test_imputed_vals is not None:
            features_test_imputed = pd.DataFrame(features_test_imputed_vals, columns=self.fitted_feature_cols_ordered, index=features_test_aligned.index)
        else:
            features_test_imputed = pd.DataFrame(columns=self.fitted_feature_cols_ordered)

        features_train_scaled_vals, features_test_scaled_vals = self.normalize_features(
            features_train_imputed,
            features_test_imputed if features_test_imputed_vals is not None else None
        )
        
        features_train_scaled = pd.DataFrame(features_train_scaled_vals, columns=self.fitted_feature_cols_ordered, index=features_train_imputed.index)
        if features_test_scaled_vals is not None:
            features_test_scaled = pd.DataFrame(features_test_scaled_vals, columns=self.fitted_feature_cols_ordered, index=features_test_imputed.index)
        else:
            features_test_scaled = pd.DataFrame(columns=self.fitted_feature_cols_ordered)
        
        df_train_processed = pd.concat([ptid_train.reset_index(drop=True), 
                                        target_train.reset_index(drop=True), 
                                        features_train_scaled.reset_index(drop=True)], axis=1)
        
        df_test_processed_list = [ptid_test.reset_index(drop=True), target_test.reset_index(drop=True)]
        df_test_processed_list.append(features_test_scaled.reset_index(drop=True))
            
        df_test_processed = pd.concat(df_test_processed_list, axis=1)

        df_train_processed.to_csv(config_data.PROCESSED_CSV_PATH.replace(".csv", "_train.csv"), index=False)
        df_test_processed.to_csv(config_data.PROCESSED_CSV_PATH.replace(".csv", "_test.csv"), index=False)
        print(f"Processed training data saved. Shape: {df_train_processed.shape}")
        print(f"Processed test data saved. Shape: {df_test_processed.shape}")
        
        return df_train_processed, df_test_processed

    def transform_new_data(self, df_new: pd.DataFrame):
        """Transforms new, unseen data using pre-fitted imputer and scaler."""
        if self.imputer is None or self.scaler is None or not self.fitted_feature_cols_ordered:
            print("Loading pre-fitted imputer, scaler, and OHE column list...")
            self.imputer = utils.load_object(config_data.IMPUTER_PATH)
            self.scaler = utils.load_object(config_data.SCALER_PATH)
            self.one_hot_encoded_columns = utils.load_encoded_columns(config_data.ENCODED_COLUMNS_PATH)
            
            if hasattr(self.imputer, 'feature_names_in_'):
                self.fitted_feature_cols_ordered = self.imputer.feature_names_in_.tolist()
                print(f"Loaded feature order from imputer: {self.fitted_feature_cols_ordered}")
            else:
                raise AttributeError("Fitted imputer does not have 'feature_names_in_'. Cannot determine feature order for new data.")
        
        print("Transforming new data...")
        df_new_copy = df_new.copy()

        df_new_ohe = self.one_hot_encode(df_new_copy, fit_mode=False)
        
        ptid_new = None
        target_new = None 
        
        if self.ptid_col in df_new_ohe.columns:
            ptid_new = df_new_ohe[self.ptid_col]
        if self.target_col in df_new_ohe.columns:
            target_new = df_new_ohe[self.target_col]
        
        features_new = df_new_ohe.drop(columns=[self.ptid_col, self.target_col], errors='ignore')
        features_new_aligned = features_new.reindex(columns=self.fitted_feature_cols_ordered, fill_value=0)
        
        imputed_values_new = self.imputer.transform(features_new_aligned)
        features_new_imputed = pd.DataFrame(imputed_values_new, columns=self.fitted_feature_cols_ordered, index=features_new_aligned.index)

        scaled_values_new = self.scaler.transform(features_new_imputed)
        features_new_scaled = pd.DataFrame(scaled_values_new, columns=self.fitted_feature_cols_ordered, index=features_new_imputed.index)
        
        df_new_processed_list = []
        if ptid_new is not None:
            df_new_processed_list.append(ptid_new.reset_index(drop=True))
        if target_new is not None:
            df_new_processed_list.append(target_new.reset_index(drop=True))
        df_new_processed_list.append(features_new_scaled.reset_index(drop=True))
        
        df_new_processed = pd.concat(df_new_processed_list, axis=1)
        
        print(f"New data transformation completed. Shape: {df_new_processed.shape}")
        return df_new_processed

if __name__ == "__main__":
    processor = ClinicalDataProcessor()
    df_train_processed, df_test_processed = processor.process()

    print("\n--- Training Data Head ---")
    print(df_train_processed.head())
    print(f"Training Data Shape: {df_train_processed.shape}")
    print(f"Training Data NaN counts:\n{df_train_processed.isnull().sum().sum()} (should be 0 for features)")


    print("\n--- Test Data Head ---")
    print(df_test_processed.head())
    print(f"Test Data Shape: {df_test_processed.shape}")
    print(f"Test Data NaN counts:\n{df_test_processed.isnull().sum().sum()} (should be 0 for features)")

    if not df_test_processed.empty:
        sample_new_data = pd.read_csv(config_data.INPUT_CSV_PATH).iloc[df_test_processed.index[:5]]
        print("\n--- Transforming a sample of new data ---")
        if config_data.TARGET_COLUMN in sample_new_data.columns:
            sample_new_data_for_transform = sample_new_data.drop(columns=[config_data.TARGET_COLUMN])
        else:
            sample_new_data_for_transform = sample_new_data
            
        transformed_sample = processor.transform_new_data(sample_new_data_for_transform)
        print(transformed_sample.head())