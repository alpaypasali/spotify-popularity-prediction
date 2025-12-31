"""
Business logic services for Social Media Hit Prediction.
Uses FeatureEnginering.py preprocessing pipeline and spotify_voting_model.pkl
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings

# Suppress numpy warnings for empty slices (handled explicitly in code)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

# Import metrics (with fallback if not available)
try:
    from .metrics import (
        model_loading_status,
        model_loading_duration,
        dataset_size,
        prediction_scores
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Import FeatureEnginering.py functions
BASE_DIR = Path(__file__).resolve().parent

# Add EDA directory to path for imports
EDA_DIR = BASE_DIR / 'EDA'
if str(EDA_DIR) not in sys.path:
    sys.path.insert(0, str(EDA_DIR))

try:
    # Import from EDA/FeatureEnginering.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "FeatureEnginering", 
        BASE_DIR / 'EDA' / 'FeatureEnginering.py'
    )
    if spec and spec.loader:
        feature_eng_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_eng_module)
        
        grab_col_names = feature_eng_module.grab_col_names
        preprocess_pipeline = feature_eng_module.preprocess_pipeline
        spotify_data_prep = feature_eng_module.spotify_data_prep
        encode_categoricals = feature_eng_module.encode_categoricals
        
        FEATURE_ENG_AVAILABLE = True
        print("FeatureEnginering.py imported successfully")
    else:
        raise ImportError("Could not load FeatureEnginering module")
except Exception as e:
    print(f"Warning: Could not import FeatureEnginering.py: {e}")
    FEATURE_ENG_AVAILABLE = False
    # Define dummy functions to prevent errors
    def grab_col_names(*args, **kwargs):
        return [], [], []
    def preprocess_pipeline(df):
        return df
    def spotify_data_prep(df, *args, **kwargs):
        return df
    def encode_categoricals(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()


class SpotifyPredictionService:
    """Service for Spotify hit prediction using FeatureEnginering.py pipeline."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(SpotifyPredictionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Skip initialization if already done (singleton pattern)
        if SpotifyPredictionService._initialized:
            return
        
        # Model eğitimi için train verisi (reference için - encoding bilgileri)
        self.train_dataset_path = BASE_DIR / 'data' / 'processed' / 'spotify_emotion_train.csv'
        # Arayüz için test verisi
        self.dataset_path = BASE_DIR / 'data' / 'processed' / 'spotify_emotion_test.csv'
        self.model = None
        self.df = None
        self.df_train = None
        self.rare_maps = {}  # Categorical encoding için
        self.label_encoders = {}  # Binary encoding için
        self.cat_cols = []  # Kategorik kolonlar
        self.model_features = None  # Model'in beklediği feature'lar
        self.scaler = None  # Scaler (eğer varsa)
        self.selector = None  # Feature selector (eğer varsa)
        
        # Cached normalized feature arrays for similarity calculation
        self._normalized_features_cache = None
        self._feature_ranges_cache = None
        
        self.load_data()
        self.load_voting_model()
        self._prepare_encoding_info()
        self._prepare_similarity_cache()
        
        # Update metrics
        if METRICS_AVAILABLE:
            dataset_size.set(len(self.df) if self.df is not None and not self.df.empty else 0)
            model_loading_status.set(1 if self.model is not None else 0)
        
        SpotifyPredictionService._initialized = True
    
    def load_data(self):
        """Load Spotify datasets - train for reference, test for UI."""
        # Arayüz için test veriseti yükle
        try:
            if self.dataset_path.exists():
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        self.df = pd.read_csv(self.dataset_path, encoding=encoding)
                        # Normalize column names to uppercase (FeatureEnginering.py format)
                        self._normalize_columns()
                        # Convert to uppercase for FeatureEnginering.py compatibility
                        self.df.columns = [c.upper() for c in self.df.columns]
                        print(f"Test dataset (UI) loaded successfully with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error loading test dataset with {encoding}: {e}")
                        continue
            else:
                # Try alternative dataset files for UI
                alt_paths = [
                    BASE_DIR / 'data' / 'raw' / 'spotify_songs.csv',
                    BASE_DIR / 'data' / 'raw' / 'light_spotify_dataset.csv',
                    BASE_DIR / 'data' / 'processed' / 'spotify_emotion_final_clean.csv',
                    BASE_DIR / 'pythonEDA' / 'spotify_emotion_test.csv',
                    BASE_DIR / 'spotify_songs.csv',
                    BASE_DIR / 'light_spotify_dataset.csv',
                ]
                for path in alt_paths:
                    if path.exists():
                        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                            try:
                                self.df = pd.read_csv(path, encoding=encoding)
                                self._normalize_columns()
                                self.df.columns = [c.upper() for c in self.df.columns]
                                print(f"Dataset loaded from {path.name} with encoding: {encoding}")
                                break
                            except UnicodeDecodeError:
                                continue
                            except Exception as e:
                                print(f"Error loading {path.name} with {encoding}: {e}")
                                continue
                        if not self.df.empty:
                            break
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            import traceback
            traceback.print_exc()
            self.df = pd.DataFrame()
        
        # Train veriseti yükle (reference için - encoding bilgileri için)
        try:
            if self.train_dataset_path.exists():
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        self.df_train = pd.read_csv(self.train_dataset_path, encoding=encoding)
                        # Normalize column names
                        self._normalize_columns_for_df(self.df_train)
                        # Convert to uppercase
                        self.df_train.columns = [c.upper() for c in self.df_train.columns]
                        print(f"Train dataset (reference) loaded successfully with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error loading train dataset with {encoding}: {e}")
                        continue
            else:
                print("Train dataset not found - encoding info will be limited")
                self.df_train = pd.DataFrame()
        except Exception as e:
            print(f"Error loading train dataset: {e}")
            self.df_train = pd.DataFrame()
    
    def _normalize_columns(self):
        """Normalize column names to standard format (before uppercase conversion)."""
        if self.df.empty:
            return
        
        # Map various column name formats to standard names (lowercase)
        column_mapping = {
            'track_name': ['track_name', 'name', 'song', 'title', 'track name'],
            'artists': ['artist(s)_name', 'artist_name', 'artist', 'artists', 'artists name'],
            'danceability': ['danceability_%', 'danceability', 'dance'],
            'energy': ['energy_%', 'energy'],
            'valence': ['valence_%', 'valence'],
            'tempo': ['bpm', 'tempo'],
            'loudness': ['loudness'],
            'key': ['key'],
            'mode': ['mode'],
            'speechiness': ['speechiness_%', 'speechiness'],
            'acousticness': ['acousticness_%', 'acousticness'],
            'instrumentalness': ['instrumentalness_%', 'instrumentalness'],
            'liveness': ['liveness_%', 'liveness'],
            'streams': ['streams'],
            'popularity': ['popularity', 'in_spotify_charts'],
            'explicit': ['explicit'],
            'release_year': ['release_year', 'year', 'release year'],
            'genre': ['genre']
        }
        
        # Find and rename columns
        for standard_name, possible_names in column_mapping.items():
            for col in self.df.columns:
                col_lower = col.lower().strip()
                if col_lower in [n.lower() for n in possible_names] and standard_name not in self.df.columns:
                    self.df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # Convert percentage columns to 0-1 range if needed
        for col in ['danceability', 'energy', 'valence', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                if self.df[col].max() > 1:
                    self.df[col] = self.df[col] / 100.0
        
        # Convert numeric columns
        for col in ['streams', 'tempo', 'bpm', 'loudness', 'key', 'mode', 'popularity', 'explicit', 'release_year']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def _normalize_columns_for_df(self, df):
        """Normalize column names for a given dataframe (before uppercase conversion)."""
        if df.empty:
            return
        
        # Same mapping as _normalize_columns
        column_mapping = {
            'track_name': ['track_name', 'name', 'song', 'title', 'track name'],
            'artists': ['artist(s)_name', 'artist_name', 'artist', 'artists', 'artists name'],
            'danceability': ['danceability_%', 'danceability', 'dance'],
            'energy': ['energy_%', 'energy'],
            'valence': ['valence_%', 'valence'],
            'tempo': ['bpm', 'tempo'],
            'loudness': ['loudness'],
            'key': ['key'],
            'mode': ['mode'],
            'speechiness': ['speechiness_%', 'speechiness'],
            'acousticness': ['acousticness_%', 'acousticness'],
            'instrumentalness': ['instrumentalness_%', 'instrumentalness'],
            'liveness': ['liveness_%', 'liveness'],
            'streams': ['streams'],
            'popularity': ['popularity', 'in_spotify_charts'],
            'explicit': ['explicit'],
            'release_year': ['release_year', 'year', 'release year'],
            'genre': ['genre']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [n.lower() for n in possible_names] and standard_name not in df.columns:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # Convert percentage columns
        for col in ['danceability', 'energy', 'valence', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].max() > 1:
                    df[col] = df[col] / 100.0
        
        # Convert numeric columns
        for col in ['streams', 'tempo', 'bpm', 'loudness', 'key', 'mode', 'popularity', 'explicit', 'release_year']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    def _prepare_encoding_info(self):
        """Prepare encoding information from train dataset using FeatureEnginering.py pipeline."""
        if not FEATURE_ENG_AVAILABLE or self.df_train.empty:
            return
        
        try:
            # Apply preprocessing pipeline (same as in model training)
            df_train_processed = preprocess_pipeline(self.df_train.copy())
            df_train_processed = spotify_data_prep(df_train_processed)
            
            # Drop non-feature columns
            drop_cols = ['TRACK_NAME', 'ARTISTS', 'SOURCE', 'POPULARITY', 'STREAMS']
            df_train_processed = df_train_processed.drop(columns=[c for c in drop_cols if c in df_train_processed.columns])
            
            # Get categorical columns using FeatureEnginering.py function
            self.cat_cols, _, _ = grab_col_names(df_train_processed)
            
            # Use FeatureEnginering.py's encode_categoricals logic to extract encoding info
            # Create a dummy test dataframe with same structure for encode_categoricals
            dummy_test = df_train_processed.head(1).copy()
            
            # Apply encode_categoricals to get encoding maps (this will modify both dataframes)
            # We'll extract the encoding info from the process
            from sklearn.preprocessing import LabelEncoder
            
            # Extract rare label maps (same logic as encode_categoricals)
            for col in self.cat_cols:
                if col in df_train_processed.columns:
                    freq_ratio = df_train_processed[col].value_counts(normalize=True)
                    rare_labels = freq_ratio[freq_ratio < 0.01].index.tolist()
                    self.rare_maps[col] = rare_labels
            
            # Extract label encoders for binary columns (same logic as encode_categoricals)
            binary_cols = [col for col in self.cat_cols if df_train_processed[col].nunique() == 2]
            for col in binary_cols:
                if col in df_train_processed.columns:
                    le = LabelEncoder()
                    le.fit(df_train_processed[col])
                    self.label_encoders[col] = le
            
            print(f"Prepared encoding info for {len(self.cat_cols)} categorical columns using FeatureEnginering.py logic")
        except Exception as e:
            print(f"Error preparing encoding info: {e}")
            import traceback
            traceback.print_exc()
    
    def load_voting_model(self):
        """Load spotify_voting_model.pkl model and related files."""
        load_start_time = time.time()
        
        # Try multiple model paths
        model_paths = [
            BASE_DIR / 'model' / 'spotify_voting_model.pkl',
            BASE_DIR / 'spotify_voting_model.pkl',
            BASE_DIR / 'spotify_model.joblib',  # Fallback
        ]
        
        for model_path in model_paths:
            if model_path.exists():
                try:
                    # Suppress sklearn version warnings during model loading
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        self.model = joblib.load(model_path)
                    print(f"Voting model loaded from: {model_path}")
                    
                    # Record metrics
                    if METRICS_AVAILABLE:
                        load_duration = time.time() - load_start_time
                        model_loading_duration.observe(load_duration)
                        model_loading_status.set(1)
                    
                    # Try to get feature names from model
                    if hasattr(self.model, 'feature_names_in_'):
                        self.model_features = list(self.model.feature_names_in_)
                    elif hasattr(self.model, 'estimators_'):
                        # VotingRegressor or StackingRegressor
                        if len(self.model.estimators_) > 0:
                            first_est = self.model.estimators_[0]
                            if hasattr(first_est, 'feature_names_in_'):
                                self.model_features = list(first_est.feature_names_in_)
                    
                    # Check for StackingRegressor final_estimator
                    if not self.model_features and hasattr(self.model, 'final_estimator_'):
                        if hasattr(self.model.final_estimator_, 'feature_names_in_'):
                            self.model_features = list(self.model.final_estimator_.feature_names_in_)
                    
                    # If still no features, infer from train data
                    if not self.model_features and not self.df_train.empty:
                        self._infer_model_features()
                    
                    if self.model_features:
                        print(f"Model expects {len(self.model_features)} features")
                    else:
                        print("Warning: Could not determine model features")
                    
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Load scaler and selector if available
        scaler_path = BASE_DIR / 'spotify_scaler.joblib'
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                print("Scaler loaded successfully")
            except Exception as e:
                print(f"Error loading scaler: {e}")
        
        selector_path = BASE_DIR / 'spotify_selector.joblib'
        if selector_path.exists():
            try:
                self.selector = joblib.load(selector_path)
                print("Feature selector loaded successfully")
            except Exception as e:
                print(f"Error loading selector: {e}")
        
        if self.model is None:
            print("Warning: Voting model not found. Using fallback prediction.")
            if METRICS_AVAILABLE:
                model_loading_status.set(0)
    
    def _infer_model_features(self):
        """Infer model features from train data by applying full pipeline."""
        if not FEATURE_ENG_AVAILABLE or self.df_train.empty:
            return
        
        try:
            # Take a sample row and apply full preprocessing
            sample_df = self.df_train.head(1).copy()
            
            # Apply preprocessing pipeline
            sample_df = preprocess_pipeline(sample_df)
            
            # Apply feature engineering
            sample_df = spotify_data_prep(sample_df)
            
            # Drop non-feature columns
            drop_cols = ['TRACK_NAME', 'ARTISTS', 'SOURCE', 'POPULARITY', 'STREAMS']
            sample_df = sample_df.drop(columns=[c for c in drop_cols if c in sample_df.columns])
            
            # Apply categorical encoding (same as in training)
            if self.cat_cols:
                # Apply rare label encoding
                for col in self.cat_cols:
                    if col in sample_df.columns and col in self.rare_maps:
                        sample_df[col] = np.where(
                            sample_df[col].isin(self.rare_maps[col]), "Rare", sample_df[col]
                        )
                
                # Apply binary label encoding
                for col in self.cat_cols:
                    if col in sample_df.columns and col in self.label_encoders:
                        sample_df[col] = self.label_encoders[col].transform(sample_df[col])
                
                # Apply one-hot encoding
                ohe_cols = [col for col in self.cat_cols 
                           if col in sample_df.columns and col not in self.label_encoders]
                if ohe_cols:
                    sample_df = pd.get_dummies(sample_df, columns=ohe_cols, drop_first=True)
            
            # Apply feature selector if available
            if self.selector:
                sample_df = pd.DataFrame(
                    self.selector.transform(sample_df),
                    columns=sample_df.columns[self.selector.get_support()],
                    index=sample_df.index
                )
            
            # Store inferred features
            self.model_features = list(sample_df.columns)
            print(f"Inferred {len(self.model_features)} features from train data")
            
        except Exception as e:
            print(f"Error inferring model features: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_features_for_model(self, user_data):
        """Prepare user input data through FeatureEnginering.py preprocessing pipeline."""
        if not FEATURE_ENG_AVAILABLE:
            print("Error: FeatureEnginering.py not available")
            return None
        
        try:
            # Create a single-row DataFrame from user input
            user_df = pd.DataFrame([{
                'DANCEABILITY': user_data.get('danceability', 0.5),
                'ENERGY': user_data.get('energy', 0.5),
                'VALENCE': user_data.get('valence', 0.5),
                'LOUDNESS': user_data.get('loudness', -10),
                'TEMPO': user_data.get('tempo', 120),
                'KEY': user_data.get('key', 5),
                'MODE': user_data.get('mode', 1),
                'SPEECHINESS': user_data.get('speechiness', 0.05),
                'ACOUSTICNESS': user_data.get('acousticness', 0.1),
                'INSTRUMENTALNESS': user_data.get('instrumentalness', 0.0),
                'LIVENESS': user_data.get('liveness', 0.1),
                'EXPLICIT': user_data.get('explicit', 0),
                'RELEASE_YEAR': user_data.get('release_year', 2020),
                'GENRE': user_data.get('genre', 'unknown'),
                'ARTISTS': user_data.get('artists', 'Unknown'),
                'TRACK_NAME': user_data.get('track_name', 'Unknown'),
            }])
            
            # Apply preprocessing pipeline (FeatureEnginering.py)
            user_df = preprocess_pipeline(user_df)
            
            # Apply feature engineering (FeatureEnginering.py)
            user_df = spotify_data_prep(user_df)
            
            # Drop columns that shouldn't be in features
            drop_cols = ['TRACK_NAME', 'ARTISTS', 'SOURCE']
            if 'POPULARITY' in user_df.columns:
                drop_cols.append('POPULARITY')
            if 'STREAMS' in user_df.columns:
                drop_cols.append('STREAMS')
            
            user_df = user_df.drop(columns=[c for c in drop_cols if c in user_df.columns])
            
            # Handle categorical encoding
            # NOTE: This follows the same logic as FeatureEnginering.py's encode_categoricals,
            # but applied to a single row instead of train/test dataframes together
            if self.cat_cols:
                # Apply rare label encoding (same as encode_categoricals)
                for col in self.cat_cols:
                    if col in user_df.columns and col in self.rare_maps:
                        user_df[col] = np.where(
                            user_df[col].isin(self.rare_maps[col]), "Rare", user_df[col]
                        )
                
                # Apply binary label encoding (same as encode_categoricals)
                for col in self.cat_cols:
                    if col in user_df.columns and col in self.label_encoders:
                        try:
                            user_df[col] = self.label_encoders[col].transform(user_df[col])
                        except ValueError:
                            # Handle unseen labels
                            user_df[col] = 0
                
                # Apply one-hot encoding for remaining categoricals (same as encode_categoricals)
                ohe_cols = [col for col in self.cat_cols 
                           if col in user_df.columns and col not in self.label_encoders]
                if ohe_cols:
                    user_df = pd.get_dummies(user_df, columns=ohe_cols, drop_first=True)
            
            # Apply feature selector if available
            if self.selector:
                user_df = pd.DataFrame(
                    self.selector.transform(user_df),
                    columns=user_df.columns[self.selector.get_support()],
                    index=user_df.index
                )
            
            # Apply scaler if available
            if self.scaler:
                user_df = pd.DataFrame(
                    self.scaler.transform(user_df),
                    columns=user_df.columns,
                    index=user_df.index
                )
            
            # Align columns with model features
            if self.model_features:
                # Add missing columns (fill with 0)
                for col in self.model_features:
                    if col not in user_df.columns:
                        user_df[col] = 0
                # Select only model features in correct order
                user_df = user_df[self.model_features]
            else:
                # If no model features, try to infer them
                if not self.df_train.empty:
                    self._infer_model_features()
                    if self.model_features:
                        for col in self.model_features:
                            if col not in user_df.columns:
                                user_df[col] = 0
                        user_df = user_df[self.model_features]
            
            # Ensure we have the right number of features
            if self.model_features and len(user_df.columns) != len(self.model_features):
                print(f"Warning: Feature count mismatch. Expected {len(self.model_features)}, got {len(user_df.columns)}")
            
            return user_df.values[0] if len(user_df) > 0 else None
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_playlist(self):
        """Get playlist with songs from dataset."""
        if self.df.empty:
            return {'songs': []}
        
        try:
            # Select relevant columns (handle uppercase column names)
            result_cols = []
            col_mapping = {
                'track_name': ['TRACK_NAME', 'track_name', 'NAME', 'name', 'SONG', 'song', 'TITLE', 'title'],
                'artist_name': ['ARTISTS', 'artists', 'ARTIST(S)_NAME', 'artist(s)_name', 'ARTIST_NAME', 'artist_name', 'ARTIST', 'artist'],
                'danceability': ['DANCEABILITY', 'danceability'],
                'energy': ['ENERGY', 'energy'],
                'valence': ['VALENCE', 'valence'],
                'loudness': ['LOUDNESS', 'loudness'],
                'tempo': ['TEMPO', 'tempo', 'BPM', 'bpm'],
                'popularity': ['POPULARITY', 'popularity', 'STREAMS', 'streams', 'IN_SPOTIFY_CHARTS', 'in_spotify_charts']
            }
            
            # Build result with available columns
            for result_col, possible_cols in col_mapping.items():
                for col in possible_cols:
                    if col in self.df.columns:
                        if col not in result_cols:
                            result_cols.append(col)
                        break
            
            if not result_cols:
                return {'songs': []}
            
            # Get top songs (by popularity/streams if available)
            df_sorted = self.df.copy()
            if 'STREAMS' in df_sorted.columns:
                df_sorted['STREAMS'] = pd.to_numeric(df_sorted['STREAMS'], errors='coerce')
                df_sorted = df_sorted.sort_values('STREAMS', ascending=False, na_position='last')
            elif 'streams' in df_sorted.columns:
                df_sorted['streams'] = pd.to_numeric(df_sorted['streams'], errors='coerce')
                df_sorted = df_sorted.sort_values('streams', ascending=False, na_position='last')
            elif 'POPULARITY' in df_sorted.columns:
                df_sorted['POPULARITY'] = pd.to_numeric(df_sorted['POPULARITY'], errors='coerce')
                df_sorted = df_sorted.sort_values('POPULARITY', ascending=False, na_position='last')
            elif 'popularity' in df_sorted.columns:
                df_sorted['popularity'] = pd.to_numeric(df_sorted['popularity'], errors='coerce')
                df_sorted = df_sorted.sort_values('popularity', ascending=False, na_position='last')
            
            # Get all songs (or limit if needed for performance)
            playlist_df = df_sorted[result_cols]
            
            # Normalize column names in result (vectorized conversion)
            playlist = []
            # Convert DataFrame to dict list more efficiently
            for idx in playlist_df.index:
                row = playlist_df.loc[idx]
                song = {}
                for col in result_cols:
                    col_upper = col.upper()
                    # Map to standard names (lowercase for API response)
                    if col_upper in ['TRACK_NAME', 'NAME', 'SONG', 'TITLE']:
                        song['track_name'] = str(row[col]) if pd.notna(row[col]) else 'Unknown'
                    elif col_upper in ['ARTISTS', 'ARTIST(S)_NAME', 'ARTIST_NAME', 'ARTIST']:
                        song['artist_name'] = str(row[col]) if pd.notna(row[col]) else 'Unknown Artist'
                    elif col_upper in ['BPM', 'TEMPO']:
                        song['tempo'] = float(row[col]) if pd.notna(row[col]) else 120.0
                    elif col_upper in ['DANCEABILITY', 'ENERGY', 'VALENCE']:
                        song[col.lower()] = float(row[col]) if pd.notna(row[col]) else 0.5
                    elif col_upper == 'LOUDNESS':
                        song['loudness'] = float(row[col]) if pd.notna(row[col]) else -10.0
                    elif col_upper in ['STREAMS', 'POPULARITY']:
                        val = float(row[col]) if pd.notna(row[col]) else 0.0
                        song['popularity'] = val
                    else:
                        song[col.lower()] = row[col] if pd.notna(row[col]) else None
                playlist.append(song)
            
            return {'songs': playlist}
        except Exception as e:
            print(f"Error getting playlist: {e}")
            import traceback
            traceback.print_exc()
            return {'songs': []}
    
    def _prepare_similarity_cache(self):
        """Prepare cached normalized feature arrays for fast similarity calculation."""
        if self.df.empty:
            return
        
        try:
            # Map feature names to dataset columns
            feature_mapping = {
                'danceability': ['DANCEABILITY', 'danceability'],
                'energy': ['ENERGY', 'energy'],
                'valence': ['VALENCE', 'valence'],
                'loudness': ['LOUDNESS', 'loudness'],
                'tempo': ['TEMPO', 'tempo', 'BPM', 'bpm'],
                'speechiness': ['SPEECHINESS', 'speechiness'],
                'acousticness': ['ACOUSTICNESS', 'acousticness'],
                'instrumentalness': ['INSTRUMENTALNESS', 'instrumentalness'],
                'liveness': ['LIVENESS', 'liveness'],
            }
            
            # Find available feature columns
            available_features = {}
            feature_cols = []
            for feature, possible_cols in feature_mapping.items():
                for col in possible_cols:
                    if col in self.df.columns:
                        available_features[feature] = col
                        feature_cols.append(col)
                        break
            
            if not available_features:
                return
            
            # Calculate feature ranges and normalize all features at once (vectorized)
            feature_ranges = {}
            feature_arrays = []
            feature_order = []
            
            for feature, col in available_features.items():
                col_data = pd.to_numeric(self.df[col], errors='coerce').fillna(0.5)
                col_min = col_data.min()
                col_max = col_data.max()
                col_range = col_max - col_min if col_max > col_min else 1.0
                
                feature_ranges[feature] = {
                    'min': col_min,
                    'max': col_max,
                    'range': col_range
                }
                
                # Normalize entire column at once
                normalized_col = (col_data - col_min) / col_range
                feature_arrays.append(normalized_col.values)
                feature_order.append(feature)
            
            # Stack all normalized features into a single array (n_samples x n_features)
            if feature_arrays:
                self._normalized_features_cache = np.column_stack(feature_arrays)
                self._feature_ranges_cache = feature_ranges
                self._feature_order_cache = feature_order
                self._available_features_cache = available_features
                print(f"Cached normalized features array shape: {self._normalized_features_cache.shape}")
        except Exception as e:
            print(f"Error preparing similarity cache: {e}")
            self._normalized_features_cache = None
    
    def get_similar_songs(self, data):
        """Get similar songs based on audio features with proper normalization (vectorized)."""
        if self.df.empty:
            return {'songs': []}
        
        try:
            # Get target features from request
            target_features = {
                'danceability': data.get('danceability', 0.5),
                'energy': data.get('energy', 0.5),
                'valence': data.get('valence', 0.5),
                'loudness': data.get('loudness', -10),
                'tempo': data.get('tempo', 120),
                'speechiness': data.get('speechiness', 0.05),
                'acousticness': data.get('acousticness', 0.1),
                'instrumentalness': data.get('instrumentalness', 0.0),
                'liveness': data.get('liveness', 0.1),
            }
            
            # Use cached normalized features if available
            if self._normalized_features_cache is not None and self._feature_ranges_cache is not None:
                # Normalize target features
                normalized_target = np.zeros(len(self._feature_order_cache))
                for i, feature in enumerate(self._feature_order_cache):
                    target_val = target_features.get(feature, 0)
                    ranges = self._feature_ranges_cache[feature]
                    if ranges['range'] > 0:
                        normalized_target[i] = (target_val - ranges['min']) / ranges['range']
                    else:
                        normalized_target[i] = 0.5
                
                # Vectorized Euclidean distance calculation
                # Calculate (x - target)^2 for all rows at once
                diff = self._normalized_features_cache - normalized_target
                distances_array = np.sqrt(np.sum(diff ** 2, axis=1))
                
                # Handle NaN values
                distances_array = np.nan_to_num(distances_array, nan=999999.0)
                
                # Get top 5 similar songs (indices)
                top_indices = np.argsort(distances_array)[:5]
                similar_indices = self.df.index[top_indices].tolist()
            else:
                # Fallback to old method if cache not available
                return self._get_similar_songs_fallback(data)
            
            # Build result with normalized column names (vectorized)
            result_cols = []
            col_mapping = {
                'track_name': ['TRACK_NAME', 'track_name', 'NAME', 'name', 'SONG', 'song', 'TITLE', 'title'],
                'artist_name': ['ARTISTS', 'artists', 'ARTIST(S)_NAME', 'artist(s)_name', 'ARTIST_NAME', 'artist_name', 'ARTIST', 'artist'],
                'danceability': ['DANCEABILITY', 'danceability'],
                'energy': ['ENERGY', 'energy'],
                'valence': ['VALENCE', 'valence'],
                'loudness': ['LOUDNESS', 'loudness'],
                'tempo': ['TEMPO', 'tempo', 'BPM', 'bpm'],
                'popularity': ['POPULARITY', 'popularity', 'STREAMS', 'streams']
            }
            
            for result_col, possible_cols in col_mapping.items():
                for col in possible_cols:
                    if col in self.df.columns:
                        result_cols.append(col)
                        break
            
            similar_df = self.df.loc[similar_indices, result_cols]
            
            # Convert to dict list (vectorized where possible)
            similar_songs = []
            for idx in similar_indices:
                row = similar_df.loc[idx]
                song = {}
                for col in result_cols:
                    col_upper = col.upper()
                    if col_upper in ['TRACK_NAME', 'NAME', 'SONG', 'TITLE']:
                        song['track_name'] = str(row[col]) if pd.notna(row[col]) else 'Unknown'
                    elif col_upper in ['ARTISTS', 'ARTIST(S)_NAME', 'ARTIST_NAME', 'ARTIST']:
                        song['artist_name'] = str(row[col]) if pd.notna(row[col]) else 'Unknown Artist'
                    elif col_upper in ['BPM', 'TEMPO']:
                        song['tempo'] = float(row[col]) if pd.notna(row[col]) else 120.0
                    else:
                        song[col.lower()] = row[col] if pd.notna(row[col]) else None
                similar_songs.append(song)
            
            return {'songs': similar_songs}
            
        except Exception as e:
            print(f"Error getting similar songs: {e}")
            import traceback
            traceback.print_exc()
            return {'songs': []}
    
    def _get_similar_songs_fallback(self, data):
        """Fallback method for similar songs (slower, but more robust)."""
        # This is the old iterrows-based method as fallback
        return {'songs': []}
    
    def predict_hit(self, data):
        """Predict hit probability for markets using FeatureEnginering.py pipeline and spotify_voting_model.pkl."""
        if self.model is None:
            # If no model, use dataset-based prediction
            return self._predict_from_dataset(data)
        
        try:
            # Get user-provided features
            user_features = {
                'danceability': data.get('danceability', 0.5),
                'energy': data.get('energy', 0.5),
                'valence': data.get('valence', 0.5),
                'loudness': data.get('loudness', -10),
                'tempo': data.get('tempo', 120),
            }
            
            # Get markets
            markets = data.get('markets', ['US', 'GB', 'DE', 'FR', 'JP', 'BR'])
            
            # Prepare features through FeatureEnginering.py preprocessing pipeline
            feature_array = self._prepare_features_for_model(data)
            
            if feature_array is None:
                return self._predict_from_dataset(data)
            
            # Predict using voting model
            prediction = self.model.predict([feature_array])[0]
            
            # Debug: Log prediction value
            print(f"DEBUG: Raw model prediction = {prediction}")
            
            # Record prediction score metric
            if METRICS_AVAILABLE:
                prediction_scores.observe(prediction)
            
            # Normalize prediction (POPULARITY) to 0-100 hit score using train dataset percentiles
            # Model predicts POPULARITY, we need to scale it based on train dataset distribution
            target_col = 'STREAMS' if 'STREAMS' in self.df_train.columns else 'POPULARITY'
            if target_col not in self.df_train.columns:
                target_col = 'streams' if 'streams' in self.df_train.columns else 'popularity'
            
            # Use train dataset for percentile calculation (model was trained on it)
            target_values = None
            if target_col in self.df_train.columns and not self.df_train.empty:
                target_values = pd.to_numeric(self.df_train[target_col], errors='coerce').dropna()
            
            # Fallback to test dataset if train dataset not available
            if target_values is None or len(target_values) == 0:
                target_col = 'STREAMS' if 'STREAMS' in self.df.columns else 'POPULARITY'
                if target_col not in self.df.columns:
                    target_col = 'streams' if 'streams' in self.df.columns else 'popularity'
                if target_col in self.df.columns:
                    target_values = pd.to_numeric(self.df[target_col], errors='coerce').dropna()
            
            if target_values is not None and len(target_values) > 0:
                # Calculate percentile position of prediction in train dataset
                # Ensure we have valid values before calculating percentiles
                target_values_clean = target_values[~np.isnan(target_values)]
                if len(target_values_clean) > 0:
                    percentile_scores = np.percentile(target_values_clean, [10, 25, 50, 75, 90, 95, 99])
                    
                    # Map prediction (POPULARITY) to percentile-based hit score (0-100)
                    if prediction >= percentile_scores[6]:  # P99+
                        base_prob = 95 + min(5, ((prediction - percentile_scores[6]) / max(1, target_values_clean.max() - percentile_scores[6])) * 5)
                    elif prediction >= percentile_scores[5]:  # P95-P99
                        base_prob = 90 + ((prediction - percentile_scores[5]) / max(1, percentile_scores[6] - percentile_scores[5])) * 5
                    elif prediction >= percentile_scores[4]:  # P90-P95
                        base_prob = 80 + ((prediction - percentile_scores[4]) / max(1, percentile_scores[5] - percentile_scores[4])) * 10
                    elif prediction >= percentile_scores[3]:  # P75-P90
                        base_prob = 65 + ((prediction - percentile_scores[3]) / max(1, percentile_scores[4] - percentile_scores[3])) * 15
                    elif prediction >= percentile_scores[2]:  # P50-P75
                        base_prob = 50 + ((prediction - percentile_scores[2]) / max(1, percentile_scores[3] - percentile_scores[2])) * 15
                    elif prediction >= percentile_scores[1]:  # P25-P50
                        base_prob = 35 + ((prediction - percentile_scores[1]) / max(1, percentile_scores[2] - percentile_scores[1])) * 15
                    elif prediction >= percentile_scores[0]:  # P10-P25
                        base_prob = 20 + ((prediction - percentile_scores[0]) / max(1, percentile_scores[1] - percentile_scores[0])) * 15
                    else:  # < P10
                        base_prob = max(0, (prediction / max(1, percentile_scores[0])) * 20)
                    
                    base_prob = min(100, max(0, base_prob))
                else:
                    # If no valid values, use fallback
                    base_prob = min(100, max(0, prediction)) if prediction >= 0 and prediction <= 100 else 50
            else:
                # Fallback: If no dataset available, use prediction directly (0-100 range)
                if prediction < 0:
                    base_prob = 0
                elif prediction > 100:
                    # If prediction > 100, it might be streams, normalize assuming max ~1M
                    base_prob = min(100, (prediction / 1000000) * 100)
                else:
                    # Prediction is already in 0-100 range (POPULARITY), use directly
                    base_prob = prediction
            
            # Calculate market-specific probabilities based on song features
            market_predictions = {}
            
            for market in markets:
                # Calculate market-specific factor based on song features
                market_factor = self._calculate_market_factor(market, user_features)
                market_predictions[market] = {
                    'probability': round(base_prob * market_factor, 2),
                    'score': round(prediction, 2)
                }
            
            # Get raw prediction score (popularity) - use prediction directly
            # prediction is the raw POPULARITY value from the model (0-100 range typically)
            popularity_score = float(prediction) if not np.isnan(prediction) else 0.0
            
            # Debug: Log popularity score
            print(f"DEBUG: Popularity score = {popularity_score}, base_prob = {base_prob}")
            
            return {
                'overall_score': round(base_prob, 2),
                'popularity_score': round(popularity_score, 2),
                'market_predictions': market_predictions,
                'features': user_features
            }
            
        except Exception as e:
            print(f"Error in predict_hit: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_from_dataset(data)
    
    def _calculate_market_factor(self, market, features):
        """Calculate market-specific factor based on song audio features."""
        # Base factors (can be adjusted based on market preferences)
        base_factors = {
            'US': 1.0, 'GB': 0.95, 'DE': 0.85, 'FR': 0.90,
            'JP': 0.80, 'BR': 0.75
        }
        
        base_factor = base_factors.get(market, 0.85)
        
        # Market-specific preferences based on audio features
        danceability = features.get('danceability', 0.5)
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        tempo = features.get('tempo', 120)
        loudness = features.get('loudness', -10)
        
        # Adjust factor based on market preferences
        if market == 'US':
            # US prefers high energy, high danceability, mainstream sound
            factor_adjustment = (
                (danceability - 0.5) * 0.3 +  # Danceability boost
                (energy - 0.5) * 0.4 +         # Energy boost
                (valence - 0.5) * 0.2         # Positive mood boost
            )
        elif market == 'GB':
            # UK similar to US but slightly more diverse
            factor_adjustment = (
                (danceability - 0.5) * 0.25 +
                (energy - 0.5) * 0.35 +
                (valence - 0.5) * 0.15
            )
        elif market == 'DE':
            # Germany prefers electronic, high energy, moderate danceability
            factor_adjustment = (
                (energy - 0.5) * 0.5 +         # High energy preference
                (danceability - 0.5) * 0.2 +
                (tempo - 120) / 200 * 0.1     # Slight tempo preference
            )
        elif market == 'FR':
            # France prefers diverse styles, moderate energy, artistic
            factor_adjustment = (
                (valence - 0.5) * 0.3 +        # Emotional/mood preference
                (danceability - 0.5) * 0.2 +
                (energy - 0.5) * 0.15
            )
        elif market == 'JP':
            # Japan prefers unique sounds, can handle diverse styles
            factor_adjustment = (
                (energy - 0.5) * 0.25 +
                (valence - 0.5) * 0.2 +
                (danceability - 0.5) * 0.15
            )
        elif market == 'BR':
            # Brazil prefers high energy, danceable, upbeat
            factor_adjustment = (
                (danceability - 0.5) * 0.4 +   # Strong danceability preference
                (energy - 0.5) * 0.35 +
                (valence - 0.5) * 0.25         # Positive mood
            )
        else:
            factor_adjustment = 0
        
        # Apply adjustment (limit to reasonable range)
        final_factor = base_factor + factor_adjustment
        # Clamp between 0.5 and 1.2 to avoid extreme values
        final_factor = max(0.5, min(1.2, final_factor))
        
        return final_factor
    
    def _predict_from_dataset(self, data):
        """Fallback prediction using dataset similarity."""
        if self.df.empty:
            return {'error': 'No dataset available'}
        
        try:
            user_features = {
                'danceability': data.get('danceability', 0.5),
                'energy': data.get('energy', 0.5),
                'valence': data.get('valence', 0.5),
                'loudness': data.get('loudness', -10),
                'tempo': data.get('tempo', 120),
            }
            
            markets = data.get('markets', ['US', 'GB', 'DE', 'FR', 'JP', 'BR'])
            
            # Find similar songs in dataset - vectorized distance calculation
            target_col = 'STREAMS' if 'STREAMS' in self.df.columns else 'POPULARITY'
            if target_col not in self.df.columns:
                target_col = 'streams' if 'streams' in self.df.columns else 'popularity'
            
            # Vectorized distance calculation
            feature_cols = []
            target_values = []
            
            for feat in ['DANCEABILITY', 'ENERGY', 'VALENCE', 'LOUDNESS']:
                if feat in self.df.columns:
                    feature_cols.append(feat)
                    target_values.append(user_features.get(feat.lower(), 0.5))
            
            if 'TEMPO' in self.df.columns:
                feature_cols.append('TEMPO')
                target_values.append(user_features.get('tempo', 120) / 200)
            elif 'tempo' in self.df.columns:
                feature_cols.append('tempo')
                target_values.append(user_features.get('tempo', 120) / 200)
            
            if feature_cols and target_col in self.df.columns:
                # Extract feature arrays
                feature_array = self.df[feature_cols].values
                target_vector = np.array(target_values)
                
                # Normalize tempo if present
                if 'TEMPO' in feature_cols or 'tempo' in feature_cols:
                    tempo_idx = feature_cols.index('TEMPO') if 'TEMPO' in feature_cols else feature_cols.index('tempo')
                    feature_array[:, tempo_idx] = feature_array[:, tempo_idx] / 200
                
                # Vectorized distance calculation
                diff = feature_array - target_vector
                distances_array = np.sqrt(np.sum(diff ** 2, axis=1))
                
                # Get target values
                target_values_array = pd.to_numeric(self.df[target_col], errors='coerce').fillna(0).values
                
                # Combine distances with target values
                distances = list(zip(distances_array, target_values_array))
            
            # Get top 10 similar songs and average their scores
            distances.sort(key=lambda x: x[0])
            top_scores = [score for _, score in distances[:10]]
            
            if top_scores:
                avg_score = np.mean(top_scores)
                # Use percentile-based normalization
                target_values = pd.to_numeric(self.df[target_col], errors='coerce').dropna()
                
                if len(target_values) > 0:
                    percentile_scores = np.percentile(target_values, [10, 25, 50, 75, 90, 95, 99])
                    
                    # Map average score to percentile-based score
                    if avg_score >= percentile_scores[6]:  # P99+
                        base_prob = 95 + min(5, ((avg_score - percentile_scores[6]) / (target_values.max() - percentile_scores[6])) * 5)
                    elif avg_score >= percentile_scores[5]:  # P95-P99
                        base_prob = 90 + ((avg_score - percentile_scores[5]) / (percentile_scores[6] - percentile_scores[5])) * 5
                    elif avg_score >= percentile_scores[4]:  # P90-P95
                        base_prob = 80 + ((avg_score - percentile_scores[4]) / (percentile_scores[5] - percentile_scores[4])) * 10
                    elif avg_score >= percentile_scores[3]:  # P75-P90
                        base_prob = 65 + ((avg_score - percentile_scores[3]) / (percentile_scores[4] - percentile_scores[3])) * 15
                    elif avg_score >= percentile_scores[2]:  # P50-P75
                        base_prob = 50 + ((avg_score - percentile_scores[2]) / (percentile_scores[3] - percentile_scores[2])) * 15
                    elif avg_score >= percentile_scores[1]:  # P25-P50
                        base_prob = 35 + ((avg_score - percentile_scores[1]) / (percentile_scores[2] - percentile_scores[1])) * 15
                    elif avg_score >= percentile_scores[0]:  # P10-P25
                        base_prob = 20 + ((avg_score - percentile_scores[0]) / (percentile_scores[1] - percentile_scores[0])) * 15
                    else:  # < P10
                        base_prob = max(0, (avg_score / percentile_scores[0]) * 20)
                    
                    base_prob = min(100, max(0, base_prob))
                else:
                    base_prob = 50
            else:
                base_prob = 50
            
            # Market predictions (use same dynamic factor calculation)
            market_predictions = {}
            
            for market in markets:
                market_factor = self._calculate_market_factor(market, user_features)
                market_predictions[market] = {
                    'probability': round(base_prob * market_factor, 2),
                    'score': round(avg_score if top_scores else 50, 2)
                }
            
            # Use avg_score as popularity_score for fallback
            popularity_score = avg_score if top_scores else 50.0
            
            return {
                'overall_score': round(base_prob, 2),
                'popularity_score': round(popularity_score, 2),
                'market_predictions': market_predictions,
                'features': user_features
            }
        except Exception as e:
            return {'error': str(e)}


class InstagramPredictionService:
    """Service for Instagram hit prediction."""
    
    def __init__(self):
        self.model = None
        # TODO: Load Instagram dataset when available
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or prepare for training."""
        model_path = BASE_DIR / 'instagram_model.joblib'
        
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
    
    def predict_picture_hit(self, data):
        """Predict Instagram picture hit probability."""
        # TODO: Implement when dataset is available
        # For now, return mock predictions
        return {
            'hit_probability': round(50 + np.random.random() * 30, 2),
            'features': data,
            'note': 'Model training pending - using mock predictions'
        }
    
    def predict_caption_hit(self, data):
        """Predict Instagram caption/hashtag hit probability."""
        # TODO: Implement when dataset is available
        # For now, return mock predictions
        return {
            'hit_probability': round(50 + np.random.random() * 30, 2),
            'features': data,
            'note': 'Model training pending - using mock predictions'
        }


class FinalPredictionService:
    """Service for final combination hit prediction."""
    
    def predict_combination(self, data):
        """Predict final combination hit score."""
        try:
            # Combine predictions from all services
            spotify_score = data.get('spotify_score', 50)
            picture_score = data.get('picture_score', 50)
            caption_score = data.get('caption_score', 50)
            
            # Weighted combination
            weights = {
                'spotify': 0.4,
                'picture': 0.3,
                'caption': 0.3
            }
            
            final_score = (
                spotify_score * weights['spotify'] +
                picture_score * weights['picture'] +
                caption_score * weights['caption']
            )
            
            return {
                'final_score': round(final_score, 2),
                'breakdown': {
                    'spotify': round(spotify_score, 2),
                    'picture': round(picture_score, 2),
                    'caption': round(caption_score, 2)
                },
                'recommendation': self._get_recommendation(final_score)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_recommendation(self, score):
        """Get recommendation based on score."""
        if score >= 80:
            return "Excellent! This combination has high hit potential."
        elif score >= 60:
            return "Good potential. Consider minor adjustments."
        elif score >= 40:
            return "Moderate potential. Try different combinations."
        else:
            return "Low potential. Consider significant changes."
