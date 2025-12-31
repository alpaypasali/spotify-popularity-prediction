"""
Model başarısını değerlendirmek için script.
Kullanım: python evaluate_model.py
"""
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Şimdi gerekli modülleri import et
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Services modülünü import et
USE_SERVICE = False
import joblib

# Django modülünü dene (opsiyonel)
try:
    import sys
    from pathlib import Path
    # Proje kök dizinini path'e ekle
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from projects.socialMediaHit.services import SpotifyPredictionService
    USE_SERVICE = True
except ImportError:
    pass

def evaluate_model():
    """Mevcut modeli yükleyip başarısını değerlendir."""
    global USE_SERVICE
    
    # Windows konsol encoding sorununu çöz
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("Model Basari Degerlendirmesi\n")
    print("=" * 60)
    
    if USE_SERVICE:
        # Services kullanarak (Django varsa)
        try:
            service = SpotifyPredictionService()
            model = service.model
            df = service.df
            
            if model is None:
                print("Model bulunamadi! Model egitiliyor...")
                service.train_model()
                model = service.model
                if model is None:
                    print("Model egitilemedi!")
                    return
            
            if df is None or df.empty:
                print("Dataset yuklenemedi!")
                return
                
            print(f"Model yuklendi: {type(model).__name__}")
            print(f"Dataset yuklendi: {len(df)} satir, {len(df.columns)} sutun\n")
            
        except Exception as e:
            print(f"Service yuklenirken hata: {e}")
            USE_SERVICE = False
    
    if not USE_SERVICE:
        # Doğrudan model yükleme (Django yoksa)
        model_path = BASE_DIR / 'spotify_model.joblib'
        if not model_path.exists():
            print("Model dosyasi bulunamadi!")
            return
        
        try:
            model = joblib.load(model_path)
            print(f"Model yuklendi: {type(model).__name__}")
        except Exception as e:
            print(f"Model yuklenirken hata: {e}")
            return
        
        # Dataset'i yükle
        dataset_paths = [
            BASE_DIR / 'pythonEDA' / 'spotify_emotion_test.csv',
            BASE_DIR / 'spotify_songs.csv',
            BASE_DIR / 'light_spotify_dataset.csv',
            BASE_DIR / 'Most_Streamed_Spotify_Songs_2024.csv',
        ]
        
        df = None
        for path in dataset_paths:
            if path.exists():
                try:
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            df = pd.read_csv(path, encoding=encoding)
                            print(f"Dataset yuklendi: {path.name} ({encoding})")
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is not None:
                        break
                except Exception as e:
                    continue
        
        if df is None or df.empty:
            print("Dataset yuklenemedi!")
            return
        
        print(f"📁 Dataset boyutu: {df.shape[0]} satır, {df.shape[1]} sütun\n")
        
        # Kolon normalizasyonu (services.py'deki gibi)
        column_mapping = {
            'track_name': ['track_name', 'name', 'song', 'title'],
            'artist_name': ['artist(s)_name', 'artist_name', 'artist', 'artists'],
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
            'popularity': ['popularity', 'in_spotify_charts']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for col in df.columns:
                if col in possible_names and standard_name not in df.columns:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # Yüzde kolonlarını normalize et
        for col in ['danceability', 'energy', 'valence', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness']:
            if col in df.columns:
                if df[col].max() > 1:
                    df[col] = df[col] / 100.0
        
        # Numeric kolonları dönüştür
        for col in ['streams', 'tempo', 'bpm', 'loudness', 'key', 'mode']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Feature kolonlarını bul
    feature_mapping = {
        'danceability': ['danceability'],
        'energy': ['energy'],
        'valence': ['valence'],
        'loudness': ['loudness'],
        'tempo': ['tempo', 'bpm'],
        'key': ['key'],
        'mode': ['mode'],
        'speechiness': ['speechiness'],
        'acousticness': ['acousticness'],
        'instrumentalness': ['instrumentalness'],
        'liveness': ['liveness']
    }
    
    available_cols = []
    for standard_name, possible_cols in feature_mapping.items():
        for col in possible_cols:
            if col in df.columns:
                available_cols.append(col)
                break
    
    # Target kolonunu bul
    target_col = None
    for col in ['streams', 'popularity', 'in_spotify_charts']:
        if col in df.columns:
            target_col = col
            break
    
    if not available_cols or target_col is None:
        print("Gerekli kolonlar bulunamadi!")
        return
    
    print(f"Kullanilan ozellikler: {available_cols}")
    print(f"Hedef degisken: {target_col}\n")
    
    # Veriyi hazırla
    X = df[available_cols].copy()
    
    # Eksik değerleri doldur
    for col in available_cols:
        if X[col].isna().any():
            if col == 'loudness':
                X[col] = X[col].fillna(-10)
            elif col in ['tempo', 'bpm']:
                X[col] = X[col].fillna(120)
            elif col in ['key', 'mode']:
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(0.5)
    
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    
    # Geçerli verileri filtrele
    valid_mask = (y > 0) & (y.notna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Outlier handling (model eğitimindekiyle aynı)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (y >= lower_bound) & (y <= upper_bound)
    X = X[outlier_mask]
    y = y[outlier_mask]
    
    # Feature engineering (model eğitimindekiyle aynı - TAM VERSIYON)
    original_feature_count = len(X.columns)
    X_enhanced = X.copy()
    
    # 1. Interaction features
    if 'danceability' in X.columns and 'energy' in X.columns:
        X_enhanced['danceability_energy'] = X['danceability'] * X['energy']
    if 'valence' in X.columns and 'energy' in X.columns:
        X_enhanced['valence_energy'] = X['valence'] * X['energy']
    if 'danceability' in X.columns and 'valence' in X.columns:
        X_enhanced['danceability_valence'] = X['danceability'] * X['valence']
    if 'energy' in X.columns and 'loudness' in X.columns:
        X_enhanced['energy_loudness'] = X['energy'] * (X['loudness'] + 60) / 60  # Normalize loudness
    if 'danceability' in X.columns and 'tempo' in X.columns:
        X_enhanced['danceability_tempo'] = X['danceability'] * (X['tempo'] / 200)  # Normalize tempo
    elif 'danceability' in X.columns and 'bpm' in X.columns:
        X_enhanced['danceability_tempo'] = X['danceability'] * (X['bpm'] / 200)
    
    # 2. Ratio features
    if 'energy' in X.columns and 'acousticness' in X.columns:
        X_enhanced['energy_acoustic_ratio'] = X['energy'] / (X['acousticness'] + 0.001)
    if 'danceability' in X.columns and 'speechiness' in X.columns:
        X_enhanced['danceability_speech_ratio'] = X['danceability'] / (X['speechiness'] + 0.001)
    if 'valence' in X.columns and 'energy' in X.columns:
        X_enhanced['valence_energy_ratio'] = X['valence'] / (X['energy'] + 0.001)
    
    # 3. Sum features
    if 'danceability' in X.columns and 'energy' in X.columns and 'valence' in X.columns:
        X_enhanced['dance_energy_valence_sum'] = X['danceability'] + X['energy'] + X['valence']
    if 'speechiness' in X.columns and 'acousticness' in X.columns:
        X_enhanced['speech_acoustic_sum'] = X['speechiness'] + X['acousticness']
    
    # 4. Categorical features
    if 'tempo' in X.columns:
        X_enhanced['tempo_category'] = pd.cut(
            X['tempo'], 
            bins=[0, 90, 120, 150, 200], 
            labels=[0, 1, 2, 3]
        ).astype(float)
    elif 'bpm' in X.columns:
        X_enhanced['tempo_category'] = pd.cut(
            X['bpm'], 
            bins=[0, 90, 120, 150, 200], 
            labels=[0, 1, 2, 3]
        ).astype(float)
    
    if 'energy' in X.columns:
        X_enhanced['energy_category'] = pd.cut(
            X['energy'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=[0, 1, 2]
        ).astype(float)
    
    if 'danceability' in X.columns:
        X_enhanced['danceability_category'] = pd.cut(
            X['danceability'], 
            bins=[0, 0.4, 0.7, 1.0], 
            labels=[0, 1, 2]
        ).astype(float)
    
    # 5. Polynomial features (selected important features)
    if 'danceability' in X.columns:
        X_enhanced['danceability_squared'] = X['danceability'] ** 2
    if 'energy' in X.columns:
        X_enhanced['energy_squared'] = X['energy'] ** 2
    if 'valence' in X.columns:
        X_enhanced['valence_squared'] = X['valence'] ** 2
    if 'tempo' in X.columns:
        X_enhanced['tempo_squared'] = (X['tempo'] / 200) ** 2  # Normalized
    elif 'bpm' in X.columns:
        X_enhanced['tempo_squared'] = (X['bpm'] / 200) ** 2
    
    # 6. Domain-specific features
    # "Hit potential" score (weighted combination)
    if all(col in X.columns for col in ['danceability', 'energy', 'valence']):
        X_enhanced['hit_potential'] = (
            X['danceability'] * 0.4 + 
            X['energy'] * 0.4 + 
            X['valence'] * 0.2
        )
    
    # "Energy balance" (how balanced the song is)
    if all(col in X.columns for col in ['energy', 'acousticness', 'instrumentalness']):
        X_enhanced['energy_balance'] = (
            abs(X['energy'] - X['acousticness']) + 
            abs(X['energy'] - X['instrumentalness'])
        )
    
    # "Mood score" (positive/negative)
    if 'valence' in X.columns:
        X_enhanced['mood_score'] = X['valence'] * 2 - 1  # -1 to 1 scale
    
    X = X_enhanced
    print(f"Advanced feature engineering: Added {len(X_enhanced.columns) - original_feature_count} new features")
    
    print(f"Gecerli veri sayisi: {len(X):,}")
    print(f"Feature sayisi (engineering sonrasi): {len(X.columns)}\n")
    
    # Log transformasyonu (model eğitimindekiyle aynı)
    # Model log scale'de eğitilmiş, bu yüzden y için log transformasyonu yapılmıyor
    # (sadece değerlendirme için original scale'de kalıyor)
    
    # Train/test split (model eğitimindekiyle aynı random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # NaN değerleri doldur (feature selection için)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Feature selector'ı yükle ve uygula (train/test split'ten sonra, scaling'den önce)
    selector_path = BASE_DIR / 'spotify_selector.joblib'
    feature_selector = None
    if selector_path.exists():
        try:
            feature_selector = joblib.load(selector_path)
            # Feature selection uygula
            X_train_selected = feature_selector.transform(X_train)
            X_test_selected = feature_selector.transform(X_test)
            # DataFrame'e çevir
            selected_features = X_train.columns[feature_selector.get_support()].tolist()
            X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            print(f"Feature selection uygulandi: {len(selected_features)} feature secildi\n")
        except Exception as e:
            print(f"Feature selector yuklenemedi: {e}\n")
            print(f"Feature selection olmadan devam ediliyor...\n")
    
    # Log transformasyonu (model log scale'de eğitilmiş)
    # Original scale'i sakla (değerlendirme için)
    y_test_original = y_test.copy()
    y_train_original = y_train.copy()
    
    # Log transformasyonu uygula
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Scaler'ı yükle ve uygula
    scaler_path = BASE_DIR / 'spotify_scaler.joblib'
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            # DataFrame'i numpy array'e çevir
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            X_train_scaled = scaler.transform(X_train_array)
            X_test_scaled = scaler.transform(X_test_array)
            print("Scaler yuklendi ve uygulandi\n")
        except Exception as e:
            print(f"Scaler yuklenemedi: {e}\n")
            X_train_scaled = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_scaled = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    else:
        X_train_scaled = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_scaled = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    print("=" * 60)
    print("MODEL BASARI METRIKLERI")
    print("=" * 60)
    
    # Test seti üzerinde tahmin yap (log scale'de)
    y_pred_log = model.predict(X_test_scaled)
    
    # Log transformasyonunu geri dönüştür (original scale'e)
    y_pred = np.expm1(y_pred_log)
    
    # Metrikleri hesapla (original scale'de)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    
    # Log scale'de de metrikler
    r2_log = r2_score(y_test_log, y_pred_log)
    
    # MAPE hesapla (y_test_original'de 0 olmaması gerekir)
    y_test_nonzero = y_test_original[y_test_original > 0]
    y_pred_nonzero = y_pred[y_test_original > 0]
    if len(y_test_nonzero) > 0:
        mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
    else:
        mape = None
    
    # İstatistikler
    print(f"\nTest Seti Istatistikleri:")
    print(f"   - Test ornek sayisi: {len(X_test):,}")
    print(f"   - Gercek deger ortalamasi: {y_test_original.mean():,.2f}")
    print(f"   - Gercek deger medyani: {y_test_original.median():,.2f}")
    print(f"   - Gercek deger std: {y_test_original.std():,.2f}")
    print(f"   - Tahmin ortalamasi: {np.mean(y_pred):,.2f}")
    print(f"   - Tahmin medyani: {np.median(y_pred):,.2f}")
    
    print(f"\nBasari Metrikleri (Original Scale):")
    print(f"   - R2 Score (R-squared): {r2:.4f}")
    print(f"   - MSE (Mean Squared Error): {mse:,.2f}")
    print(f"   - RMSE (Root Mean Squared Error): {rmse:,.2f}")
    print(f"   - MAE (Mean Absolute Error): {mae:,.2f}")
    if mape is not None:
        print(f"   - MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    print(f"\nBasari Metrikleri (Log Scale):")
    print(f"   - R2 Score (R-squared): {r2_log:.4f}")
    
    # R² yorumu
    print(f"\nR2 Score Yorumu:")
    if r2 >= 0.9:
        print(f"   MUKEMMEL! Model varyansin %{r2*100:.1f}'ini acikliyor.")
    elif r2 >= 0.7:
        print(f"   IYI! Model varyansin %{r2*100:.1f}'ini acikliyor.")
    elif r2 >= 0.5:
        print(f"   ORTA. Model varyansin %{r2*100:.1f}'ini acikliyor.")
    elif r2 >= 0.3:
        print(f"   ZAYIF. Model varyansin sadece %{r2*100:.1f}'ini acikliyor.")
    else:
        print(f"   COK ZAYIF. Model yeterince iyi tahmin yapamiyor.")
    
    # Örnek tahminler
    print(f"\nOrnek Tahminler (Ilk 10):")
    print("-" * 60)
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    for idx in sample_indices:
        actual = y_test_original.iloc[idx]
        predicted = y_pred[idx]
        error = abs(actual - predicted)
        error_pct = (error / actual * 100) if actual > 0 else 0
        print(f"   Gerçek: {actual:,.0f} | Tahmin: {predicted:,.0f} | Hata: {error:,.0f} ({error_pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Degerlendirme tamamlandi!")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model()
