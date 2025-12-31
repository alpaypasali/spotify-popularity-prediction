"""
Django management command to evaluate model performance.
Usage: python manage.py evaluate_model
"""
from django.core.management.base import BaseCommand
from projects.socialMediaHit.services import SpotifyPredictionService
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Command(BaseCommand):
    help = 'Evaluate Spotify prediction model performance'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('\n' + '='*60))
        self.stdout.write(self.style.SUCCESS('📊 MODEL BAŞARI DEĞERLENDİRMESİ'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))
        
        # Service'i yükle
        try:
            service = SpotifyPredictionService()
            self.stdout.write(self.style.SUCCESS('✅ Model ve dataset yüklendi\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Hata: {e}'))
            return
        
        if service.model is None:
            self.stdout.write(self.style.WARNING('⚠️  Model bulunamadı. Model eğitiliyor...'))
            service.train_model()
            if service.model is None:
                self.stdout.write(self.style.ERROR('❌ Model eğitilemedi!'))
                return
        
        if service.df is None or service.df.empty:
            self.stdout.write(self.style.ERROR('❌ Dataset yüklenemedi!'))
            return
        
        # Dataset bilgileri
        self.stdout.write(f'📁 Dataset: {len(service.df)} satır, {len(service.df.columns)} sütun')
        
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
                if col in service.df.columns:
                    available_cols.append(col)
                    break
        
        # Target kolonunu bul
        target_col = None
        for col in ['streams', 'popularity', 'in_spotify_charts']:
            if col in service.df.columns:
                target_col = col
                break
        
        if not available_cols or target_col is None:
            self.stdout.write(self.style.ERROR('❌ Gerekli kolonlar bulunamadı!'))
            return
        
        self.stdout.write(f'🔧 Özellikler: {", ".join(available_cols)}')
        self.stdout.write(f'🎯 Hedef: {target_col}\n')
        
        # Veriyi hazırla
        X = service.df[available_cols].copy()
        
        # Eksik değerleri doldur
        for col in available_cols:
            if X[col].isna().any():
                if col == 'loudness':
                    X[col].fillna(-10, inplace=True)
                elif col in ['tempo', 'bpm']:
                    X[col].fillna(120, inplace=True)
                elif col in ['key', 'mode']:
                    X[col].fillna(0, inplace=True)
                else:
                    X[col].fillna(0.5, inplace=True)
        
        y = pd.to_numeric(service.df[target_col], errors='coerce').fillna(0)
        
        # Geçerli verileri filtrele
        valid_mask = (y > 0) & (y.notna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.stdout.write(f'📊 Geçerli veri: {len(X):,} örnek\n')
        
        # Model'in beklediği feature sırasını kontrol et
        if hasattr(service.model, 'feature_names_in_'):
            expected_features = list(service.model.feature_names_in_)
            X = X[expected_features]
        
        # Train/test split (model eğitimindekiyle aynı)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.stdout.write(self.style.SUCCESS('='*60))
        self.stdout.write(self.style.SUCCESS('📈 BAŞARI METRİKLERİ'))
        self.stdout.write(self.style.SUCCESS('='*60 + '\n'))
        
        # Test seti üzerinde tahmin
        y_pred = service.model.predict(X_test)
        
        # Metrikleri hesapla
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE
        y_test_nonzero = y_test[y_test > 0]
        y_pred_nonzero = y_pred[y_test > 0]
        if len(y_test_nonzero) > 0:
            mape = np.mean(np.abs((y_test_nonzero - y_pred_nonzero) / y_test_nonzero)) * 100
        else:
            mape = None
        
        # İstatistikler
        self.stdout.write('📊 Test Seti İstatistikleri:')
        self.stdout.write(f'   • Test örnek sayısı: {len(X_test):,}')
        self.stdout.write(f'   • Gerçek değer ortalaması: {y_test.mean():,.2f}')
        self.stdout.write(f'   • Gerçek değer medyanı: {y_test.median():,.2f}')
        self.stdout.write(f'   • Gerçek değer std: {y_test.std():,.2f}')
        self.stdout.write(f'   • Tahmin ortalaması: {y_pred.mean():,.2f}')
        self.stdout.write(f'   • Tahmin medyanı: {y_pred.median():,.2f}\n')
        
        # Başarı metrikleri
        self.stdout.write('🎯 Başarı Metrikleri:')
        self.stdout.write(self.style.SUCCESS(f'   • R² Score: {r2:.4f}'))
        self.stdout.write(f'   • MSE: {mse:,.2f}')
        self.stdout.write(f'   • RMSE: {rmse:,.2f}')
        self.stdout.write(f'   • MAE: {mae:,.2f}')
        if mape is not None:
            self.stdout.write(f'   • MAPE: {mape:.2f}%\n')
        
        # R² yorumu
        self.stdout.write('📝 R² Score Yorumu:')
        if r2 >= 0.9:
            self.stdout.write(self.style.SUCCESS(
                f'   ✅ Mükemmel! Model varyansın %{r2*100:.1f}\'ini açıklıyor.'
            ))
        elif r2 >= 0.7:
            self.stdout.write(self.style.SUCCESS(
                f'   ✅ İyi! Model varyansın %{r2*100:.1f}\'ini açıklıyor.'
            ))
        elif r2 >= 0.5:
            self.stdout.write(self.style.WARNING(
                f'   ⚠️  Orta. Model varyansın %{r2*100:.1f}\'ini açıklıyor.'
            ))
        elif r2 >= 0.3:
            self.stdout.write(self.style.WARNING(
                f'   ⚠️  Zayıf. Model varyansın sadece %{r2*100:.1f}\'ini açıklıyor.'
            ))
        else:
            self.stdout.write(self.style.ERROR(
                f'   ❌ Çok zayıf. Model yeterince iyi tahmin yapamıyor.'
            ))
        
        # Örnek tahminler
        self.stdout.write('\n🔍 Örnek Tahminler (İlk 5):')
        self.stdout.write('-'*60)
        sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        for idx in sample_indices:
            actual = y_test.iloc[idx]
            predicted = y_pred[idx]
            error = abs(actual - predicted)
            error_pct = (error / actual * 100) if actual > 0 else 0
            self.stdout.write(
                f'   Gerçek: {actual:,.0f} | Tahmin: {predicted:,.0f} | '
                f'Hata: {error:,.0f} ({error_pct:.1f}%)'
            )
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('✅ Değerlendirme tamamlandı!'))
        self.stdout.write('='*60 + '\n')

