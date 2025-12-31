# Social Media Hit Prediction

AI-powered platform to predict and create viral Instagram Reels content. Built for influencers, music producers, and DJs who want to maximize their reach.

## 🎯 Project Overview

This platform helps users create the best Instagram Reels content by predicting hit potential across multiple stages:

1. **Spotify Hit Prediction** - Analyze music tracks and predict hit potential
2. **Instagram Picture Prediction** - Predict image performance for Reels
3. **Instagram Caption/Hashtag Prediction** - Optimize captions and hashtags
4. **Final Combination Score** - Overall hit prediction combining all factors

## ✨ Features

### Spotify Analysis
- Real-time audio feature analysis (danceability, energy, valence, loudness, tempo)
- Interactive playlist with song selection
- Similar song recommendations based on audio features
- Market-specific hit predictions (US, GB, DE, FR, JP, BR)
- Adjustable feature sliders for custom analysis
- Live visualization with Chart.js

### Instagram Picture Analysis
- Image upload and analysis
- Visual feature extraction
- Hit probability prediction
- Performance recommendations

### Instagram Caption/Hashtag Analysis
- Text analysis and sentiment detection
- Hashtag optimization
- Engagement recommendations
- Real-time prediction updates

### Final Prediction
- Combined score calculation
- Weighted breakdown (Spotify 40%, Picture 30%, Caption 30%)
- Actionable recommendations

## 🛠️ Technology Stack

### Backend
- **Django 4.2** - Web framework
- **Django REST Framework** - API endpoints
- **XGBoost** - Gradient boosting for predictions
- **LightGBM** - Fast gradient boosting
- **Random Forest** - Ensemble learning
- **scikit-learn** - ML utilities
- **Pandas** - Data processing
- **NumPy** - Numerical computing

### Frontend
- **Tailwind CSS** - Modern, responsive design
- **Chart.js** - Interactive visualizations
- **Vanilla JavaScript** - Client-side interactivity

### Design
- EA Games / Crytek inspired aesthetic
- Energetic, playful, professional
- Gradient backgrounds with glassmorphism
- Smooth animations and transitions

## 📁 Project Structure

```
socialMediaHit/
├── models.py          # Django models
├── views.py           # View functions and API endpoints
├── urls.py            # URL routing
├── services.py         # ML models and business logic
├── serializers.py     # API serializers
├── requirements.txt   # Python dependencies
├── templates/
│   └── socialMediaHit/
│       ├── home.html                    # About page
│       ├── spotify_prediction.html      # Spotify analysis
│       ├── instagram_picture.html       # Picture analysis
│       ├── instagram_caption.html       # Caption analysis
│       └── final_prediction.html       # Final score
└── static/
    └── socialMediaHit/                  # Static assets
```

## 🚀 Usage

### Access the Platform

1. **About Page**: `http://localhost:8000/socialMediaHit/`
2. **Spotify Prediction**: `http://localhost:8000/socialMediaHit/spotify/`
3. **Instagram Picture**: `http://localhost:8000/socialMediaHit/instagram-picture/`
4. **Instagram Caption**: `http://localhost:8000/socialMediaHit/instagram-caption/`
5. **Final Prediction**: `http://localhost:8000/socialMediaHit/final/`

### API Endpoints

- `GET /api/playlist/` - Get Spotify playlist
- `POST /api/predict/spotify/` - Predict Spotify hit
- `POST /api/similar-songs/` - Get similar songs
- `POST /api/predict/instagram-picture/` - Predict picture hit
- `POST /api/predict/instagram-caption/` - Predict caption hit
- `POST /api/predict/final/` - Predict final combination

## 📊 ML Models

### Spotify Prediction
- **XGBoost Regressor** - Primary model for hit prediction
- Features: danceability, energy, valence, loudness, tempo, etc.
- Market-specific adjustments
- Similar song matching using Euclidean distance

### Instagram Prediction
- **Placeholder models** - Ready for dataset integration
- Will use Random Forest, GBM, XGBoost, LightGBM
- Image feature extraction (brightness, contrast, saturation)
- Text analysis (sentiment, hashtag optimization)

## 📈 Data

### Spotify Dataset
- `spotify-2023.csv` - Main dataset
- `spotify_songs.csv` - Alternative dataset
- `light_spotify_dataset.csv` - Lightweight version
- `Most_Streamed_Spotify_Songs_2024.csv` - Latest data

### Instagram Dataset
- Coming soon - Will be added for picture and caption predictions

## 🔧 Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

Models are automatically trained on first run if dataset is available. Models are saved as `.joblib` files.

### Add Instagram Dataset

1. Place Instagram dataset CSV in project directory
2. Update `InstagramPredictionService` in `services.py`
3. Implement feature extraction and model training

## 🎨 Design Philosophy

- **Energetic** - Vibrant gradients and animations
- **Playful** - Interactive elements and smooth transitions
- **Professional** - Clean layouts and clear information hierarchy
- **Cezbedici** - Eye-catching visuals that engage users

## 📝 Notes

- Spotify prediction is fully functional with real ML models
- Instagram predictions use mock data until datasets are available
- All models use ensemble methods for better accuracy
- Real-time updates provide instant feedback to users

## 🔮 Future Enhancements

- [ ] Complete Instagram dataset integration
- [ ] Advanced image analysis (CNN-based)
- [ ] NLP models for caption optimization
- [ ] User accounts and history tracking
- [ ] A/B testing recommendations
- [ ] Export predictions to CSV/PDF
