# Django Microservice Template

AWS Free Tier için optimize edilmiş, Docker ve Grafana içeren temiz ve sade bir Django microwebservis template'i.

## 🚀 Hızlı Başlangıç

```bash
# Tek komut ile başlat (migration'lar otomatik çalışır)
docker compose up -d

# Tarayıcıda aç: http://localhost:8000
```

**✨ Otomatik:** Migration'lar, static files ve tüm servisler otomatik başlar!

## ✨ Özellikler

- ✅ Django 4.2 + Django REST Framework
- ✅ HTML Templates + Tailwind CSS + JavaScript
- ✅ Docker ve Docker Compose
- ✅ PostgreSQL + Prometheus + Grafana
- ✅ GitHub Actions CI/CD
- ✅ AWS Free Tier uyumlu
- ✅ 3 Örnek Proje (ML Example + DS Example + DL Example)
- ✅ **Modüler Mimari** - Sınırsız bağımsız proje
- ✅ **Otomatik Requirements Merge** - Proje requirements.txt dosyaları otomatik algılanır

## 📁 Proje Yapısı

```
├── api/              # Ana API uygulaması
├── config/           # Django ayarları
├── templates/        # HTML template'leri
├── static/           # CSS, JS dosyaları
├── projects/         # Modüler projeler (örnek: ml_example, ds_example, dl_example)
├── monitoring/       # Prometheus/Grafana
└── scripts/          # Yardımcı scriptler
```

## 🎯 Modüler Mimari

**Sınırsız sayıda bağımsız proje** oluşturabilir ve istediğiniz zaman **ayırıp satabilirsiniz**.

### ✨ Yeni Proje Oluştur (Otomatik)

```bash
# Script ile proje oluştur (önerilen)
python scripts/create_project.py ml_project_1 --type=ml

# Django'yu yeniden başlat
docker compose restart web

# Proje otomatik algılanır! URL: http://localhost:8000/ml-project-1/
```

### Projeyi Ayır (Sat/Ver)

```bash
python scripts/extract_project.py ml_project_1
```

**Detaylar:** Her proje tamamen bağımsızdır ve kendi `models.py`, `views.py`, `urls.py`, `templates/`, `static/` dosyalarına sahiptir.

## 📦 Örnek Projeler

Template içinde üç örnek bağımsız proje bulunmaktadır:

### 1. ML Example (`projects/ml_example/`)

ML Prediction örneği. Sklearn modeli ile prediction yapma.

**✨ Otomatik algılanır!** Hiçbir şey eklemenize gerek yok.

**Kullanım:**
- Web: http://localhost:8000/ml-example/
- API: http://localhost:8000/ml-example/predict/

**Model oluşturma:**
```bash
python projects/ml_example/scripts/create_sample_model.py
```

### 2. DS Example (`projects/ds_example/`)

Data Analysis örneği. CSV dosyası yükleme ve pandas ile analiz.

**✨ Otomatik algılanır!** Hiçbir şey eklemenize gerek yok.

**Kullanım:**
- Web: http://localhost:8000/ds-example/
- API: http://localhost:8000/ds-example/analyze/

### 3. DL Example (`projects/dl_example/`)

Deep Learning örneği. TensorFlow/Keras modeli ile image classification ve array prediction.

**✨ Otomatik algılanır!** Hiçbir şey eklemenize gerek yok.

**Kullanım:**
- Web: http://localhost:8000/dl-example/
- API: 
  - Image: http://localhost:8000/dl-example/predict-image/
  - Array: http://localhost:8000/dl-example/predict-array/

**Model oluşturma:**
```bash
python projects/dl_example/scripts/create_sample_model.py
```

**Detaylar:** Her projenin kendi `README.md` dosyasına bakın.

## 🛠️ Geliştirme

### Yeni Sayfa/API Ekleme

- **Sayfa:** `templates/my_page.html` → `api/views.py` → `config/urls.py`
- **API:** `api/views.py` → `api/urls.py`

### Frontend (Tailwind CSS)

```bash
npm install
npm run build-css      # Build
npm run watch-css      # Watch mode
```

### Otomatik Requirements Merge

`projects/*/requirements.txt` dosyaları Docker build sırasında otomatik birleştirilir. Manuel ekleme gerekmez!

## 🌐 Servisler

- **Web**: http://localhost:8000
- **API**: http://localhost:8000/api/
- **Admin**: http://localhost:8000/admin
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📚 Daha Fazla Bilgi

- **AWS Deployment**: `aws/README.md`
- **Örnek Projeler**: Her projenin kendi `README.md` dosyasına bakın

## 🔒 Güvenlik

Production'da `DEBUG=False`, `SECRET_KEY` ve `ALLOWED_HOSTS` ayarlarını güncelleyin.

---

**Lisans:** Bu template serbestçe kullanılabilir.
