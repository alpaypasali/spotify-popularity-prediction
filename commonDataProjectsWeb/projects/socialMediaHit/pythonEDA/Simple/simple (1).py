# Veri manipülasyonu için
import pandas as pd
import numpy as np

# Görselleştirme için
import matplotlib.pyplot as plt
import seaborn as sns


# Uyarıları devre dışı bırak
import warnings
warnings.filterwarnings('ignore')

# Pandas gösterim ayarları
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', 100)      # En fazla 100 satır göster
pd.set_option('display.width', 1000)        # Tablo genişliği 1000 karakter
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Ondalık sayılar 3 basamak (0.123)

# Görselleştirme ayarları
sns.set_theme(style="whitegrid")  # Seaborn grafikleri beyaz + ızgara
plt.rcParams['figure.figsize'] = (12, 8)  # Varsayılan grafik boyutu

# | Kolon                    | Açıklama                                 | Data Durumu            |
# | ------------------------ | ----------------------------------------- | ----------------------- |
# | `track_name`             | Şarkı adı                                | ✔ 330k non-null         |
# | `artist_name`            | Sanatçı adı                              | ✔ 330k non-null         |
# | `release_year`           | Yayın yılı                               | ~234k non-null          |
# | `explicit`               | +18 içerik etiketi (0/1)                 | ✔ 327k non-null         |
# | `danceability`           | Dans edilebilirlik (0–1)                 | ✔ 327k non-null         |
# | `energy`                 | Enerji seviyesi (0–1)                    | ✔ 327k non-null         |
# | `valence`                | Neşelilik / mutluluk seviyesi (0–1)      | ✔ 327k non-null         |
# | `speechiness`            | Konuşma oranı                            | ✔ 327k non-null         |
# | `liveness`               | Canlı kayıt olasılığı                    | ✔ 327k non-null         |
# | `acousticness`           | Akustiklik olasılığı                     | ✔ 327k non-null         |
# | `instrumentalness`       | Enstrümantal olma olasılığı              | ✔ 327k non-null         |
# | `tempo`                  | BPM                                      | ✔ 327k non-null         |
# | `loudness_db`            | Ses yüksekliği (dB)                      | ✔ 326k non-null         |
# | `spotify_popularity_0_100` | Spotify popülerlik skoru (0–100)        | ✔ 328k non-null         |
# | `source_dataset`         | Verinin kaynağı (df1/df2/df3/df4/df5)    | ✔ 330k non-null         |
# | `duration_ms`            | Şarkı süresi (ms)                        | ~95k non-null           |
# | `key`                    | Müzikal ton (0–11)                       | ~96k non-null           |
# | `mode`                   | Majör/Minör (1/0)                        | ~96k non-null           |
# | `streams`                | Genel stream sayısı (df5)                | ~661 non-null           |
# | `in_spotify_playlists`   | Spotify playlist sayısı (df5)            | ~661 non-null           |
# | `in_spotify_charts`      | Spotify chart bilgisi (df5)              | ~661 non-null           |
# | `spotify_streams`        | Spotify stream sayısı (df3)              | ~2857 non-null          |
# | `track_score`            | Global hit skoru (df3)                   | ~2951 non-null          |
# | `youtube_views`          | YouTube görüntülenme sayısı (df3)        | ~2741 non-null          |
# | `tiktok_views`           | TikTok görüntülenme sayısı (df3)         | ~2110 non-null          |
# | `shazam_counts`          | Shazam tanıma sayısı (df3)               | ~2581 non-null          |
# | `hit`                    | popularity > 70                          | ~330k non-null          |
# | `shazam_counts`          | popularity 3 kolona ayrilmis             | ~330k non-null          |

def read_dataset(path):

    df = pd.read_csv(path)
    return df




############################################
# 1. Keşifçi Veri Analizi
############################################


def check_df(dataframe, head=5, name=""):
    print(f'##################### {name} Dataset Overview #####################')
    print('\n##################### Shape #####################')
    print(dataframe.shape)

    print('\n##################### Types #####################')
    print(dataframe.dtypes)

    print('\n##################### Head #####################')
    print(dataframe.head(head))

    print('\n##################### Tail #####################')
    print(dataframe.tail(head))

    print('\n##################### NA #####################')
    print(dataframe.isnull().sum())

    print('\n##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)





############################################
# 2. Eksik Veya Analiz Edilemiyecek Vefilerin Silinmesi
############################################
def isNeedDropped(df):
    """
    Bu fonksiyon, veri ön işleme aşamasında analiste 'Human-in-the-loop' (İnsan döngüde)
    özelliği kazandırarak interaktif sütun temizliği yapılmasını sağlar.

    Ne İşe Yarar?
    ----------------
    1. İnteraktif Karar: Kod çalışırken kullanıcıya gereksiz sütunları sorar, böylece
       statik kod yazmak yerine anlık kararlarla veriyi temizlemeyi sağlar.
    2. Hata Önleme (Validation): Kullanıcının yazdığı sütun isimlerini kontrol eder;
       yanlış veya olmayan bir isim girilirse kodun hata verip durmasını (crash) engeller.
    3. Anlık Raporlama: Silme işlemi sonrası DataFrame'in yeni boyutlarını (satır/sütun)
       göstererek işlemin başarısını doğrular.
    4. Bellek Yönetimi: İşlemi 'inplace=True' ile yaparak verinin kopyasını oluşturmaz,
       doğrudan mevcut veri üzerinde değişiklik yapar.

    Args:
        df (pd.DataFrame): İşlem yapılacak Pandas DataFrame.
    """

    # 1. Kullanıcıdan Sütun Silme Onayı ve İsimleri Alma
    cevap = input(
        "\nVeri ön incelemede silmek istediğiniz (gereksiz gördüğünüz) sütunlar var mı? (evet/hayır): ").lower()

    if cevap == 'evet':

        print("\nMevcut Sütunlar:")
        print(list(df.columns))

        drop_list_input = input(
            "\nSilmek istediğiniz sütunların adlarını virgülle ayırarak yazın (Örn: Sütun1,Sütun2): ")

        # Girişi temizle ve liste oluştur
        # (Boşlukları temizler ve sadece DataFrame'de gerçekten var olan sütunları seçer)
        drop_list = [col.strip() for col in drop_list_input.split(',') if col.strip() in df.columns]

        if drop_list:
            # 3. Sütunları Silme İşlemi
            # axis=1 sütunları, inplace=True ise DataFrame'i kalıcı olarak değiştirir.
            df.drop(drop_list, axis=1, inplace=True)

            # 4. Çıktı ve Sonuçları Gösterme
            print(f"\n✅ Başarıyla Silinen Sütunlar: {', '.join(drop_list)}")
            print("\n--- 📝 İşlem Sonrası Durum ---")
            print(f"Yeni Satır Sayısı: {df.shape[0]}")
            print(f"Yeni Sütun Sayısı: {df.shape[1]}")
            print("-----------------------------")
        else:
            print(
                "\n❌ Geçerli silinecek sütun adı girilmedi veya mevcut sütunlar arasında bulunamadı. Silme işlemi yapılmadı.")
    else:
        print("\nİnceleme sonrası herhangi bir sütun silme işlemi yapılmadı.")




############################################
# 3. Sayısal ve Kategorik Değişkenlerin Tespiti
############################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe
    cat_th: int, float
        Numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    """

    # Kategorik kolonların listesi
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Numerik ama kategorik kolonlar
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # Kategorik ama kardinal kolonlar
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Kategorik kolonların son listesi
    cat_cols = cat_cols + num_but_cat

    # Kategorik ama kardinal olmayan kolonlar
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerik kolonlar
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(dataframe.head())
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(cat_cols)
    print(f"num_cols: {len(num_cols)}")
    print(num_cols)
    print(f"cat_but_car: {len(cat_but_car)}")
    print(cat_but_car)
    print(f"num_but_cat: {len(num_but_cat)}")
    print(num_but_cat)

    return cat_cols, num_cols, cat_but_car, num_but_cat



############################
# 4. Kategorik Veri Analizi
###########################


def cat_summary_advanced(dataframe, col_name, plot=False):
    """
    Bir kategorik değişken için özet tablo ve 4 farklı görselleştirme (Dashboard) oluşturur.

    Görseller:
    1. Countplot (Dikey Çubuk)
    2. Pie Chart (Yüzdelik Pasta)
    3. Horizontal Bar Plot (Yatay ve Sıralı)
    4. Donut Chart (Halka Grafik)
    """

    # 1. Veri Hazırlama ve Tablo Yazdırma
    col_count = dataframe[col_name].value_counts()
    summary_df = pd.DataFrame({
        col_name: col_count,
        'Ratio (%)': 100 * col_count / len(dataframe)
    })

    print(f"--- 📊 {col_name.upper()} DEĞİŞKENİ ÖZETİ ---")
    print(summary_df)
    print('##########################################')

    if plot:
        # 4'lü Grafik Alanı Oluşturma (2 Satır, 2 Sütun)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"'{col_name}' Değişkeni İçin Detaylı Analiz", fontsize=20, fontweight='bold')

        # --- GÖRSEL 1: Sütun Grafik (Countplot) ---
        # Klasik sıklık grafiği
        sns.countplot(x=dataframe[col_name], ax=axes[0, 0], palette="viridis", order=col_count.index)
        axes[0, 0].set_title("1. Sıklık Dağılımı (Bar Plot)", fontsize=14)
        axes[0, 0].set_xlabel(col_name)
        axes[0, 0].set_ylabel("Adet")

        # Barların üzerine sayıları yazdırma döngüsü
        for p in axes[0, 0].patches:
            axes[0, 0].annotate(f'{int(p.get_height())}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=11)

        # --- GÖRSEL 2: Pasta Grafik (Pie Chart) ---
        # Oransal dağılımı görmek için
        axes[0, 1].pie(col_count, labels=col_count.index, autopct='%1.1f%%',
                       startangle=140, colors=sns.color_palette("pastel"))
        axes[0, 1].set_title("2. Oransal Dağılım (Pie Chart)", fontsize=14)

        # --- GÖRSEL 3: Yatay Çubuk Grafik (Horizontal Bar Plot) ---
        # Okunabilirlik için yatay ve büyükten küçüğe sıralı
        sns.barplot(x=col_count.values, y=col_count.index, ax=axes[1, 0], palette="magma")
        axes[1, 0].set_title("3. Sıralı Görünüm (Horizontal Bar)", fontsize=14)
        axes[1, 0].set_xlabel("Adet")
        axes[1, 0].set_ylabel(col_name)

        # --- GÖRSEL 4: Halka Grafik (Donut Chart) ---
        # Pasta grafiğin modern alternatifi
        # Önce pasta çizilir, sonra ortasına beyaz daire eklenir
        wedges, texts, autotexts = axes[1, 1].pie(col_count, labels=col_count.index, autopct='%1.1f%%',
                                                  pctdistance=0.85, colors=sns.color_palette("Set2"))
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')  # Ortadaki beyaz daire
        axes[1, 1].add_artist(centre_circle)
        axes[1, 1].set_title("4. Halka Görünüm (Donut Chart)", fontsize=14)

        plt.tight_layout(pad=3.0)  # Grafikler arası boşluğu ayarla
        plt.show(block=True)




############################
# 5. Numerik Veri Analizi
###########################

def num_summary(dataframe, numerical_col, plot=False):
    """
    Sayısal değişkenler için profesyonel istatistiksel özet ve görselleştirme.

    Özellikler:
    1. İstatistikler: Quantiles, Skewness (Çarpıklık), Kurtosis (Basıklık).
    2. Görselleştirme: Boxplot (Aykırı değerler) ve Histogram (Dağılım) bir arada.
    3. Referanslar: Ortalama ve Medyan çizgileri ile dağılımın yönü.
    """

    # 1. Gelişmiş İstatistiksel Özet
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(f"########## 📊 {numerical_col.upper()} İSTATİSTİKLERİ ##########")
    desc = dataframe[numerical_col].describe(quantiles).T

    # Ekstra metrikleri hesapla ve ekle
    # Skewness > 0 ise sağa çarpık, < 0 ise sola çarpık.
    print(desc)
    print(f"\nSkewness (Çarpıklık): {dataframe[numerical_col].skew():.4f}")
    print(f"Kurtosis (Basıklık) : {dataframe[numerical_col].kurtosis():.4f}")
    print(f"Eksik Değer Sayısı  : {dataframe[numerical_col].isnull().sum()}")
    print("############################################################")

    if plot:
        # 2. Profesyonel Görselleştirme (Boxplot + Histogram)
        # sharex=True ile iki grafiğin X eksenini ortak yapıyoruz.
        # gridspec_kw ile üstteki grafiği daha dar (ince) yapıyoruz.
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                              gridspec_kw={"height_ratios": (.15, .85)})

        fig.suptitle(f"'{numerical_col}' Değişkeni Dağılım Analizi", fontsize=16, fontweight='bold')

        # Üst Grafik: Boxplot (Aykırı değerleri yakalamak için)
        sns.boxplot(x=dataframe[numerical_col], ax=ax_box, color="lightblue")
        ax_box.set(xlabel="")  # Üst grafiğin x label'ını kaldır (alttaki yeterli)

        # Alt Grafik: Histogram + KDE (Yoğunluk Eğrisi)
        sns.histplot(x=dataframe[numerical_col], ax=ax_hist, kde=True, bins=30, color="steelblue")

        # 3. Ortalama ve Medyan Çizgileri (Dağılımın yönünü anlamak için kritik)
        mean_val = dataframe[numerical_col].mean()
        median_val = dataframe[numerical_col].median()

        ax_hist.axvline(mean_val, color='r', linestyle='--', label=f'Ortalama: {mean_val:.2f}')
        ax_hist.axvline(median_val, color='g', linestyle='-', label=f'Medyan: {median_val:.2f}')

        plt.legend(loc='upper right')
        plt.xlabel(numerical_col, fontsize=12)
        plt.ylabel("Frekans", fontsize=12)

        # Grafikler arası boşluğu ayarla
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=True)





############################
# 6. Hedef Değişkene Göre Kategorik Değişken Analizi
###########################

def target_summary_with_cat(dataframe, target, categorical_col, project_type="supervised", model_type="regression",
                            plot=False):
    """
    Kategorik değişkenin hedef değişkenle olan ilişkisini analiz eder.
    Proje türüne ve hedef değişkenin tipine (Sayısal/Kategorik) göre dinamik davranır.

    Args:
        dataframe (pd.DataFrame): Veri seti.
        target (str): Hedef değişken adı.
        categorical_col (str): Analiz edilecek kategorik sütun.
        project_type (str): 'supervised' veya 'unsupervised'.
        model_type (str): 'regression' (Sayısal Hedef) veya 'classification' (Kategorik Hedef).
        plot (bool): Grafik çizilsin mi?
    """

    # 1. SENARYO: Gözetimsiz Öğrenme (Unsupervised)
    # Hedef değişken olmadığı için analiz yapılmaz.
    if project_type.lower() == "unsupervised":
        print(f"⚠️ DİKKAT: Proje türü '{project_type}' seçildiği için Target analizi atlandı.")
        print("Bilgi: Gözetimsiz öğrenmede (Clustering vb.) hedef değişken analizi yapılmaz.")
        return

    # 2. SENARYO: Gözetimli Öğrenme (Supervised)
    print(f"########## 📊 {categorical_col.upper()} vs {target.upper()} ANALİZİ ##########")

    # A) REGRESYON (Sayısal Hedef Değişkeni, Örn: Fiyat, Maaş)
    if model_type.lower() == "regression":
        # Sadece ortalamaya bakmak yetmez, kaç veri olduğuna (count) da bakmalıyız.
        # Az veriyle hesaplanan ortalama yanıltıcıdır.
        summary = dataframe.groupby(categorical_col)[target].agg(["mean", "count", "median"])
        summary.columns = ['Target Mean', 'Count', 'Target Median']
        print(summary.sort_values("Target Mean", ascending=False))

        if plot:
            plt.figure(figsize=(12, 6))
            # Barplot ortalamayı gösterir, hata çubukları (error bars) güven aralığını verir.
            sns.barplot(x=categorical_col, y=target, data=dataframe, palette="viridis")
            plt.title(f"{categorical_col} Kategorisine Göre '{target}' Ortalamaları (Regression)")
            plt.ylabel(f"Ortalama {target}")
            plt.xticks(rotation=45)
            plt.show(block=True)

    # B) SINIFLANDIRMA (Kategorik Hedef Değişkeni, Örn: Churn, Onay/Red)
    elif model_type.lower() == "classification":
        # Crosstab ile frekans tablosu oluşturmak daha doğrudur.
        # Normalize='index' ile satır bazlı oranları görürüz.
        print("\n--- Sınıf Frekansları ve Oranları ---")
        ct = pd.crosstab(dataframe[categorical_col], dataframe[target], normalize='index') * 100
        print(ct)

        if plot:
            plt.figure(figsize=(12, 6))
            # Countplot hue parametresi ile kırılımı gösterir
            sns.countplot(x=categorical_col, hue=target, data=dataframe, palette="Set2")
            plt.title(f"{categorical_col} İçindeki '{target}' Sınıf Dağılımı (Classification)")
            plt.ylabel("Kişi/Veri Sayısı")
            plt.xticks(rotation=45)
            plt.legend(title=target)
            plt.show(block=True)

    else:
        print("❌ HATA: Geçersiz 'model_type'. Lütfen 'regression' veya 'classification' giriniz.")

    print("#################################################################\n")




############################
# 6. Hedef Değişkene Göre Sayisal Değişken Analizi
###########################


def target_summary_with_num_advanced(
    dataframe,
    target,
    numerical_col,
    project_type="supervised",
    model_type="classification",
    plot=False
):
    """
    Sayısal değişken ile hedef değişken arasındaki ilişkiyi
    Boxplot, KDE ve ECDF ile karar odaklı analiz eder.
    """

    if project_type.lower() == "unsupervised":
        return

    print(f"########## {numerical_col.upper()} vs {target.upper()} ##########")

    # -------------------- CLASSIFICATION --------------------
    if model_type.lower() == "classification":

        # İstatistiksel özet
        print(dataframe.groupby(target)[numerical_col].describe().T)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1️⃣ BOXPLOT – Medyan & IQR
            sns.boxplot(
                x=target,
                y=numerical_col,
                data=dataframe,
                ax=axes[0]
            )
            axes[0].set_title("Boxplot", fontsize=10)
            axes[0].tick_params(axis='both', labelsize=9)

            # 2️⃣ KDE – Dağılım Ayrımı
            for cls in dataframe[target].unique():
                sns.kdeplot(
                    dataframe[dataframe[target] == cls][numerical_col],
                    label=f"{target}={cls}",
                    fill=True,
                    ax=axes[1],
                    alpha=0.5
                )
            axes[1].set_title("Density (KDE)", fontsize=10)
            axes[1].legend(fontsize=8)
            axes[1].tick_params(axis='both', labelsize=9)

            # 3️⃣ ECDF – Eşik Yorumu
            for cls in dataframe[target].unique():
                sns.ecdfplot(
                    dataframe[dataframe[target] == cls][numerical_col],
                    label=f"{target}={cls}",
                    ax=axes[2]
                )
            axes[2].set_title("ECDF", fontsize=10)
            axes[2].legend(fontsize=8)
            axes[2].tick_params(axis='both', labelsize=9)

            fig.suptitle(
                f"{numerical_col} vs {target} – Karar Odaklı Analiz",
                fontsize=12
            )
            plt.tight_layout()
            plt.show()

    # -------------------- REGRESSION --------------------
    elif model_type.lower() == "regression":

        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    else:
        print("HATA: model_type 'classification' veya 'regression' olmalıdır.")

    print("############################################################\n")




############################
# 7. Korelasyon Analizi Ham Verilerle
###########################

def advanced_correlation_analysis(dataframe, target_col=None, drop_high_corr=False, corr_th=0.90, plot=True):
    """
    Kapsamlı korelasyon analizi, hedef değişken incelemesi ve multicollinearity tespiti yapar.

    Args:
        dataframe (pd.DataFrame): Veri seti.
        target_col (str, optional): Hedef değişken. Varsa özel analiz yapılır.
        drop_high_corr (bool): True ise, yüksek korelasyonlu değişkenlerden birini öneri listesine ekler.
        corr_th (float): Yüksek korelasyon eşik değeri (Örn: 0.90).
        plot (bool): Görseller çizilsin mi?

    Returns:
        drop_list (list): Yüksek korelasyon sebebiyle silinmesi önerilen değişkenler listesi.
    """

    # 1. Sadece Sayısal Değişkenleri Seç
    num_df = dataframe.select_dtypes(include=[np.number])

    if num_df.shape[1] < 2:
        print("❌ Analiz için en az 2 sayısal değişken gerekli.")
        return []

    # 2. Korelasyon Matrisini Hesapla
    corr_matrix = num_df.corr()

    # --- BÖLÜM 1: MULTICOLLINEARITY TESPİTİ (Yüksek Korelasyonlu Çiftler) ---
    # Matrisin üst üçgenini al (çünkü matris simetriktir, aynı işi iki kere yapmayalım)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Eşik değerden yüksek olan sütunları bul (Mutlak değer olarak bakılır)
    drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col].abs() > corr_th)]

    print(f"########## 📊 KORELASYON ANALİZİ RAPORU ##########")
    if drop_list:
        print(f"\n⚠️ DİKKAT: Yüksek Korelasyonlu ({corr_th}+) Değişkenler Tespit Edildi!")
        print(f"Model kararlılığı için şu değişkenlerin birer eşini silmeyi düşünebilirsiniz:")
        print(f"Önerilen Silinecekler Listesi: {drop_list}")

        # Hangi değişken hangisiyle çakışıyor detaylı göster
        print("\n--- Detaylı Çakışma Listesi ---")
        for col in drop_list:
            # O sütundaki yüksek korelasyonlu diğer değişkeni bul
            high_corr_rows = upper_triangle[col][upper_triangle[col].abs() > corr_th].index.tolist()
            for row in high_corr_rows:
                print(f"• '{col}' <-> '{row}': {upper_triangle.loc[row, col]:.2f}")
    else:
        print(f"\n✅ Temiz: {corr_th} eşik değerini aşan çoklu bağlantı (multicollinearity) bulunamadı.")

    # --- BÖLÜM 2: GÖRSELLEŞTİRME ---
    if plot:
        # A) Genel Heatmap
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                    mask=mask, linewidths=0.5, square=True)
        plt.title(f"Genel Korelasyon Matrisi (Numeric)", fontsize=16)
        plt.show(block=True)

        # B) Hedef Değişken Analizi (Varsa)
        if target_col and target_col in num_df.columns:
            plt.figure(figsize=(10, 6))

            # Hedef değişkenle korelasyonları al, kendisini hariç tut ve sırala
            target_corrs = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

            # Renklendirme: Pozitifler Mavi, Negatifler Kırmızı
            colors = ['#3498db' if c > 0 else '#e74c3c' for c in target_corrs.values]

            sns.barplot(x=target_corrs.values, y=target_corrs.index, palette=colors)

            plt.axvline(0, color='black', linewidth=1)  # Sıfır noktasına çizgi
            plt.title(f"Hedef Değişken '{target_col}' ile Korelasyon Düzeyleri", fontsize=14)
            plt.xlabel("Korelasyon Katsayısı")
            plt.grid(True, axis='x', linestyle='--', alpha=0.5)

            # Çubukların ucuna değerleri yaz
            for i, v in enumerate(target_corrs.values):
                plt.text(v, i, f" {v:.2f}", va='center', fontsize=10, fontweight='bold')

            plt.show(block=True)

    print("############################################################\n")

    return drop_list



def plot_speechiness_by_genre(
    df,
    threshold=0.10,
    condition="low",     # "low" veya "high"
    top_n=15,
    figsize=(10, 5)
):
    """
    Speechiness threshold'una göre genre dağılımını bar chart olarak çizer.
    """

    if condition == "low":
        filtered = df[df["speechiness"] < threshold]
        title = f"Low Speechiness (< {threshold}) Tracks by Genre"
    elif condition == "high":
        filtered = df[df["speechiness"] > threshold]
        title = f"High Speechiness (> {threshold}) Tracks by Genre"
    else:
        raise ValueError("condition parametresi 'low' veya 'high' olmalıdır.")

    genre_counts = (
        filtered["main_genre"]
        .value_counts()
        .head(top_n)
    )

    # İstersen konsola yazdır
    print(genre_counts)

    plt.figure(figsize=figsize)
    genre_counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Genre")
    plt.ylabel("Number of Tracks")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()





def run_eda(df):
    print("\n\n======================== EDA START ========================\n")
    check_df(df)

    print("\n==============================Sayısal ve Kategorik Değişkenlerin Tespiti===================================\n")
    # Değişkenleri kategorize edelim
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    print("\nCategorical Columns:", cat_cols)
    print("\nNumerical Columns:", num_cols)
    print("\n=================================Kategorik Veri Analizi===============================\n")
    for col in cat_cols:
        cat_summary_advanced(df, col, plot=True)
    print("\n=================================Numerik Veri Analizi======================================\n")
    for col in num_cols:
        num_summary(df, col, plot=True)
    print(
        "\n================================= Hedef Değişkene Göre Kategorik Değişken Analizi ======================================\n")

    plot_speechiness_by_genre(df, condition="low")

    # cat_cols listesindeki her bir kategorik değişken için hedef analizi
    for col in cat_cols:
        target_summary_with_cat(
            dataframe=df,
            target="popularity",
            categorical_col=col,
            project_type="supervised",
            model_type="regression",
            plot=True
        )

    print("\n================================= Hedef Değişkene Göre Sayisal Değişken Analizi ======================================\n")
    for col in num_cols:
        target_summary_with_num_advanced(
            dataframe=df,
            target="popularity",
            numerical_col=col,
            project_type="supervised",
            model_type="regression",
            plot=True
        )
    print("\n================================= Korelasyon Analizi Ham Verilerle ======================================\n")
    drop_candidates = advanced_correlation_analysis(df, target_col="popularity", corr_th=0.85, plot=True)
    return drop_candidates , df

df = pd.read_csv("pythonEDA/Simple/spotify_emotion_final_clean.csv")

drop_candidates , df = run_eda(df)

df[df["main_genre"] == "unknown"].groupby("emotion")["popularity"].mean().sort_values(ascending=False)
df[df["main_genre"] == "unknown"].groupby("explicit")["popularity"].mean()
# “Unknown genre içerisinde popularity dağılımı emotion değişkeninden bağımsızdır; explicit olmayan
# içeriklerin daha yüksek ortalama popularity’ye sahip olması, bu grubun spoken-word ve clean içerik ağırlıklı olabileceğine işaret etmektedir.”




