import simple


###############################################################################
# MAIN — FULL EDA EXECUTION
###############################################################################
def run_eda(df):
    print("\n\n======================== EDA START ========================\n")
    simple.check_df(df)

    print("\n==============================Sayısal ve Kategorik Değişkenlerin Tespiti===================================\n")
    # Değişkenleri kategorize edelim
    cat_cols, num_cols, cat_but_car, num_but_cat = simple.grab_col_names(df)
    print("\nCategorical Columns:", cat_cols)
    print("\nNumerical Columns:", num_cols)
    print("\n=================================Kategorik Veri Analizi===============================\n")
    for col in cat_cols:
        simple.cat_summary_advanced(df, col, plot=True)
    print("\n=================================Numerik Veri Analizi======================================\n")
    for col in num_cols:
        simple.num_summary(df, col, plot=True)
    print(
        "\n================================= Hedef Değişkene Göre Kategorik Değişken Analizi ======================================\n")



    # cat_cols listesindeki her bir kategorik değişken için hedef analizi
    for col in cat_cols:
        simple.target_summary_with_cat(
            dataframe=df,
            target="popularity",
            categorical_col=col,
            project_type="supervised",
            model_type="regression",
            plot=True
        )

    print("\n================================= Hedef Değişkene Göre Sayisal Değişken Analizi ======================================\n")
    for col in num_cols:
        simple.target_summary_with_num_advanced(
            dataframe=df,
            target="popularity",
            numerical_col=col,
            project_type="supervised",
            model_type="regression",
            plot=True
        )
    print("\n================================= Korelasyon Analizi Ham Verilerle ======================================\n")
    drop_candidates = simple.advanced_correlation_analysis(df, target_col="popularity", corr_th=0.85, plot=True)
    return drop_candidates , df
dataset_path = "spotify_emotion_final_clean.csv"
df = simple.read_dataset(dataset_path)
df["genre"] = (
    df["genre"]
    .str.replace("Unknown", "", regex=False)
    .str.replace(",,", ",")
    .str.strip(", ")
)

df.loc[df["genre"] == "", "genre"] = "unknown"


df["main_genre"] = (
    df["genre"]
    .str.split(",")
    .str[0]
    .str.lower()
    .str.strip()
)
genre_counts = df["main_genre"].value_counts()
rare_genres = genre_counts[genre_counts < 500].index

df["main_genre"] = df["main_genre"].replace(
    rare_genres, "other"
)
df["main_genre"].value_counts()
df["explicit"] = df["explicit"].astype(int)
drop_candidates , df = run_eda(df)

