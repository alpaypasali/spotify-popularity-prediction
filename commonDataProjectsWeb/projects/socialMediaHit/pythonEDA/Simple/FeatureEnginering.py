import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error




def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [
        col for col in cat_cols
        if col not in cat_but_car or col == "GENRE"
    ]
    cat_but_car = [col for col in cat_but_car if col != "GENRE"]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car
# preprocessing
def clean_and_extract_genre(df, rare_threshold=500):
    df = df.copy()

    df["GENRE"] = (
        df["GENRE"]
        .str.replace("Unknown", "", regex=False)
        .str.replace(",,", ",")
        .str.strip(", ")
    )

    df.loc[df["GENRE"] == "", "GENRE"] = "unknown"

    df["GENRE"] = (
        df["GENRE"]
        .str.split(",")
        .str[0]
        .str.lower()
        .str.strip()
    )


    return df
def cast_explicit(df):
    df = df.copy()
    df["EXPLICIT"] = df["EXPLICIT"].astype(int)
    return df
def clean_release_year(df, min_year=1912):
    df = df.copy()

    df["RELEASE_YEAR"] = df["RELEASE_YEAR"].replace(1900, np.nan)

    df["RELEASE_YEAR"] = (
        df.groupby("ARTISTS")["RELEASE_YEAR"]
          .transform(lambda x: x.fillna(x.median()))
    )

    df = df[df["RELEASE_YEAR"] >= min_year].reset_index(drop=True)
    return df
def filter_loudness(df, threshold=-40):
    df = df.copy()

    return df[df["LOUDNESS"] > threshold].reset_index(drop=True)
def create_era(df, min_year=1912):
    df = df.copy()

    df["ERA"] = pd.cut(
        df["RELEASE_YEAR"],
        bins=[min_year, 1970, 1990, 2005, 2015, 2025],
        labels=[
            "pre_modern",
            "old",
            "classic",
            "modern",
            "streaming"
        ],
        include_lowest=True
    )
    return df
def preprocess_pipeline(df):
    df = clean_and_extract_genre(df)
    df = cast_explicit(df)
    df = clean_release_year(df)
    df = filter_loudness(df)
    df = create_era(df)
    return df
# ENCODİNG
def encode_categoricals(
    train_df,
    test_df,
    cat_cols,
    rare_perc=0.01,
    drop_first=True
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # ---------- RARE FIT (TRAIN) ----------
    rare_maps = {}

    for col in cat_cols:
        freq_ratio = train_df[col].value_counts(normalize=True)
        rare_labels = freq_ratio[freq_ratio < rare_perc].index.tolist()
        rare_maps[col] = rare_labels

        train_df[col] = np.where(
            train_df[col].isin(rare_labels), "Rare", train_df[col]
        )

        test_df[col] = np.where(
            test_df[col].isin(rare_labels), "Rare", test_df[col]
        )

    # ---------- BINARY LABEL ENCODING ----------
    binary_cols = [col for col in cat_cols if train_df[col].nunique() == 2]

    for col in binary_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # ---------- ONE HOT ----------
    ohe_cols = [col for col in cat_cols if col not in binary_cols]

    train_df = pd.get_dummies(train_df, columns=ohe_cols, drop_first=drop_first)
    test_df = pd.get_dummies(test_df, columns=ohe_cols, drop_first=drop_first)

    # ---------- COLUMN ALIGN ----------
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    return train_df, test_df



def spotify_feature_fit(train_df):
    ref = {
        "DANCEABILITY_MEAN": train_df["DANCEABILITY"].mean(),
        "ENERGY_MEAN": train_df["ENERGY"].mean(),
        "LOUDNESS_MEAN": train_df["LOUDNESS"].mean()
    }
    return ref



def spotify_data_prep(dataframe, reference_year=2025):
    df = dataframe.copy()

    # =========================
    # INTERACTION / CORE AUDIO
    # =========================
    df["NEW_ENERGY_LOUDNESS"] = df["ENERGY"] * df["LOUDNESS"].abs()
    df["NEW_DANCE_INTENSITY"] = df["DANCEABILITY"] * df["ENERGY"]
    df["NEW_MOOD_SCORE"] = (df["VALENCE"] + df["ENERGY"]) / 2
    df["NEW_ELECTRONICNESS"] = 1 - df["ACOUSTICNESS"]
    df["NEW_ACOUSTIC_ENERGY_RATIO"] = df["ACOUSTICNESS"] / (df["ENERGY"] + 1e-6)
    df["NEW_EMOTIONAL_POLARITY"] = df["VALENCE"] * (2 * df["ENERGY"] - 1)

    # =========================
    # VOCAL / FORMAT
    # =========================
    df["NEW_IS_INSTRUMENTAL"] = (df["INSTRUMENTALNESS"] > 0).astype(int)
    df["NEW_IS_SPEECHY"] = (df["SPEECHINESS"] > 0.33).astype(int)
    df["NEW_EXPLICIT_ENERGY"] = df["EXPLICIT"] * df["ENERGY"]

    # =========================
    # PLATFORM / ALGO SIGNALS
    # =========================
    df["NEW_ALGO_SCORE"] = (
        df["DANCEABILITY"] *
        df["ENERGY"] *
        (1 - df["ACOUSTICNESS"])
    )

    df["NEW_COMMERCIAL_SCORE"] = (
        (1 - df["EXPLICIT"]) *
        (1 - df["SPEECHINESS"]) *
        (1 - df["ACOUSTICNESS"])
    )

    df["NEW_MAINSTREAM_SCORE"] = (
        abs(df["DANCEABILITY"] - 0.6) +
        abs(df["ENERGY"] - 0.7) +
        abs(df["LOUDNESS"] + 6)     # ✅
    )

    # =========================
    # RHYTHM
    # =========================
    df["NEW_RHYTHMIC_DRIVE"] = df["TEMPO"] * df["ENERGY"]

    df["NEW_GROOVE_STABILITY"] = (
        df["DANCEABILITY"] * df["ENERGY"]
    ) / (abs(df["TEMPO"] - 120) + 1)

    # =========================
    # CONTRAST / EXTREMENESS
    # =========================
    df["NEW_EXTREMENESS_SCORE"] = (
        df["SPEECHINESS"] +
        df["INSTRUMENTALNESS"] +
        abs(df["VALENCE"] - df["ENERGY"])
    )

    df["NEW_MOOD_ENERGY_CONFLICT"] = abs(df["VALENCE"] - df["ENERGY"])

    # =========================
    # TIME
    # =========================
    df["NEW_TRACK_AGE"] = reference_year - df["RELEASE_YEAR"]
    df["NEW_IS_STREAMING_ERA"] = df["ERA"].isin(
        ["streaming", "early_stream"]
    ).astype(int)

    # =========================
    # GENRE FLAG
    # =========================
    df["NEW_IS_UNKNOWN_GENRE"] = (df["GENRE"] == "unknown").astype(int)


    # =========================
    # EXTRA AUDIO INTERACTIONS
    # =========================
    df["NEW_SPEECH_ENERGY"] = df["SPEECHINESS"] * df["ENERGY"]
    df["NEW_ACOUSTIC_VALENCE"] = df["ACOUSTICNESS"] * df["VALENCE"]
    df["NEW_LIVE_ENERGY"] = df["LIVENESS"] * df["ENERGY"]

    return df



def base_regression_models_large(X, y, scoring="neg_root_mean_squared_error", top_n=3):
    print("Base Regression Models (Large Data)....\n")

    regressors = [
        ('LR', LinearRegression()),
        ('Ridge', Ridge()),
        ('Lasso', Lasso()),
        ('KNN', KNeighborsRegressor()),
        ('CART', DecisionTreeRegressor( )),
        ('RF', RandomForestRegressor()),
        ('GBM', GradientBoostingRegressor()),
        ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMRegressor(verbosity=-1))
    ]

    results = []

    for name, regressor in regressors:
        cv_results = cross_validate(
            regressor,
            X,
            y,
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )

        score = -cv_results["test_score"].mean()
        results.append((name, score, regressor))

        print(f"{scoring.replace('neg_', '')}: {round(score, 4)} ({name})")


    results = sorted(results, key=lambda x: x[1])


    return results

xgb_params_fast = {
    "n_estimators": [700, 900],
    "learning_rate": [0.03, 0.5],
    "max_depth": [5, 8],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [3, 5],
    "gamma": [0.2],
    "reg_alpha": [0.5],
}


lgbm_params = {
      "n_estimators": [1000],
    "learning_rate": [0.04],
    "num_leaves": [48, 52],
    "max_depth": [8],
    "min_child_samples": [40, 60],
    "reg_lambda": [4, 6],
    "reg_alpha": [0.3, 0.5],
}
regressors = [

    ("XGBoost", XGBRegressor(objective="reg:squarederror", n_jobs=-1), xgb_params_fast),
    ("LightGBM", LGBMRegressor( n_jobs=-1, verbosity=-1), lgbm_params)

]

def hyperparameter_optimization_regression(
    X, y,
    cv=3,
    scoring="neg_root_mean_squared_error"
):
    print("Hyperparameter Optimization (Regression)...\n")
    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")

        # BEFORE
        cv_results = cross_validate(
            regressor,
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        before_rmse = -cv_results["test_score"].mean()
        print(f"RMSE (Before): {round(before_rmse, 4)}")

        # GRID SEARCH
        gs = GridSearchCV(
            regressor,
            params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

        gs.fit(X, y)

        # ✅ FIT EDİLMİŞ EN İYİ MODEL
        final_model = gs.best_estimator_

        # AFTER (aynı estimator ile CV)
        cv_results = cross_validate(
            final_model,
            X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        after_rmse = -cv_results["test_score"].mean()
        print(f"RMSE (After): {round(after_rmse, 4)}")
        print(f"{name} best params: {gs.best_params_}\n")

        best_models[name] = final_model

    return best_models


def voting_regressor(best_models, X, y, cv=3):
    print("Voting Regressor...\n")

    voting_reg = VotingRegressor(
        estimators=[
            ("XGBoost", best_models["XGBoost"]),
            ("LightGBM", best_models["LightGBM"]),

        ]
    )

    cv_results = cross_validate(
        voting_reg,
        X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse = -cv_results["test_score"].mean()
    print(f"RMSE (CV): {round(rmse, 4)}")

    voting_reg.fit(X, y)

    return voting_reg


def main():
    # =========================
    # 1. LOAD DATA
    # =========================
    df = pd.read_csv("spotify_emotion_final_clean.csv")
    df.columns = [c.upper() for c in df.columns]

    TARGET = "POPULARITY"
    DROP_COLS = ["ARTISTS", "TRACK_NAME", "SOURCE"]

    # =========================
    # 2. BASIC PREPROCESS
    # =========================
    df = preprocess_pipeline(df)

    # =========================
    # 3. TRAIN / TEST SPLIT
    # =========================
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42
    )

    # =========================
    # 4. FEATURE ENGINEERING
    # =========================
    train_df = spotify_data_prep(train_df)
    test_df  = spotify_data_prep(test_df)

    # =========================
    # 5. DROP UNUSED COLS
    # =========================
    train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
    test_df  = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])

    # =========================
    # 6. SPLIT X / y
    # =========================
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    X_train = train_df.drop(TARGET, axis=1)
    X_test  = test_df.drop(TARGET, axis=1)

    # =========================
    # 7. CATEGORICAL ENCODING
    # =========================
    cat_cols, num_cols, cat_but_car = grab_col_names(X_train)

    X_train_enc, X_test_enc = encode_categoricals(
        train_df=X_train,
        test_df=X_test,
        cat_cols=cat_cols,
        rare_perc=0.01,
        drop_first=True
    )
    base_regression_models_large(X_test_enc, y_test,scoring="neg_root_mean_squared_error", top_n=3)
    # =========================
    # 8. HYPERPARAMETER OPTIMIZATION
    # =========================
    best_models = hyperparameter_optimization_regression(
        X_train_enc,
        y_train
    )

    # =========================
    # 9. VOTING REGRESSOR
    # =========================
    voting_reg = voting_regressor(
        best_models,
        X_train_enc,
        y_train,
        cv=3
    )

    # =========================
    # 10. EVALUATION
    # =========================
    y_train_pred = voting_reg.predict(X_train_enc)
    y_test_pred  = voting_reg.predict(X_test_enc)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")
    print(f"Gap       : {test_rmse - train_rmse:.4f}")

    # =========================
    # 11. SAVE MODEL
    # =========================
    joblib.dump(voting_reg, "spotify_voting_model.pkl")

    print("✅ Voting model kaydedildi.")

    return voting_reg



if __name__ == "__main__":
    print("İşlem başladı")
    main()