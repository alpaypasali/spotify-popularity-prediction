# df = read_dataset("pythonEDA/Simple/merged_spotify_dataset.csv")
#
# features = [
#     "energy", "danceability",
#     "speechiness", "acousticness",
#     "loudness_db"
# ]
#
# df_year = (
#     df
#     .dropna(subset=["release_year"])
#     .groupby("release_year")[features]
#     .mean()
#     .rolling(window=5)
#     .mean()
# )
#
# df_year.plot(figsize=(10, 5))
# plt.title("Evolution of Audio Features Over Time (5-Year Rolling Avg)")
# plt.ylabel("Average Feature Value")
# plt.grid(alpha=0.3)
# plt.show()
#
#
#
# df["year_bin"] = pd.cut(
#     df["release_year"],
#     bins=[1960, 1975, 1990, 2005, 2015, 2025]
# )
#
# pivot = df.pivot_table(
#     index="year_bin",
#     values=features,
#     aggfunc="mean"
# )
#
# sns.heatmap(
#     pivot,
#     cmap="coolwarm",
#     annot=True,
#     fmt=".2f"
# )
# plt.title("Audio Feature Evolution by Era")
# plt.show()
#
#
#
#
# df["era"] = df["release_year"].apply(
#     lambda x: "Pre-1980" if x < 1980 else "Post-2010"
# )
#
# sns.kdeplot(
#     data=df[df["era"] == "Pre-1980"],
#     x="loudness_db",
#     label="Pre-1980",
#     fill=True
# )
# sns.kdeplot(
#     data=df[df["era"] == "Post-2010"],
#     x="loudness_db",
#     label="Post-2010",
#     fill=True
# )
#
# plt.title("Loudness Distribution Shift")
# plt.show()
#
#
#
#
#
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="tempo",
#     estimator="mean",
#     ci=None
# )
# plt.title("Tempo Evolution Over Time")
# plt.show()
#
#
#
#
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="valence",
#     estimator="mean",
#     ci=None,
#     label="Valence"
# )
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="energy",
#     estimator="mean",
#     ci=None,
#     label="Energy"
# )
# plt.legend()
# plt.title("Mood Evolution: Valence vs Energy")
# plt.show()
#
#
#
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="speechiness",
#     estimator="mean",
#     ci=None
# )
# plt.title("Speechiness Over Time")
# plt.show()
#
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="instrumentalness",
#     estimator="mean",
#     ci=None
# )
# plt.title("Instrumentalness Over Time")
# plt.show()
#
#
#
#
#
# df["modernity_score"] = (
#     df["energy"] +
#     df["danceability"] +
#     df["speechiness"] -
#     df["acousticness"]
# )
#
# sns.lineplot(
#     data=df,
#     x="release_year",
#     y="modernity_score",
#     estimator="mean",
#     ci=None
# )
# plt.title("Music Modernity Score Over Time")
# plt.show()
#
#
#
# sns.lineplot(
#     data=df.dropna(subset=["spotify_popularity_0_100"]),
#     x="release_year",
#     y="spotify_popularity_0_100",
#     estimator="mean",
#     ci=None
# )
# plt.title("Average Popularity Score Over Time")
# plt.show()
#
#
# df["era"] = pd.cut(
#     df["release_year"],
#     bins=[1960, 1980, 2000, 2010, 2025],
#     labels=["60-80", "80-00", "00-10", "10-25"]
# )
#
# sns.boxplot(
#     data=df,
#     x="era",
#     y="spotify_popularity_0_100"
# )
# plt.title("Popularity Distribution by Era")
# plt.show()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# features = ["energy", "loudness_db", "speechiness"]
#
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#
# for i, col in enumerate(features):
#     sns.scatterplot(
#         data=df,
#         x=col,
#         y="spotify_popularity_0_100",
#         alpha=0.25,
#         ax=axes[i]
#     )
#     sns.regplot(
#         data=df,
#         x=col,
#         y="spotify_popularity_0_100",
#         scatter=False,
#         color="red",
#         ax=axes[i]
#     )
#     axes[i].set_title(f"Popularity vs {col}", fontsize=11)
#     axes[i].tick_params(labelsize=9)
#
# fig.suptitle(
#     "Popularity vs Audio Features",
#     fontsize=13
# )
# plt.tight_layout()
# plt.show()
#
#
#
#
# df["era"] = pd.cut(
#     df["release_year"],
#     bins=[1960, 1980, 2000, 2010, 2025],
#     labels=["60–80", "80–00", "00–10", "10–25"]
# )
#
#
# plt.figure(figsize=(8, 5))
#
# sns.boxplot(
#     data=df,
#     x="era",
#     y="spotify_popularity_0_100"
# )
#
# plt.title("Popularity Distribution by Era", fontsize=12)
# plt.xlabel("Release Era")
# plt.ylabel("Popularity Score")
# plt.grid(axis="y", alpha=0.3)
# plt.show()
#
