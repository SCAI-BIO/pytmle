# %%
import matplotlib.pyplot as plt
import pandas as pd


# %%
df = pd.read_csv("hodgkins_disease.csv")

# %%
# cross table over chemotherapy and event
pd.crosstab(df["chemo"], df["status"], margins=True)

# %%
# distribution of follow-up times
labels = {0: "censored", 1: "relapsed", 2: "deceased"}
fig, ax = plt.subplots(figsize=(8,6))
for label, group in df.groupby('status'):
    group.time.plot(kind="kde", ax=ax, label=labels[label])
plt.legend()
ax.set_xlabel("Follow-up time")
ax.set_ylabel("Density")
ax.set_title("Distribution of follow-up times by event type")

# %%
# proportion of missing data
df.isnull().sum() / len(df)
# no missing data in any column, so no imputation is needed

# %% 
# descriptive statistics for baseline covariates
def compute_statistics(df):
    stats_cont = df[["age"]].describe(include="all").T
    stats_cat = (df[["female", "extranod", "stage2", "medwidsi_S", "medwidsi_N"]] == 1).sum().to_frame(name="count")
    stats_cat["percent"] = stats_cat["count"] / len(df) * 100
    return stats_cont, stats_cat

compute_statistics(df)

# %%
# descriptive statistics by treatment group
print(compute_statistics(df[df["chemo"] == 0]))
print(compute_statistics(df[df["chemo"] == 1]))

# %%
# at-risk population over time
time_points = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
for t in time_points:
    at_risk = (df['time'] >= t).sum()
    at_risk_chemo = (df[(df['time'] >= t) & (df['chemo'] == 1)]).shape[0]
    at_risk_control = (df[(df['time'] >= t) & (df['chemo'] == 0)]).shape[0]
    print(f"At t={t}: {at_risk} patients at risk (chemo: {at_risk_chemo}, control: {at_risk_control})")
