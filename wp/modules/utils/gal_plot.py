
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
sns.set(style="darkgrid")

df_gal = pd.read_csv('Gal_al_paper_data.csv')
for c in df_gal.columns[::2]:
    label = c[:-2]
    plt.plot(df_gal[f"{label}_X"], df_gal[f"{label}_Y"], label=label)
    
plt.legend()
df_gal.head(10)