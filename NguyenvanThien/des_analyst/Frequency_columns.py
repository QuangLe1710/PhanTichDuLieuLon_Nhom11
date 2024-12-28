import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../Dataset_weather_daily.csv")

# Lặp qua từng cột của DataFrame và vẽ histogram cho các cột kiểu nguyên
for column in df.select_dtypes(include=['number']):
    plt.hist(df[column], bins=10, edgecolor='k')
    plt.title(f'Biểu đồ tần suất {column}')
    plt.xlabel(column)
    plt.ylabel('Tần suất')
    plt.show()