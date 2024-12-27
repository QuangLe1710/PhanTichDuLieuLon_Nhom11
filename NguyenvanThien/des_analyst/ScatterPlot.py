import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../Dataset_weather_daily.csv")

# Vẽ biểu đồ ScatterPlot Temp vs Humidity
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Temp", y="Humidity", color="black", s=10)

plt.title(" Biểu đồ ScatterPlot: Temp vs Humidity", fontsize=16)
plt.xlabel("Temp ", fontsize=12)
plt.ylabel("Humidity ", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Vẽ biểu đồ ScatterPlot Temp vs Windspeed
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Temp", y="Windspeed", color="black", s=10)

plt.title(" Biểu đồ ScatterPlot: Temp vs Windspeed", fontsize=16)
plt.xlabel("Temp ", fontsize=12)
plt.ylabel("Windspeed", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Vẽ biểu đồ ScatterPlot Humidity vs Windspeed
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Humidity", y="Windspeed", color="black", s=10)

plt.title(" Biểu đồ ScatterPlot: Humidity vs Windspeed", fontsize=16)
plt.xlabel("Humidity ", fontsize=12)
plt.ylabel("Windspeed ", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Vẽ biểu đồ ScatterPlot Humidity vs Pressure
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Humidity", y="Pressure", color="black", s=10)

plt.title(" Biểu đồ ScatterPlot: Humidity vs Pressure", fontsize=16)
plt.xlabel("Humidity ", fontsize=12)
plt.ylabel("Pressure", fontsize=12)

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


