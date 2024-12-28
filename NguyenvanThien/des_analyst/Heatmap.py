import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử bạn có bộ dữ liệu dạng DataFrame
# df = pd.read_csv("weather_data.csv")

df = pd.read_csv("../Dataset_weather_daily.csv")

# Chuyển đổi cột Date và tạo cột Month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Tính giá trị trung bình của Temp và Humidity theo tháng
monthly_avg = df.groupby('Month')[['Temp', 'Humidity', 'Windspeed']].mean()

# Chuyển dữ liệu sang dạng phù hợp cho heatmap
heatmap_data = monthly_avg.T  # Chuyển đổi cột và hàng

# Vẽ heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="viridis", cbar_kws={'label': 'Giá trị trung bình'})

# Tùy chỉnh biểu đồ
plt.title("Heatmap Biểu Diễn Trung Bình Temp và Humidity Theo Tháng")
plt.xlabel("Tháng")
plt.ylabel("Chỉ số")
plt.xticks(ticks=range(1, 13), labels=range(1, 13))  # Gắn nhãn các tháng
plt.show()
