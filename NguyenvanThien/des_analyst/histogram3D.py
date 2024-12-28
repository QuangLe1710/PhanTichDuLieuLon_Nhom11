import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Giả sử bạn đã có bộ dữ liệu dưới dạng DataFrame pandas
# Dữ liệu mẫu (thay thế bằng dữ liệu thực tế của bạn)
df = pd.read_csv("../Dataset_weather_daily.csv")

# Tạo figure cho biểu đồ 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Xác định các giá trị cho ba biến
x = df['Temp']
y = df['Humidity']
z = df['Windspeed']

# Vẽ histogram 3D
hist, xedges, yedges = np.histogram2d(x, y, bins=20)

# Định nghĩa các giá trị cho các trục x, y và z
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Độ cao của các cột trong histogram
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Vẽ các cột 3D
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# Thêm nhãn cho các trục
ax.set_xlabel('Temp')
ax.set_ylabel('Humidity')
ax.set_zlabel('Windspeed')

# Hiển thị biểu đồ
plt.show()
