import pandas as pd

df = pd.read_csv("../Dataset_weather_daily.csv")


# Tạo bảng thống kê số giá trị khuyết thiếu và giá trị trùng lặp
summary_table = pd.DataFrame({
    "Column": df.columns,
    "Gia tri khuyet thieu": df.isnull().sum().values,
    "Gia tri trung lap": [df[col].duplicated().sum() for col in df.columns]
})

# Hiển thị bảng
print(summary_table)
