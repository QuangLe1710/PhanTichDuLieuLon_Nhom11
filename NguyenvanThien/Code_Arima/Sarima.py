from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ví dụ: SARIMA với chu kỳ mùa vụ 365 ngày
model = SARIMAX(temperature, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
model_fit = model.fit()

# Dự báo
forecast = model_fit.forecast(steps=30)



# BUOC 1 .CHUAN BI DU LIEU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Doc du lieu file Dataset_weather_daily.csv
data = pd.read_csv("Dataset_weather_daily.csv")
data['Date'] = pd.to_datetime(data['Date'])     # Chuyển đổi định dạng thời gian
data.set_index('Date', inplace=True)        # Đặt cột Date làm index

# Lấy cột dữ liệu thời tiết Nhiệt độ ̣Temp
temperature = data['Temp']

# Vẽ đồ thị chuỗi thời gian cua Nhiet do (Temp)
plt.figure(figsize=(10, 6))
plt.plot(temperature, label='Temp')
plt.title("Biểu đồ thay đổi nhiệt độ trên chuỗi thời gian")
plt.xlabel("Date")
plt.ylabel("Temp")
plt.legend()
plt.show()


#BUOC 2. KIEM TRA TINH DUNG CUA CHUOI THOI GIAN
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Kiểm tra tính dừng ban đầu bằng kiểm định ADF
result = adfuller(temperature)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Nếu p-value > 0.05, chuỗi không dừng, cần lấy sai phân
if result[1] > 0.05:
    print("Chuỗi không dừng, thực hiện lấy sai phân bậc 1.")
    temperature_diff = temperature.diff().dropna()

    # Kiểm tra lại tính dừng sau khi lấy sai phân
    result_diff = adfuller(temperature_diff)
    print("ADF Statistic (sau khi lấy sai phân):", result_diff[0])
    print("p-value (sau khi lấy sai phân):", result_diff[1])
else:
    print("Chuỗi đã dừng.")
    temperature_diff = temperature

# Vẽ chuỗi sau khi lấy sai phân
plt.figure(figsize=(10, 6))
plt.plot(temperature_diff, label='Chênh lệch nhệt độ')
plt.title("Biểu đồ chênh lệch nhiệt độ qua thời gian")
plt.xlabel("Date")
plt.ylabel("Chênh lệch nhệt độ")
plt.legend()
plt.show()



# BUOC 3 XAC DINH THAM SO P Q D
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Vẽ đồ thị ACF và PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(temperature_diff, ax=axes[0], lags=40)
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(temperature_diff, ax=axes[1], lags=40)
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.show()

# Kiem tra tham so phu hop dua tren cac kha nang p d q đã tính toán trước đó
import itertools
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Vùng giá trị khả thi cho (p, d, q)
p_values = range(0, 3)  # p từ 0 đến 2
d_values = [1]          # d = 1 vì chuỗi đã được lấy sai phân bậc 1
q_values = range(0, 3)  # q từ 0 đến 2

# Tạo danh sách tất cả các tổ hợp tham số (p, d, q)
parameters = list(itertools.product(p_values, d_values, q_values))

# Chuỗi thời gian sau khi lấy sai phân
data = temperature_diff  # Giả sử bạn đã chuẩn bị temperature_diff

# Danh sách để lưu kết quả
results = []

# Thử từng tổ hợp (p, d, q) và tính AIC
print("Đang tính toán AIC cho từng tổ hợp tham số (p, d, q):")
for param in parameters:
    try:
        model = ARIMA(data, order=param)
        result = model.fit()
        aic = result.aic
        results.append((param, aic))
        print(f"Tham số: {param}, AIC: {aic:.2f}")
    except Exception as e:
        print(f"Tham số: {param} không khả thi. Lỗi: {str(e)}")

# Tạo DataFrame để dễ phân tích
results_df = pd.DataFrame(results, columns=['Parameters', 'AIC'])

# Chọn mô hình tốt nhất dựa trên AIC nhỏ nhất
best_model = results_df.loc[results_df['AIC'].idxmin()]

# In kết quả tốt nhất
print("\nMô hình tốt nhất:")
print(f"Tham số: {best_model['Parameters']}, AIC: {best_model['AIC']:.2f}")


# BUOC 4 XAY DUNG MO HINH SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ví dụ: SARIMA với chu kỳ mùa vụ 365 ngày
model = SARIMAX(temperature, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
model_fit = model.fit()

# Dự báo
forecast = model_fit.forecast(steps=30)

# In tóm tắt kết quả mô hình
print(model_fit.summary())



#BUOC 5 KIEM TRA DO PHU HOP CUA MO HINH
# Vẽ phần dư
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title("Residuals of the ARIMA Model")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# Kiểm tra phân phối phần dư
residuals.plot(kind='kde', title='Density Plot of Residuals')
plt.show()

# Kiểm định Ljung-Box
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box_results = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(ljung_box_results)

#BUOC 6 DU BAO
# Dự báo cho thơi gian mong muốn
# forecast_steps = input(int("Nhap thoi gian muon du bao: "))
# Dự báo cho 15 ngày tiếp theo ( Ở đây chọn dự báo 15 ngày tiếp theo )
forecast_steps = 15
forecast = model_fit.forecast(steps=forecast_steps)

# Vẽ kết quả dự báo
plt.figure(figsize=(10, 6))
plt.plot(temperature, label='Dữ liệu lịch sử ')
plt.plot(pd.date_range(temperature.index[-1], periods=forecast_steps + 1, freq='D')[1:],
         forecast, label='Forecast', color='red',linestyle='--')
plt.title("Dự báo nhiệt độ")
plt.xlabel("Date")
plt.ylabel("Temp")
plt.legend()
plt.show()

#Buoc 7 :Danh gia mo hinh
# Chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test)
train_size = int(len(temperature) * 0.8)  # 80% cho tập huấn luyện
train, test = temperature[:train_size], temperature[train_size:]
from statsmodels.tsa.arima.model import ARIMA

# Huấn luyện mô hình trên tập train
model = ARIMA(train, order=(1, 1, 2))
model_fit = model.fit()
# Dự báo trên tập kiểm tra
forecast = model_fit.forecast(steps=len(test))
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Tính toán các chỉ số
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((test - forecast) / test)) * 100

# In kết quả
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")



