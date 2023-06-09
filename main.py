import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def baseline_cancelation(raw_pulse, M):
    N = len(raw_pulse)

    overall_mean = np.mean(raw_pulse)
    detrended_signal = np.zeros(N)
    for i in range(0, N, M):

        segment = raw_pulse[i:i + M]
        segment_mean = np.mean(segment)
        detrended_segment = segment - (segment_mean - overall_mean)
        detrended_signal[i:i + M] = detrended_segment
    return detrended_signal


def average_below_threshold(data, threshold, window_size):
    averaged_data = np.copy(data)
    num_points = len(data)

    for i in range(num_points):
        if data[i] < threshold:

            start_index = max(0, i - (window_size // 2))
            end_index = min(num_points, i + (window_size // 2) + 1)

            valid_points = [point for point in data[start_index:end_index] if point >= threshold]
            if len(valid_points) > 0:
                avg_value = np.mean(valid_points)
            else:
                avg_value = threshold

            averaged_data[start_index:end_index] = avg_value

    return averaged_data


def five_point_cubic_smoothing(pulse_wave):
    smoothed_wave = np.zeros_like(pulse_wave)

    smoothed_wave[0] = (pulse_wave[0] + pulse_wave[1] + pulse_wave[2]) / 3
    smoothed_wave[1] = (pulse_wave[0] + pulse_wave[1] + pulse_wave[2] + pulse_wave[3]) / 4

    for i in range(2, len(pulse_wave) - 2):
        smoothed_wave[i] = (pulse_wave[i - 2] + pulse_wave[i - 1] + pulse_wave[i] + pulse_wave[i + 1] + pulse_wave[
            i + 2]) / 5

    smoothed_wave[-2] = (pulse_wave[-4] + pulse_wave[-3] + pulse_wave[-2] + pulse_wave[-1]) / 4
    smoothed_wave[-1] = (pulse_wave[-3] + pulse_wave[-2] + pulse_wave[-1]) / 3

    return smoothed_wave


def find_numbers_from_index(file_path, start_index, count):
    with open(file_path, 'r') as file:
        content = file.read()
        numbers = content.split(' ')
        end_index = start_index + count - 1
        if start_index <= len(numbers) and end_index <= len(numbers):
            numbers = [float(num) for num in numbers[start_index - 1:end_index]]
            return numbers
        else:
            return []


def count_peaks(interval_start, interval_length, data, distance):
    interval_data = data[interval_start:interval_start+interval_length]
    peaks, _ = find_peaks(interval_data, distance=distance)
    return len(peaks)


file_path = 'D:\code.txt'
start_index = 130*60*3
count = 130*60

ecg = find_numbers_from_index(file_path, start_index, count)

# = input("Введите имя файла с точками пульсовой волны, разделенными через пробел, в txt:")
#start_index = int(input("Частота частота дискретизации пульсовой волны равна 260 гц. Интервал пульсовой волны выбирается равным 60 секунд. Введите начальное время пульсовой волны (в секундах):"))
#start_index = (start_index+1) * 260
start_index = 230*60*3
file_path = 'D:\zzz.txt'
count = 230*60

y = find_numbers_from_index(file_path, start_index, count)

plt.title('График пульсовой волны')
plt.plot(y)
plt.show()
M = 9
pulse_smooth = average_below_threshold(y, 85, M)
pulse_smooth_a = baseline_cancelation(pulse_smooth, M)
pulse_smooth_b = five_point_cubic_smoothing(pulse_smooth_a)

interval_length = 260*30  # Длина интервала в точках
data_length_sec = 60  # Длина данных в секундах
interval_length_sec = 30  # Длина интервала в секундах
peaks_counts = []
for interval_start in range(0, data_length_sec - interval_length_sec + 1):
    interval_start_points = interval_start * (len(pulse_smooth_b) // data_length_sec)
    peaks_counts.append(count_peaks(interval_start_points, interval_length, pulse_smooth_b, 150))

plt.plot(peaks_counts, label='Пульсовая волна', color='blue')
mean = np.mean(peaks_counts)
print('Среднее значение пиков пульсовой волны:', mean)
std_dev = np.std(peaks_counts)
print('Стандартное отклонение пиков пульсовой волны:', std_dev)

interval_length = 130*30

peaks_counts = []
for interval_start in range(0, data_length_sec - interval_length_sec + 1):
    interval_start_points = interval_start * (len(ecg) // data_length_sec)
    peaks_counts.append(count_peaks(interval_start_points, interval_length, ecg, 75))

mean = np.mean(peaks_counts)
print('Среднее значение пиков пульса:', mean)
std_dev = np.std(peaks_counts)
print('Стандартное отклонение пиков пульсовой волны:', std_dev)
plt.plot(peaks_counts, label='Пульс')
plt.legend()
plt.xlabel('Номер интервала')
plt.ylabel('Кол-во пиков')
plt.show()