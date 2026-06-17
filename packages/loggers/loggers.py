import matplotlib.pyplot as plt
import numpy as np

class Logger:
    def __init__(self):
        # Включаем интерактивный режим
        plt.ion()
        
        # Создаём фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Настройка графика
        self.ax.set_xlabel("Время, с")
        self.ax.set_ylabel("Угол, рад")
        self.ax.set_title("Динамика угла маятника")
        self.ax.grid(True)
        
        # Создаём пустую линию (она будет обновляться)
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='θ(t)')
        self.ax.legend()
        
        # Показываем окно (один раз)
        plt.show(block=False)
        
        # Для хранения данных (чтобы не пересоздавать массив)
        self.trajectory = None
        self.dt = None
    
    def draw_dynamic_plot(self, trajectory, dt):
        """Обновляет график с новой траекторией"""
        
        # Сохраняем данные
        self.trajectory = trajectory
        self.dt = dt
        
        # Создаём ось времени
        time = np.arange(len(trajectory)) * dt
        
        # Обновляем данные линии
        self.line.set_data(time, trajectory)
        
        # Автоматически подстраиваем оси
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Перерисовываем окно
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Небольшая пауза для обновления
        plt.pause(0.01)
    
    def close(self):
        """Закрывает окно графика"""
        plt.close(self.fig)
        plt.ioff()