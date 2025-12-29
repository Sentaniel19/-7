import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class HeavisideFourierTransform:
    """
    Исследование обобщённого преобразования Фурье функции Хевисайда
    """
    
    def __init__(self, lambda_values=None, fs=5000, T=20):
        """
        Инициализация параметров
        
        Parameters:
        lambda_values: значения λ для регуляризации
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        if lambda_values is None:
            lambda_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        self.lambda_values = lambda_values
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Частотная ось
        self.f = np.linspace(-10, 10, 2000)
    
    def heaviside_function(self, t):
        """Функция Хевисайда θ(t)"""
        return np.where(t >= 0, 1.0, 0.0)
    
    def sign_function(self, t):
        """Знаковая функция sgn(t)"""
        return np.sign(t)
    
    def heaviside_via_sign(self, t):
        """
        Выражение функции Хевисайда через знаковую функцию
        
        θ(t) = (1 + sgn(t))/2
        """
        return (1 + self.sign_function(t)) / 2
    
    def regularized_heaviside(self, t, lambda_val):
        """
        Регуляризованная функция Хевисайда
        
        θ_λ(t) = θ(t) * exp(-λ|t|), λ > 0
        Для t < 0: 0
        Для t ≥ 0: exp(-λt)
        """
        return np.where(t >= 0, np.exp(-lambda_val * t), 0.0)
    
    def analytic_fourier_regularized_heaviside(self, f, lambda_val):
        """
        Аналитическое преобразование Фурье регуляризованной функции Хевисайда
        
        F{θ(t) * exp(-λt)} = 1/(λ + i2πf) для t ≥ 0
        """
        return 1.0 / (lambda_val + 1j * 2 * np.pi * f)
    
    def compute_fourier_heaviside_cauchy(self, f):
        """
        Вычисление преобразования Фурье функции Хевисайда
        через главное значение Коши
        """
        # Обрабатываем как скаляр, так и массив
        if np.isscalar(f):
            if np.abs(f) < 1e-10:  # f = 0
                # При f=0: 1/2 δ(f)
                return 0.0
            return 1.0/(1j * 2 * np.pi * f)  # Главное значение
        else:
            # Для массива
            result = np.zeros_like(f, dtype=complex)
            mask = np.abs(f) >= 1e-10
            result[mask] = 1.0/(1j * 2 * np.pi * f[mask])
            return result
    
    def get_theoretical_result(self, f):
        """
        Теоретический результат из задания:
        
        F{θ(t)} = 1/2 [δ(f) + 1/(iπf)]
        В смысле главного значения
        """
        # Дельта-компонент при f=0
        delta_part = 0.5 * (np.abs(f) < 1e-8).astype(float)
        
        # Главное значение компонент
        pv_part = 0.5 * 1.0/(1j * np.pi * f)
        # Обработка f=0
        if np.isscalar(f):
            if np.abs(f) < 1e-10:
                pv_part = 0.0
        else:
            pv_part[np.abs(f) < 1e-10] = 0.0
        
        return delta_part + pv_part
    
    def verify_expression_via_sign(self):
        """
        Проверка выражения функции Хевисайда через знаковую функцию
        и использование результата предыдущего пункта
        """
        print("=" * 80)
        print("ВЫРАЖЕНИЕ ФУНКЦИИ ХЕВИСАЙДА ЧЕРЕЗ ЗНАКОВУЮ ФУНКЦИЮ")
        print("=" * 80)
        
        print("\nУказание из задания:")
        print("Выразить функцию Хевисайда через знаковую функцию")
        print("и воспользоваться результатом из предыдущего пункта")
        print()
        
        print("Математическое выражение:")
        print("θ(t) = (1 + sgn(t))/2")
        print()
        
        print("Из предыдущего пункта:")
        print("F{sgn(t)} = -i/(πf) (в смысле главного значения)")
        print()
        
        print("Тогда:")
        print("F{θ(t)} = F{(1 + sgn(t))/2}")
        print("        = 1/2 F{1} + 1/2 F{sgn(t)}")
        print("        = 1/2 δ(f) + 1/2 * (-i/(πf))")
        print("        = 1/2 [δ(f) + 1/(iπf)]")
        print()
        
        # Численная проверка выражения через знаковую функцию
        print("Численная проверка выражения θ(t) = (1 + sgn(t))/2:")
        print("-" * 60)
        print(f"{'t (с)':<10} {'θ(t)':<15} {'(1+sgn(t))/2':<15} {'Разница':<15}")
        print("-" * 60)
        
        t_test = np.array([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])
        
        for t_val in t_test:
            heaviside = self.heaviside_function(t_val)
            via_sign = self.heaviside_via_sign(t_val)
            diff = abs(heaviside - via_sign)
            print(f"{t_val:<10.3f} {heaviside:<15.6f} {via_sign:<15.6f} {diff:<15.6f}")
        
        print("-" * 60)
        
        # Графическая проверка
        self.plot_heaviside_via_sign()
        
        return t_test
    
    def plot_heaviside_via_sign(self):
        """Графическая проверка выражения через знаковую функцию"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        t_plot = np.linspace(-3, 3, 1000)
        
        # 1. Функции θ(t) и (1+sgn(t))/2
        ax1 = axes[0]
        
        heaviside = self.heaviside_function(t_plot)
        via_sign = self.heaviside_via_sign(t_plot)
        
        ax1.plot(t_plot, heaviside, 'b-', linewidth=3, label='θ(t)')
        ax1.plot(t_plot, via_sign, 'r--', linewidth=2, label='(1+sgn(t))/2')
        ax1.set_title('Выражение Хевисайда через знаковую функцию')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-0.2, 1.2)
        
        # 2. Разница
        ax2 = axes[1]
        
        diff = np.abs(heaviside - via_sign)
        ax2.plot(t_plot, diff, 'g-', linewidth=2)
        ax2.set_title('Разница между функциями')
        ax2.set_xlabel('Время t (с)')
        ax2.set_ylabel('|θ(t) - (1+sgn(t))/2|')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-0.01, 0.01)
        
        plt.suptitle('Проверка выражения θ(t) = (1 + sgn(t))/2', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def verify_fourier_transform(self):
        """
        Проверка преобразования Фурье функции Хевисайда
        """
        print("\n" + "=" * 80)
        print("ПРОВЕРКА ПРЕОБРАЗОВАНИЯ ФУРЬЕ ФУНКЦИИ ХЕВИСАЙДА")
        print("=" * 80)
        
        print("\nТеоретический результат из задания:")
        print("F{θ(t)} = 1/2 [δ(f) + 1/(iπf)]")
        print()
        
        print("Проверим через регуляризацию θ_λ(t) = θ(t)e^{-λt}, λ > 0")
        print("F{θ_λ(t)} = 1/(λ + i2πf)")
        print()
        
        print("Предел при λ → 0:")
        print("lim_{λ→0} 1/(λ + i2πf) = 1/(i2πf)")
        print("Но это не учитывает дельта-компонент!")
        print()
        
        print("Правильный предел:")
        print("lim_{λ→0} F{θ_λ(t)} = 1/2 δ(f) + 1/(i2πf)")
        print("                    = 1/2 [δ(f) + 1/(iπf)]")
        print()
        
        # Проверка для нескольких частот
        test_freqs = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        
        print("Сравнение при разных λ:")
        print("-" * 80)
        print(f"{'f (Гц)':<10} {'λ':<10} {'Регуляризованное':<25} {'Теоретическое':<25}")
        print("-" * 80)
        
        for f_val in test_freqs:
            for lambda_val in [0.5, 0.1, 0.05, 0.01]:
                # Регуляризованное преобразование
                reg_transform = self.analytic_fourier_regularized_heaviside(f_val, lambda_val)
                
                # Теоретическое (без дельта-компонента, т.к. f ≠ 0)
                theoretical = self.get_theoretical_result(f_val)
                
                if lambda_val == 0.5:  # Первое значение для каждой частоты
                    print(f"{f_val:<10.3f} {lambda_val:<10.3f} "
                          f"{reg_transform.real:+.6f}{reg_transform.imag:+.6f}j  "
                          f"{theoretical.real:+.6f}{theoretical.imag:+.6f}j")
                else:
                    print(f"{' ':<10} {lambda_val:<10.3f} "
                          f"{reg_transform.real:+.6f}{reg_transform.imag:+.6f}j  "
                          f"{' ':<25}")
            print()
        
        print("-" * 80)
        
        # Особый случай f = 0
        print("\nОсобый случай f = 0:")
        print("Теоретически: F{θ(t)}(0) = 1/2 δ(0) = ∞ (дельта-функция)")
        print("При регуляризации: lim_{λ→0} 1/(λ + i0) = 1/λ → ∞")
        print()
        
        # Графическая иллюстрация
        self.plot_fourier_transform_convergence()
    
    def plot_fourier_transform_convergence(self):
        """Графическая иллюстрация сходимости преобразования Фурье"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Регуляризованные функции Хевисайда
        ax1 = axes[0, 0]
        t_plot = np.linspace(-2, 5, 1000)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.lambda_values)))
        
        for i, lambda_val in enumerate(self.lambda_values):
            reg_heaviside = self.regularized_heaviside(t_plot, lambda_val)
            ax1.plot(t_plot, reg_heaviside, color=colors[i], 
                    linewidth=2, alpha=0.7, label=f'λ = {lambda_val}')
        
        # Идеальная функция Хевисайда
        ideal_heaviside = self.heaviside_function(t_plot)
        ax1.plot(t_plot, ideal_heaviside, 'k--', linewidth=3, 
                label='θ(t) (идеальная)')
        
        ax1.set_title('Регуляризованные функции Хевисайда')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('θ(t)e^{-λt}')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, 3)
        ax1.set_ylim(-0.2, 1.2)
        
        # 2. Мнимая часть преобразования Фурье
        ax2 = axes[0, 1]
        f_plot = np.linspace(-5, 5, 1000)
        
        for i, lambda_val in enumerate(self.lambda_values[:5]):  # Первые 5 для наглядности
            fourier_reg = self.analytic_fourier_regularized_heaviside(f_plot, lambda_val)
            ax2.plot(f_plot, fourier_reg.imag, color=colors[i], 
                    linewidth=2, alpha=0.7, label=f'λ = {lambda_val}')
        
        # Теоретическая мнимая часть (без дельта-компонента)
        theoretical = self.get_theoretical_result(f_plot)
        ax2.plot(f_plot, theoretical.imag, 'k--', linewidth=3, 
                label='Теоретическая: -1/(2πf)')
        
        ax2.set_title('Мнимая часть преобразования Фурье')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('Im{F{θ_λ(t)}}')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-5, 5)
        
        # 3. Действительная часть преобразования Фурье
        ax3 = axes[0, 2]
        
        for i, lambda_val in enumerate(self.lambda_values[:5]):
            fourier_reg = self.analytic_fourier_regularized_heaviside(f_plot, lambda_val)
            ax3.plot(f_plot, fourier_reg.real, color=colors[i], 
                    linewidth=2, alpha=0.7, label=f'λ = {lambda_val}')
        
        # Теоретическая действительная часть (дельта-компонент при f=0)
        # Для визуализации используем узкий гауссов импульс
        sigma = 0.1
        delta_approx = np.exp(-(f_plot**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        ax3.plot(f_plot, 0.5 * delta_approx, 'k--', linewidth=3, 
                label='Теоретическая: 1/2 δ(f)')
        
        ax3.set_title('Действительная часть преобразования Фурье')
        ax3.set_xlabel('Частота f (Гц)')
        ax3.set_ylabel('Re{F{θ_λ(t)}}')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-1, 1)
        
        # 4. Сходимость в точке f = 1 (мнимая часть)
        ax4 = axes[1, 0]
        
        f_test = 1.0
        values_imag = []
        
        for lambda_val in self.lambda_values:
            fourier_val = self.analytic_fourier_regularized_heaviside(f_test, lambda_val)
            values_imag.append(fourier_val.imag)
        
        theoretical_imag = self.get_theoretical_result(f_test).imag
        
        ax4.semilogx(self.lambda_values, values_imag, 'bo-', 
                    linewidth=2, markersize=8, label='Im{F{θ_λ}}(f=1)')
        ax4.axhline(y=theoretical_imag, color='r', linestyle='--', 
                   label=f'Теоретическая: {theoretical_imag:.4f}')
        ax4.set_xlabel('λ')
        ax4.set_ylabel('Im{F{θ_λ}}(1)')
        ax4.set_title('Сходимость мнимой части при f = 1 Гц')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.invert_xaxis()  # λ уменьшается слева направо
        
        # 5. Сходимость в точке f = 0 (действительная часть)
        ax5 = axes[1, 1]
        
        f_test = 0.0
        values_real = []
        
        for lambda_val in self.lambda_values:
            fourier_val = self.analytic_fourier_regularized_heaviside(f_test, lambda_val)
            values_real.append(fourier_val.real)
        
        ax5.semilogx(self.lambda_values, values_real, 'ro-', 
                    linewidth=2, markersize=8, label='Re{F{θ_λ}}(f=0)')
        ax5.set_xlabel('λ')
        ax5.set_ylabel('Re{F{θ_λ}}(0)')
        ax5.set_title('Расходимость действительной части при f = 0')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.invert_xaxis()
        
        # 6. Амплитудный спектр
        ax6 = axes[1, 2]
        
        for i, lambda_val in enumerate(self.lambda_values[:5]):
            fourier_reg = self.analytic_fourier_regularized_heaviside(f_plot, lambda_val)
            ax6.plot(f_plot, np.abs(fourier_reg), color=colors[i], 
                    linewidth=2, alpha=0.7, label=f'λ = {lambda_val}')
        
        # Теоретический амплитудный спектр
        # Учитываем дельта-компонент при f=0
        theoretical_amp = np.abs(theoretical)
        # Добавляем дельта-аппроксимацию
        theoretical_amp += 0.5 * delta_approx
        
        ax6.plot(f_plot, theoretical_amp, 'k--', linewidth=3, 
                label='Теоретическая')
        
        ax6.set_title('Амплитудный спектр')
        ax6.set_xlabel('Частота f (Гц)')
        ax6.set_ylabel('|F{θ_λ(t)}|')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(-2, 2)
        ax6.set_ylim(0, 5)
        
        plt.suptitle('Сходимость преобразований Фурье функции Хевисайда при λ → 0', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def analyze_delta_component(self):
        """
        Анализ дельта-компонента в преобразовании Фурье
        """
        print("\n" + "=" * 80)
        print("АНАЛИЗ ДЕЛЬТА-КОМПОНЕНТА В ПРЕОБРАЗОВАНИИ ФУРЬЕ")
        print("=" * 80)
        
        print("\nПочему появляется дельта-компонент 1/2 δ(f)?")
        print()
        
        print("1. Постоянная составляющая функции Хевисайда:")
        print("   Среднее значение θ(t) на интервале [-T, T]:")
        print("   (1/(2T)) ∫_{-T}^{T} θ(t) dt = (1/(2T)) ∫_{0}^{T} 1 dt = 1/2")
        print("   При T → ∞ среднее значение стремится к 1/2")
        print()
        
        print("2. Связь с преобразованием Фурье постоянной:")
        print("   F{1/2} = 1/2 δ(f)")
        print()
        
        print("3. Альтернативный вывод через предел:")
        print("   θ(t) = lim_{λ→0} [1/2 + (1/2)sgn(t)e^{-λ|t|}]")
        print("   F{θ(t)} = lim_{λ→0} [1/2 δ(f) + 1/2 F{sgn(t)e^{-λ|t|}}]")
        print("           = 1/2 δ(f) + 1/2 * (-i/(πf))")
        print("           = 1/2 [δ(f) + 1/(iπf)]")
        print()
        
        # Численная демонстрация постоянной составляющей
        print("Численная демонстрация постоянной составляющей:")
        print("-" * 60)
        print(f"{'Интервал T':<15} {'Среднее значение':<20}")
        print("-" * 60)
        
        T_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        
        for T_val in T_values:
            t_vals = np.linspace(-T_val, T_val, 10000)
            heaviside_vals = self.heaviside_function(t_vals)
            mean_value = np.mean(heaviside_vals)
            print(f"[-{T_val:.1f}, {T_val:.1f}]      {mean_value:<20.6f}")
        
        print("-" * 60)
        print("Предел при T → ∞: 0.5")
        
        # Графическая иллюстрация
        self.plot_delta_component_analysis()
    
    def plot_delta_component_analysis(self):
        """Графический анализ дельта-компонента"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Функция Хевисайда и её постоянная составляющая
        ax1 = axes[0, 0]
        
        t_plot = np.linspace(-5, 5, 1000)
        heaviside = self.heaviside_function(t_plot)
        
        ax1.plot(t_plot, heaviside, 'b-', linewidth=3, label='θ(t)')
        ax1.axhline(y=0.5, color='r', linestyle='--', 
                   linewidth=2, label='Среднее значение = 1/2')
        ax1.fill_between(t_plot, 0, heaviside, alpha=0.3, color='b')
        
        ax1.set_title('Функция Хевисайда и её постоянная составляющая')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('θ(t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-0.2, 1.2)
        
        # 2. Среднее значение на интервале [-T, T]
        ax2 = axes[0, 1]
        
        T_values = np.logspace(0, 2, 50)  # от 1 до 100
        mean_values = []
        
        for T_val in T_values:
            t_vals = np.linspace(-T_val, T_val, 10000)
            heaviside_vals = self.heaviside_function(t_vals)
            mean_values.append(np.mean(heaviside_vals))
        
        ax2.semilogx(T_values, mean_values, 'g-', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', 
                   label='Предел: 1/2')
        ax2.set_xlabel('Длина интервала T')
        ax2.set_ylabel('Среднее значение θ(t)')
        ax2.set_title('Сходимость среднего значения к 1/2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Разложение на чётную и нечётную части
        ax3 = axes[1, 0]
        
        # Чётная часть: 1/2
        even_part = 0.5 * np.ones_like(t_plot)
        
        # Нечётная часть: (1/2)sgn(t)
        odd_part = 0.5 * self.sign_function(t_plot)
        
        ax3.plot(t_plot, even_part, 'b-', linewidth=2, label='Чётная часть: 1/2')
        ax3.plot(t_plot, odd_part, 'r-', linewidth=2, label='Нечётная часть: (1/2)sgn(t)')
        ax3.plot(t_plot, heaviside, 'k--', linewidth=3, label='θ(t) = сумма')
        
        ax3.set_title('Разложение на чётную и нечётную части')
        ax3.set_xlabel('Время t (с)')
        ax3.set_ylabel('Амплитуда')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-0.6, 1.2)
        
        # 4. Спектры чётной и нечётной частей
        ax4 = axes[1, 1]
        
        f_plot = np.linspace(-5, 5, 1000)
        
        # Спектр чётной части: F{1/2} = 1/2 δ(f)
        sigma = 0.1
        delta_approx = np.exp(-(f_plot**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        even_spectrum = 0.5 * delta_approx
        
        # Спектр нечётной части: F{(1/2)sgn(t)} = -i/(2πf)
        odd_spectrum = -1j/(2 * np.pi * f_plot)
        odd_spectrum[np.abs(f_plot) < 1e-10] = 0
        
        ax4.plot(f_plot, even_spectrum, 'b-', linewidth=2, label='F{1/2} = 1/2 δ(f)')
        ax4.plot(f_plot, np.imag(odd_spectrum), 'r-', linewidth=2, 
                label='Im{F{(1/2)sgn(t)}} = -1/(2πf)')
        
        ax4.set_title('Спектры чётной и нечётной частей')
        ax4.set_xlabel('Частота f (Гц)')
        ax4.set_ylabel('Амплитуда')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-2, 2)
        
        plt.suptitle('Анализ дельта-компонента в преобразовании Фурье', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def verify_inverse_transform(self):
        """
        Проверка обратного преобразования Фурье
        """
        print("\n" + "=" * 80)
        print("ПРОВЕРКА ОБРАТНОГО ПРЕОБРАЗОВАНИЯ ФУРЬЕ")
        print("=" * 80)
        
        print("\nТеоретически:")
        print("F{θ(t)} = 1/2 [δ(f) + 1/(iπf)]")
        print("Обратное преобразование:")
        print("F^{-1}{1/2 [δ(f) + 1/(iπf)]} = θ(t)")
        print()
        
        print("Проверим численно с регуляризацией:")
        
        # Временные точки для проверки
        t_test = np.array([-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0])
        
        # Малый параметр регуляризации
        lambda_small = 0.01
        
        print("\n" + "-" * 80)
        print(f"{'t (с)':<10} {'θ(t)':<15} {'Обратное преобразование':<25} {'Ошибка':<15}")
        print("-" * 80)
        
        for t_val in t_test:
            # Исходная функция (регуляризованная)
            original = self.regularized_heaviside(t_val, lambda_small)
            
            # Обратное преобразование Фурье (численно)
            # F^{-1}{1/(λ + i2πf)} = θ(t)e^{-λt}
            
            # Интеграл обратного преобразования
            f_vals = np.linspace(-200, 200, 20000)
            
            # Регуляризованное преобразование
            fourier_vals = self.analytic_fourier_regularized_heaviside(f_vals, lambda_small)
            
            # Обратное преобразование
            integrand = fourier_vals * np.exp(1j * 2 * np.pi * f_vals * t_val)
            inverse = np.trapz(integrand, f_vals)
            
            error = abs(inverse - original)
            
            print(f"{t_val:<10.3f} {original:<15.6f} {inverse.real:<25.6f} {error:<15.6f}")
        
        print("-" * 80)
        
        # Графическая проверка
        self.plot_inverse_transform_check()
    
    def plot_inverse_transform_check(self):
        """Графическая проверка обратного преобразования"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Параметры
        lambda_val = 0.05
        t_vals = np.linspace(-2, 3, 500)
        
        # 1. Прямое и обратное преобразование
        ax1 = axes[0]
        
        # Исходная регуляризованная функция
        original = self.regularized_heaviside(t_vals, lambda_val)
        
        # Численное обратное преобразование
        f_vals = np.linspace(-500, 500, 40000)
        
        # Предварительно вычисляем преобразование Фурье
        fourier_vals = self.analytic_fourier_regularized_heaviside(f_vals, lambda_val)
        
        # Обратное преобразование для каждого t
        inverse_vals = np.zeros_like(t_vals, dtype=complex)
        for i, t_val in enumerate(t_vals):
            integrand = fourier_vals * np.exp(1j * 2 * np.pi * f_vals * t_val)
            inverse_vals[i] = np.trapz(integrand, f_vals)
        
        ax1.plot(t_vals, original, 'b-', linewidth=3, label='θ(t)e^{-λt}')
        ax1.plot(t_vals, inverse_vals.real, 'r--', linewidth=2, 
                label='F^{-1}{F{θ(t)e^{-λt}}}')
        ax1.set_title(f'Проверка обратного преобразования (λ = {lambda_val})')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-2, 3)
        ax1.set_ylim(-0.2, 1.2)
        
        # 2. Ошибка
        ax2 = axes[1]
        
        error = np.abs(inverse_vals.real - original)
        ax2.semilogy(t_vals, error, 'g-', linewidth=2)
        ax2.set_title('Ошибка восстановления')
        ax2.set_xlabel('Время t (с)')
        ax2.set_ylabel('Ошибка')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-2, 3)
        
        plt.suptitle('Проверка корректности преобразования Фурье', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 7")
    print("Обобщённое преобразование Фурье функции Хевисайда")
    print("=" * 80)
    
    # Создаем анализатор
    analyzer = HeavisideFourierTransform(
        lambda_values=[2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
        fs=10000,
        T=20
    )
    
    # 1. Проверка выражения через знаковую функцию
    print("\n1. Проверка выражения θ(t) = (1 + sgn(t))/2...")
    t_test = analyzer.verify_expression_via_sign()
    
    # 2. Проверка преобразования Фурье
    print("\n2. Проверка преобразования Фурье функции Хевисайда...")
    analyzer.verify_fourier_transform()
    
    # 3. Анализ дельта-компонента
    print("\n3. Анализ дельта-компонента в преобразовании...")
    analyzer.analyze_delta_component()
    
    # 4. Проверка обратного преобразования
    print("\n4. Проверка обратного преобразования...")
    analyzer.verify_inverse_transform()
    
    # 5. Итоговые выводы
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ ПО ЗАДАНИЮ 7")
    print("=" * 80)
    
    print("\n1. Выражение через знаковую функцию:")
    print("   θ(t) = (1 + sgn(t))/2")
    print()
    
    print("2. Использование результата из пункта 6:")
    print("   F{sgn(t)} = -i/(πf) (в смысле главного значения)")
    print()
    
    print("3. Преобразование Фурье функции Хевисайда:")
    print("   F{θ(t)} = 1/2 F{1} + 1/2 F{sgn(t)}")
    print("           = 1/2 δ(f) + 1/2 * (-i/(πf))")
    print("           = 1/2 [δ(f) + 1/(iπf)]")
    print()
    
    print("4. Физическая интерпретация:")
    print("   - Дельта-компонент 1/2 δ(f): постоянная составляющая = 1/2")
    print("   - Главное значение 1/(iπf): нечётная часть функции")
    print("   - Полный спектр: сумма этих двух компонентов")
    print()
    
    print("5. Свойства полученного преобразования:")
    print("   - Имеет дельта-компонент при f = 0")
    print("   - Мнимая часть: -1/(2πf) (нечётная функция)")
    print("   - Особенность (полюс) при f = 0")
    print("   - Правильно восстанавливается обратным преобразованием")
    print()
    
    print("6. Практическое значение:")
    print("   - Функция Хевисайда используется для описания включения")
    print("   - Её преобразование важно в теории линейных систем")
    print("   - Дельта-компонент соответствует постоянному смещению")
    print()
    
    print("=" * 80)
    print("ЗАДАНИЕ ВЫПОЛНЕНО ПОЛНОСТЬЮ!")
    print("=" * 80)
    
    # Дополнительно: математический вывод
    print("\n" + "=" * 80)
    print("МАТЕМАТИЧЕСКИЙ ВЫВОД (для справки)")
    print("=" * 80)
    
    print("\nПолный вывод преобразования Фурье функции Хевисайда:")
    print("1. Представление через знаковую функцию:")
    print("   θ(t) = (1 + sgn(t))/2")
    print()
    
    print("2. Преобразование Фурье постоянной:")
    print("   F{1} = δ(f)")
    print()
    
    print("3. Преобразование Фурье знаковой функции (из пункта 6):")
    print("   F{sgn(t)} = -i/(πf) (в смысле p.v.)")
    print()
    
    print("4. Линейность преобразования Фурье:")
    print("   F{θ(t)} = F{(1 + sgn(t))/2}")
    print("           = 1/2 F{1} + 1/2 F{sgn(t)}")
    print("           = 1/2 δ(f) + 1/2 * (-i/(πf))")
    print("           = 1/2 [δ(f) + 1/(iπf)]")
    print()
    
    print("5. Альтернативный вывод через регуляризацию:")
    print("   θ_λ(t) = θ(t)e^{-λt}, λ > 0")
    print("   F{θ_λ(t)} = ∫_{0}^{∞} e^{-λt} e^{-i2πft} dt")
    print("             = 1/(λ + i2πf)")
    print("   lim_{λ→0} F{θ_λ(t)} = 1/(i2πf) + 1/2 δ(f)")
    print("   (нужно аккуратно учесть предел в нуле)")

if __name__ == "__main__":
    main()
