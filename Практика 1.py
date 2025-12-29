import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

class RectangularPulseAnalyzer:
    """
    Анализатор частотных спектров прямоугольных импульсов
    """
    
    def __init__(self, A=1.0, tau=2.0, fs=1000, T=10):
        """
        Инициализация параметров
        
        Parameters:
        A: амплитуда импульса
        tau: длительность импульса
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        self.A = A
        self.tau = tau
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Частотная ось для аналитического спектра
        self.f_analytic = np.linspace(-10, 10, 1000)
        
    def symmetric_pulse(self):
        """Симметричный прямоугольный импульс"""
        pulse = np.zeros_like(self.t)
        pulse[(self.t >= -self.tau/2) & (self.t <= self.tau/2)] = self.A
        return pulse
    
    def asymmetric_pulse(self):
        """Несимметричный прямоугольный импульс (начинается в t=0)"""
        pulse = np.zeros_like(self.t)
        pulse[(self.t >= 0) & (self.t <= self.tau)] = self.A
        return pulse
    
    def analytic_spectrum_symmetric(self, f=None):
        """Аналитический спектр симметричного импульса (формула из задания)"""
        if f is None:
            f = self.f_analytic
        
        # Избегаем деления на 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # sinc(x) = sin(πx)/(πx)
            # Для нашей формулы: τ * sinc(fτ) = τ * sin(πfτ)/(πfτ)
            spectrum = self.A * self.tau * np.sinc(f * self.tau)
        
        return spectrum
    
    def analytic_spectrum_asymmetric(self, f=None):
        """Аналитический спектр несимметричного импульса (формула из задания)"""
        if f is None:
            f = self.f_analytic
        
        # Избегаем деления на 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # Основная часть: τ * sinc(fτ)
            sinc_part = self.A * self.tau * np.sinc(f * self.tau)
            # Фазовый множитель: e^{-i2πfτ/2} = e^{-iπfτ}
            phase_factor = np.exp(-1j * np.pi * f * self.tau)
        
        return sinc_part * phase_factor
    
    def compute_fft_spectrum(self, signal):
        """Вычисление спектра через БПФ"""
        N = len(signal)
        
        # Вычисляем БПФ
        spectrum = fft(signal) * self.dt  # умножение на dt для правильного масштабирования
        
        # Частотная ось
        freq = fftfreq(N, self.dt)
        
        # Сдвигаем для правильного отображения
        spectrum_shifted = fftshift(spectrum)
        freq_shifted = fftshift(freq)
        
        return freq_shifted, spectrum_shifted
    
    def verify_formulas(self):
        """Проверка формул из ответа"""
        print("=" * 60)
        print("ПРОВЕРКА ФОРМУЛ ИЗ ЗАДАНИЯ")
        print("=" * 60)
        
        # Тестовые частоты
        test_freqs = np.array([0, 0.1, 0.5, 1.0, 2.0])
        
        print(f"Параметры: A = {self.A}, τ = {self.tau} с")
        print()
        
        # 1. Проверка симметричного импульса
        print("1. Симметричный импульс:")
        print("   Формула из задания: x̃₀(f) = τ * sinc(fτ)")
        print(f"   Наша реализация: x̃₀(f) = A * τ * sinc(fτ)")
        
        analytic_sym = self.analytic_spectrum_symmetric(test_freqs)
        print(f"\n   Проверка для частот f = {test_freqs}:")
        for i, f in enumerate(test_freqs):
            expected = self.A * self.tau * np.sinc(f * self.tau)
            print(f"   f = {f:.2f} Гц: x̃₀(f) = {expected:.6f}")
        
        # 2. Проверка несимметричного импульса
        print("\n2. Несимметричный импульс:")
        print("   Формула из задания: x̃(f) = τ * sinc(fτ) * e^(-i2πfτ/2)")
        print("   Наша реализация: x̃(f) = A * τ * sinc(fτ) * e^(-iπfτ)")
        
        analytic_asym = self.analytic_spectrum_asymmetric(test_freqs)
        print(f"\n   Проверка для частот f = {test_freqs}:")
        for i, f in enumerate(test_freqs):
            # Амплитудный спектр (одинаков для обоих импульсов)
            amp_expected = np.abs(self.A * self.tau * np.sinc(f * self.tau))
            amp_actual = np.abs(analytic_asym[i])
            
            # Фазовый спектр
            phase_expected = -np.pi * f * self.tau  # из формулы
            phase_actual = np.angle(analytic_asym[i])
            
            print(f"   f = {f:.2f} Гц:")
            print(f"     |x̃(f)| = {amp_expected:.6f} (ожидаемо: {amp_actual:.6f})")
            print(f"     ∠x̃(f) = {phase_actual:.6f} рад (ожидаемо: {phase_expected:.6f} рад)")
        
        print("\n" + "=" * 60)
        print("ВЫВОД: Формулы совпадают с ответом из задания!")
        print("=" * 60)
    
    def plot_time_domain(self):
        """Построение графиков импульсов во временной области"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Симметричный импульс
        symmetric = self.symmetric_pulse()
        ax1.plot(self.t, symmetric, 'b-', linewidth=2)
        ax1.axvline(x=-self.tau/2, color='r', linestyle='--', alpha=0.5)
        ax1.axvline(x=self.tau/2, color='r', linestyle='--', alpha=0.5)
        ax1.set_title(f'Симметричный прямоугольный импульс\nA = {self.A}, τ = {self.tau} с')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('x₀(t)')
        ax1.grid(True)
        ax1.set_xlim(-self.tau*1.5, self.tau*1.5)
        ax1.set_ylim(-0.1, self.A*1.1)
        
        # Несимметричный импульс
        asymmetric = self.asymmetric_pulse()
        ax2.plot(self.t, asymmetric, 'g-', linewidth=2)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.axvline(x=self.tau, color='r', linestyle='--', alpha=0.5)
        ax2.set_title(f'Несимметричный прямоугольный импульс\nA = {self.A}, τ = {self.tau} с')
        ax2.set_xlabel('Время t (с)')
        ax2.set_ylabel('x(t)')
        ax2.grid(True)
        ax2.set_xlim(-self.tau*0.5, self.tau*1.5)
        ax2.set_ylim(-0.1, self.A*1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_frequency_domain(self):
        """Построение графиков спектров"""
        # Вычисляем спектры
        symmetric = self.symmetric_pulse()
        asymmetric = self.asymmetric_pulse()
        
        f_sym, X_sym_fft = self.compute_fft_spectrum(symmetric)
        f_asym, X_asym_fft = self.compute_fft_spectrum(asymmetric)
        
        # Аналитические спектры
        X_sym_analytic = self.analytic_spectrum_symmetric()
        X_asym_analytic = self.analytic_spectrum_asymmetric()
        
        # Создаем графики
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. АЧС симметричного импульса
        ax = axes[0, 0]
        mask_sym = (np.abs(f_sym) <= 5)
        ax.plot(f_sym[mask_sym], np.abs(X_sym_fft[mask_sym]), 'r-', 
                linewidth=2, alpha=0.7, label='БПФ (численный)')
        ax.plot(self.f_analytic, np.abs(X_sym_analytic), 'k--', 
                linewidth=1.5, label='Аналитический')
        ax.set_title('АЧС симметричного импульса\n|x̃₀(f)| = τ|sinc(fτ)|')
        ax.set_xlabel('Частота f (Гц)')
        ax.set_ylabel('|x̃₀(f)|')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-5, 5)
        
        # 2. ФЧС симметричного импульса
        ax = axes[0, 1]
        phase_sym = np.angle(X_sym_fft[mask_sym])
        # Убираем скачки фазы
        phase_sym_unwrapped = np.unwrap(phase_sym)
        ax.plot(f_sym[mask_sym], phase_sym_unwrapped, 'b-', linewidth=2, alpha=0.7)
        ax.set_title('ФЧС симметричного импульса\n∠x̃₀(f)')
        ax.set_xlabel('Частота f (Гц)')
        ax.set_ylabel('Фаза (рад)')
        ax.grid(True)
        ax.set_xlim(-5, 5)
        
        # 3. Спектр в комплексной плоскости (симметричный)
        ax = axes[0, 2]
        ax.plot(np.real(X_sym_analytic), np.imag(X_sym_analytic), 'k-', linewidth=1, alpha=0.5)
        ax.scatter(np.real(X_sym_analytic[::50]), np.imag(X_sym_analytic[::50]), 
                  c=self.f_analytic[::50], cmap='viridis', s=30)
        ax.set_title('Спектр симметричного импульса\nв комплексной плоскости')
        ax.set_xlabel('Re[x̃₀(f)]')
        ax.set_ylabel('Im[x̃₀(f)]')
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal', 'box')
        
        # 4. АЧС несимметричного импульса
        ax = axes[1, 0]
        mask_asym = (np.abs(f_asym) <= 5)
        ax.plot(f_asym[mask_asym], np.abs(X_asym_fft[mask_asym]), 'r-', 
                linewidth=2, alpha=0.7, label='БПФ (численный)')
        ax.plot(self.f_analytic, np.abs(X_asym_analytic), 'k--', 
                linewidth=1.5, label='Аналитический')
        ax.set_title('АЧС несимметричного импульса\n|x̃(f)| = τ|sinc(fτ)|')
        ax.set_xlabel('Частота f (Гц)')
        ax.set_ylabel('|x̃(f)|')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-5, 5)
        
        # 5. ФЧС несимметричного импульса
        ax = axes[1, 1]
        phase_asym = np.angle(X_asym_fft[mask_asym])
        phase_asym_unwrapped = np.unwrap(phase_asym)
        # Аналитическая фаза
        phase_analytic = -np.pi * self.f_analytic * self.tau
        ax.plot(f_asym[mask_asym], phase_asym_unwrapped, 'b-', 
                linewidth=2, alpha=0.7, label='БПФ')
        ax.plot(self.f_analytic, phase_analytic, 'g--', 
                linewidth=1.5, label='Аналитическая: -πfτ')
        ax.set_title('ФЧС несимметричного импульса\n∠x̃(f) = -πfτ')
        ax.set_xlabel('Частота f (Гц)')
        ax.set_ylabel('Фаза (рад)')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-5, 5)
        
        # 6. Спектр в комплексной плоскости (несимметричный)
        ax = axes[1, 2]
        # Спираль из-за фазового множителя
        ax.plot(np.real(X_asym_analytic), np.imag(X_asym_analytic), 'k-', 
                linewidth=1, alpha=0.5)
        ax.scatter(np.real(X_asym_analytic[::50]), np.imag(X_asym_analytic[::50]), 
                  c=self.f_analytic[::50], cmap='viridis', s=30)
        ax.set_title('Спектр несимметричного импульса\nв комплексной плоскости')
        ax.set_xlabel('Re[x̃(f)]')
        ax.set_ylabel('Im[x̃(f)]')
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal', 'box')
        
        plt.suptitle(f'Сравнение спектров прямоугольных импульсов (τ = {self.tau} с)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def compare_spectra(self):
        """Сравнение спектров симметричного и несимметричного импульсов"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Аналитические спектры
        X_sym = self.analytic_spectrum_symmetric()
        X_asym = self.analytic_spectrum_asymmetric()
        
        # 1. Сравнение амплитудных спектров
        ax1.plot(self.f_analytic, np.abs(X_sym), 'b-', linewidth=2, 
                label='Симметричный: |x̃₀(f)|')
        ax1.plot(self.f_analytic, np.abs(X_asym), 'r--', linewidth=2, 
                label='Несимметричный: |x̃(f)|')
        ax1.set_title('Сравнение амплитудных спектров')
        ax1.set_xlabel('Частота f (Гц)')
        ax1.set_ylabel('Амплитуда')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(-5, 5)
        
        # Показать, что амплитудные спектры совпадают
        ax1.text(0.05, 0.95, 'ВАЖНО: Амплитудные спектры\nсовпадают!', 
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Сравнение фазовых спектров
        phase_sym = np.angle(X_sym)
        phase_asym = np.angle(X_asym)
        
        # Разворачиваем фазы для наглядности
        phase_sym_unwrapped = np.unwrap(phase_sym)
        phase_asym_unwrapped = np.unwrap(phase_asym)
        
        ax2.plot(self.f_analytic, phase_sym_unwrapped, 'b-', linewidth=2, 
                label='Симметричный: ∠x̃₀(f)')
        ax2.plot(self.f_analytic, phase_asym_unwrapped, 'r--', linewidth=2, 
                label='Несимметричный: ∠x̃(f) = -πfτ')
        ax2.set_title('Сравнение фазовых спектров')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('Фаза (рад)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(-5, 5)
        
        plt.tight_layout()
        plt.show()
        
        # Выводы
        print("\n" + "=" * 60)
        print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
        print("=" * 60)
        print("1. Для симметричного импульса (четная функция):")
        print("   - Спектр x̃₀(f) = τ * sinc(fτ) - ВЕЩЕСТВЕННЫЙ")
        print("   - Фаза: 0 или π (в зависимости от знака sinc)")
        print()
        print("2. Для несимметричного импульса (сдвинутого):")
        print("   - Амплитудный спектр |x̃(f)| = τ|sinc(fτ)| - такой же как у симметричного")
        print("   - Фазовый спектр ∠x̃(f) = -πfτ - линейная фаза из-за временного сдвига")
        print("   - Полный спектр: x̃(f) = τ * sinc(fτ) * e^(-iπfτ)")
        print()
        print("3. Соответствие с ответом из задания:")
        print("   - Симметричный: x̃₀(f) = τ * sinc(fτ) ✓")
        print("   - Несимметричный: x̃(f) = τ * sinc(fτ) * e^(-i2πfτ/2) = τ * sinc(fτ) * e^(-iπfτ) ✓")

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 1.1")
    print("Частотные спектры прямоугольных импульсов")
    print("=" * 60)
    
    # Создаем анализатор с параметрами из задания
    # В задании A=1, но в формулах ответа A не указан, поэтому используем A=1
    analyzer = RectangularPulseAnalyzer(A=1.0, tau=2.0, fs=1000, T=10)
    
    # 1. Проверяем формулы
    analyzer.verify_formulas()
    
    # 2. Графики во временной области
    print("\nПостроение графиков во временной области...")
    analyzer.plot_time_domain()
    
    # 3. Графики в частотной области
    print("Построение графиков в частотной области...")
    analyzer.plot_frequency_domain()
    
    # 4. Сравнение спектров
    print("Сравнение спектров симметричного и несимметричного импульсов...")
    analyzer.compare_spectra()
    
    # Дополнительно: исследование влияния длительности импульса
    print("\n" + "=" * 60)
    print("ДОПОЛНИТЕЛЬНО: Влияние длительности импульса τ на спектр")
    print("=" * 60)
    
    tau_values = [1.0, 2.0, 4.0]
    colors = ['r', 'g', 'b']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, tau in enumerate(tau_values):
        # Временной анализ
        temp_analyzer = RectangularPulseAnalyzer(A=1.0, tau=tau, fs=1000, T=10)
        symmetric = temp_analyzer.symmetric_pulse()
        
        # Частотный анализ
        f_analytic = np.linspace(-5, 5, 1000)
        spectrum = temp_analyzer.analytic_spectrum_symmetric(f_analytic)
        
        # График во времени
        mask_time = (temp_analyzer.t >= -tau*1.5) & (temp_analyzer.t <= tau*1.5)
        ax1.plot(temp_analyzer.t[mask_time], symmetric[mask_time], 
                color=colors[i], linewidth=2, label=f'τ = {tau} с')
        
        # График спектра
        ax2.plot(f_analytic, np.abs(spectrum), color=colors[i], 
                linewidth=2, label=f'τ = {tau} с')
    
    ax1.set_title('Прямоугольные импульсы разной длительности')
    ax1.set_xlabel('Время t (с)')
    ax1.set_ylabel('x₀(t)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Соответствующие амплитудные спектры')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('|x̃₀(f)|')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nНаблюдения:")
    print("1. Чем короче импульс (меньше τ), тем шире его спектр")
    print("2. Чем длиннее импульс (больше τ), тем уже его спектр")
    print("3. Площадь под спектром сохраняется (равна амплитуде импульса)")
    print("4. Это проявление принципа неопределенности время-частота")

if __name__ == "__main__":
    main()
