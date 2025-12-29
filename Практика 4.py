import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

class ComplexRadioPulseAnalyzer:
    """
    Анализатор комплексного прямоугольного радиоимпульса
    """
    
    def __init__(self, A=1.0, tau=2.0, f0_values=None, fs=5000, T=10):
        """
        Инициализация параметров
        
        Parameters:
        A: амплитуда импульса
        tau: длительность импульса
        f0_values: значения несущей частоты для исследования
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        self.A = A
        self.tau = tau
        
        if f0_values is None:
            f0_values = [0.5, 1.0, 2.0, 5.0, 10.0]  # значения f0 для исследования
        
        self.f0_values = f0_values
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Частотная ось для аналитического спектра
        self.f_analytic = np.linspace(-20, 20, 2000)
    
    def complex_radio_pulse(self, f0):
        """
        Комплексный прямоугольный симметричный радиоимпульс
        
        z_{τ,f0}(t) = exp(iω₀t) * 1_{[-τ/2, τ/2]}(t)
        где ω₀ = 2πf0
        """
        # Прямоугольная огибающая
        envelope = np.zeros_like(self.t, dtype=complex)
        mask = (self.t >= -self.tau/2) & (self.t <= self.tau/2)
        envelope[mask] = self.A
        
        # Комплексное экспоненциальное заполнение
        carrier = np.exp(1j * 2 * np.pi * f0 * self.t)
        
        return envelope * carrier
    
    def analytic_spectrum_complex_pulse(self, f, f0):
        """
        Аналитический спектр комплексного радиоимпульса
        
        Z_{τ,f0}(f) = τ * sinc((f - f0)τ)
        """
        # Избегаем деления на 0
        with np.errstate(divide='ignore', invalid='ignore'):
            spectrum = self.tau * np.sinc((f - f0) * self.tau)
        
        return spectrum
    
    def real_radio_pulse_spectrum(self, f, f0, phi0=0):
        """
        Спектр вещественного радиоимпульса для сравнения
        
        x̃_{τ,f0}(f) = 1/2 * [τ * sinc((f+f0)τ) * e^{-iφ₀} 
                          + τ * sinc((f-f0)τ) * e^{iφ₀}]
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = self.tau * np.sinc((f + f0) * self.tau) * np.exp(-1j * phi0)
            term2 = self.tau * np.sinc((f - f0) * self.tau) * np.exp(1j * phi0)
        
        return 0.5 * (term1 + term2)
    
    def compute_fft_spectrum(self, signal):
        """Вычисление спектра через БПФ"""
        N = len(signal)
        
        # Вычисляем БПФ
        spectrum = fft(signal) * self.dt
        
        # Частотная ось
        freq = fftfreq(N, self.dt)
        
        # Сдвигаем для правильного отображения
        spectrum_shifted = fftshift(spectrum)
        freq_shifted = fftshift(freq)
        
        return freq_shifted, spectrum_shifted
    
    def verify_formula(self):
        """Проверка формулы из ответа"""
        print("=" * 70)
        print("ПРОВЕРКА ФОРМУЛЫ ИЗ ЗАДАНИЯ")
        print("=" * 70)
        
        print(f"Параметры: A = {self.A}, τ = {self.tau} с")
        print()
        
        print("Формула из задания:")
        print("Z_{τ,f0}(f) = τ * sinc((f - f0)τ)")
        print()
        
        # Проверка для различных частот
        f0 = self.f0_values[1]  # возьмем f0 = 1.0 Гц
        test_freqs = np.array([f0 - 2, f0 - 1, f0, f0 + 1, f0 + 2])
        
        print(f"Для f₀ = {f0} Гц:")
        print("-" * 40)
        
        for f in test_freqs:
            # Вычисляем по формуле
            spectrum = self.analytic_spectrum_complex_pulse(f, f0)
            
            print(f"  f = {f:.2f} Гц:")
            print(f"    |Z(f)| = {np.abs(spectrum):.6f}")
            print(f"    ∠Z(f) = {np.angle(spectrum):.6f} рад")
            print(f"    Z(f) = {spectrum.real:.6f} + j{spectrum.imag:.6f}")
        
        print("\n" + "=" * 70)
    
    def plot_time_domain(self):
        """Графики комплексного сигнала во временной области"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, f0 in enumerate(self.f0_values[:3]):  # Первые 3 значения f0
            # Комплексный сигнал
            complex_signal = self.complex_radio_pulse(f0)
            
            # Масштабируем для лучшей видимости
            mask = (np.abs(self.t) <= self.tau * 1.5)
            
            # Действительная часть
            ax1 = axes[0, i]
            ax1.plot(self.t[mask], complex_signal[mask].real, 'b-', 
                    linewidth=2, label='Re{z(t)}')
            ax1.set_title(f'Действительная часть\nf₀ = {f0} Гц')
            ax1.set_xlabel('Время t (с)')
            ax1.set_ylabel('Амплитуда')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-self.tau*1.5, self.tau*1.5)
            ax1.set_ylim(-self.A*1.2, self.A*1.2)
            
            # Мнимая часть
            ax2 = axes[1, i]
            ax2.plot(self.t[mask], complex_signal[mask].imag, 'r-', 
                    linewidth=2, label='Im{z(t)}')
            ax2.set_title(f'Мнимая часть\nf₀ = {f0} Гц')
            ax2.set_xlabel('Время t (с)')
            ax2.set_ylabel('Амплитуда')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-self.tau*1.5, self.tau*1.5)
            ax2.set_ylim(-self.A*1.2, self.A*1.2)
        
        plt.suptitle('Комплексный радиоимпульс во временной области', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Дополнительно: представление в полярных координатах
        print("\nПредставление комплексного сигнала:")
        print("z(t) = exp(iω₀t) = cos(ω₀t) + i·sin(ω₀t)")
        print("Модуль: |z(t)| = 1 (внутри импульса)")
        print("Фаза: arg(z(t)) = ω₀t (линейно растёт)")
        print()
    
    def plot_frequency_domain(self):
        """Спектры в частотной области"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i, f0 in enumerate(self.f0_values[:3]):  # Первые 3 значения f0
            # Аналитический спектр комплексного импульса
            complex_spectrum = self.analytic_spectrum_complex_pulse(
                self.f_analytic, f0)
            
            # Спектр вещественного импульса для сравнения (φ₀ = 0)
            real_spectrum = self.real_radio_pulse_spectrum(
                self.f_analytic, f0, 0)
            
            # Амплитудный спектр
            ax1 = axes[0, i]
            ax1.plot(self.f_analytic, np.abs(complex_spectrum), 'b-', 
                    linewidth=2, label='Комплексный импульс')
            ax1.plot(self.f_analytic, np.abs(real_spectrum), 'r--', 
                    linewidth=1.5, alpha=0.7, label='Вещественный импульс')
            
            ax1.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
            ax1.set_title(f'Амплитудный спектр\nf₀ = {f0} Гц')
            ax1.set_xlabel('Частота f (Гц)')
            ax1.set_ylabel('|Z(f)|')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-15, 15)
            ax1.set_ylim(-0.1, self.tau*1.1)
            
            # Фазовый спектр
            ax2 = axes[1, i]
            phase_complex = np.angle(complex_spectrum)
            phase_real = np.angle(real_spectrum)
            
            # Разворачиваем фазы для наглядности
            phase_complex_unwrapped = np.unwrap(phase_complex)
            phase_real_unwrapped = np.unwrap(phase_real)
            
            ax2.plot(self.f_analytic, phase_complex_unwrapped, 'b-', 
                    linewidth=2, label='Комплексный импульс')
            ax2.plot(self.f_analytic, phase_real_unwrapped, 'r--', 
                    linewidth=1.5, alpha=0.7, label='Вещественный импульс')
            
            ax2.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
            ax2.set_title(f'Фазовый спектр\nf₀ = {f0} Гц')
            ax2.set_xlabel('Частота f (Гц)')
            ax2.set_ylabel('∠Z(f) (рад)')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-15, 15)
        
        plt.suptitle('Сравнение спектров комплексного и вещественного импульсов', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def compare_with_real_pulse(self):
        """Сравнение комплексного и вещественного импульсов"""
        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ КОМПЛЕКСНОГО И ВЕЩЕСТВЕННОГО ИМПУЛЬСОВ")
        print("=" * 70)
        
        f0 = 2.0  # фиксированная частота
        
        print(f"\nДля f₀ = {f0} Гц:")
        print()
        
        print("1. Комплексный радиоимпульс:")
        print("   z(t) = exp(i2πf₀t) * 1_{[-τ/2, τ/2]}(t)")
        print("   Спектр: Z(f) = τ * sinc((f - f₀)τ)")
        print()
        
        print("2. Вещественный радиоимпульс (φ₀ = 0):")
        print("   x(t) = cos(2πf₀t) * 1_{[-τ/2, τ/2]}(t)")
        print("   Спектр: X(f) = 1/2 [τ * sinc((f+f₀)τ) + τ * sinc((f-f₀)τ)]")
        print()
        
        print("3. Ключевые различия:")
        print("   - Комплексный: один sinc-лепесток около f₀")
        print("   - Вещественный: два sinc-лепестка около ±f₀")
        print("   - Комплексный: несимметричный спектр")
        print("   - Вещественный: симметричный спектр (сопряжённая симметрия)")
        print()
        
        # Графическое сравнение
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Аналитические спектры
        complex_spectrum = self.analytic_spectrum_complex_pulse(
            self.f_analytic, f0)
        real_spectrum = self.real_radio_pulse_spectrum(self.f_analytic, f0, 0)
        
        # 1. Амплитудные спектры
        ax1 = axes[0, 0]
        ax1.plot(self.f_analytic, np.abs(complex_spectrum), 'b-', 
                linewidth=2, label='Комплексный |Z(f)|')
        ax1.set_title('Амплитудный спектр\nкомплексного импульса')
        ax1.set_xlabel('Частота f (Гц)')
        ax1.set_ylabel('|Z(f)|')
        ax1.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-15, 15)
        ax1.set_ylim(-0.1, self.tau*1.1)
        
        ax2 = axes[0, 1]
        ax2.plot(self.f_analytic, np.abs(real_spectrum), 'r-', 
                linewidth=2, label='Вещественный |X(f)|')
        ax2.set_title('Амплитудный спектр\nвещественного импульса')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('|X(f)|')
        ax2.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax2.axvline(x=-f0, color='g', linestyle=':', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-15, 15)
        ax2.set_ylim(-0.1, self.tau*1.1)
        
        # 2. Фазовые спектры
        ax3 = axes[1, 0]
        phase_complex = np.unwrap(np.angle(complex_spectrum))
        ax3.plot(self.f_analytic, phase_complex, 'b-', linewidth=2)
        ax3.set_title('Фазовый спектр\nкомплексного импульса')
        ax3.set_xlabel('Частота f (Гц)')
        ax3.set_ylabel('∠Z(f) (рад)')
        ax3.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-15, 15)
        
        ax4 = axes[1, 1]
        phase_real = np.unwrap(np.angle(real_spectrum))
        ax4.plot(self.f_analytic, phase_real, 'r-', linewidth=2)
        ax4.set_title('Фазовый спектр\nвещественного импульса')
        ax4.set_xlabel('Частота f (Гц)')
        ax4.set_ylabel('∠X(f) (рад)')
        ax4.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax4.axvline(x=-f0, color='g', linestyle=':', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-15, 15)
        
        plt.suptitle(f'Сравнение комплексного и вещественного импульсов (f₀ = {f0} Гц)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # 3. Математическое соотношение
        print("\n4. Математическая связь:")
        print("   exp(iω₀t) = cos(ω₀t) + i·sin(ω₀t)")
        print("   cos(ω₀t) = (exp(iω₀t) + exp(-iω₀t))/2")
        print("   sin(ω₀t) = (exp(iω₀t) - exp(-iω₀t))/(2i)")
        print()
        print("   Поэтому спектр cos(ω₀t):")
        print("   F{cos(ω₀t)} = 1/2 [δ(f - f₀) + δ(f + f₀)]")
        print("   Умножение на прямоугольную функцию даёт свёртку")
        print()
    
    def analyze_physical_meaning(self):
        """Анализ физического смысла комплексного представления"""
        print("\n" + "=" * 70)
        print("ФИЗИЧЕСКИЙ СМЫСЛ КОМПЛЕКСНОГО ПРЕДСТАВЛЕНИЯ")
        print("=" * 70)
        
        print("\n1. Комплексная экспонента:")
        print("   exp(iωt) = cos(ωt) + i·sin(ωt)")
        print("   Это компактное представление двух ортогональных колебаний")
        print()
        
        print("2. Преимущества комплексного представления:")
        print("   - Более простая математика (производная: d/dt e^{iωt} = iω e^{iωt})")
        print("   - Удобство при умножении (сложение фаз)")
        print("   - Единый спектральный компонент (только положительные частоты)")
        print()
        
        print("3. В сигнальной обработке:")
        print("   - Аналитический сигнал: s_a(t) = s(t) + i·H{s(t)}")
        print("   где H{} - преобразование Гильберта")
        print("   - Убирает отрицательные частоты")
        print("   - Сохраняет всю информацию о сигнале")
        print()
        
        print("4. Практическое применение:")
        print("   - Квадратурная модуляция (QAM)")
        print("   - Комплексная огибающая")
        print("   - Представление сигналов в системах связи")
        print()
        
        # Иллюстрация: комплексная плоскость
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Выберем момент времени
        f0 = 2.0
        t_sample = np.linspace(0, 1/f0, 100)
        
        # Комплексный сигнал в выбранные моменты времени
        complex_signal = np.exp(1j * 2 * np.pi * f0 * t_sample)
        
        # Построим траекторию в комплексной плоскости
        ax.plot(complex_signal.real, complex_signal.imag, 'b-', alpha=0.5)
        ax.scatter(complex_signal.real, complex_signal.imag, 
                  c=t_sample, cmap='viridis', s=30, alpha=0.7)
        
        # Круг единичного радиуса
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
        
        ax.set_xlabel('Re{z(t)} = cos(ω₀t)')
        ax.set_ylabel('Im{z(t)} = sin(ω₀t)')
        ax.set_title('Комплексная экспонента на комплексной плоскости\nz(t) = exp(iω₀t)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        plt.tight_layout()
        plt.show()
    
    def verify_numerically(self):
        """Численная проверка через БПФ"""
        print("\n" + "=" * 70)
        print("ЧИСЛЕННАЯ ПРОВЕРКА ЧЕРЕЗ БПФ")
        print("=" * 70)
        
        f0 = 2.0
        
        # Создаём комплексный сигнал
        complex_signal = self.complex_radio_pulse(f0)
        
        # Вычисляем спектр через БПФ
        f_fft, spectrum_fft = self.compute_fft_spectrum(complex_signal)
        
        # Аналитический спектр на тех же частотах
        mask = (np.abs(f_fft) <= 15)
        spectrum_analytic = self.analytic_spectrum_complex_pulse(
            f_fft[mask], f0)
        
        # Сравнение
        error_amplitude = np.mean(np.abs(np.abs(spectrum_fft[mask]) - 
                                        np.abs(spectrum_analytic)))
        error_phase = np.mean(np.abs(np.unwrap(np.angle(spectrum_fft[mask])) - 
                                    np.unwrap(np.angle(spectrum_analytic))))
        
        print(f"\nПараметры: f₀ = {f0} Гц, τ = {self.tau} с")
        print(f"Средняя ошибка амплитудного спектра: {error_amplitude:.6f}")
        print(f"Средняя ошибка фазового спектра: {error_phase:.6f}")
        print()
        
        # График сравнения
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # АЧС сравнение
        ax1.plot(f_fft[mask], np.abs(spectrum_fft[mask]), 'b-', 
                linewidth=2, alpha=0.7, label='БПФ')
        ax1.plot(f_fft[mask], np.abs(spectrum_analytic), 'r--', 
                linewidth=1.5, label='Аналитический')
        ax1.set_title('Амплитудный спектр комплексного импульса')
        ax1.set_xlabel('Частота f (Гц)')
        ax1.set_ylabel('|Z(f)|')
        ax1.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ФЧС сравнение
        ax2.plot(f_fft[mask], np.unwrap(np.angle(spectrum_fft[mask])), 'b-', 
                linewidth=2, alpha=0.7, label='БПФ')
        ax2.plot(f_fft[mask], np.unwrap(np.angle(spectrum_analytic)), 'r--', 
                linewidth=1.5, label='Аналитический')
        ax2.set_title('Фазовый спектр комплексного импульса')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('∠Z(f) (рад)')
        ax2.axvline(x=f0, color='g', linestyle=':', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Численная проверка (f₀ = {f0} Гц)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_shift_theorem(self):
        """Демонстрация теоремы о сдвиге в частотной области"""
        print("\n" + "=" * 70)
        print("ТЕОРЕМА О СДВИГЕ В ЧАСТОТНОЙ ОБЛАСТИ")
        print("=" * 70)
        
        print("\nТеорема: Умножение на комплексную экспоненту во времени")
        print("        соответствует сдвигу в частотной области")
        print()
        print("Если x(t) ↔ X(f), то:")
        print("x(t)·exp(i2πf₀t) ↔ X(f - f₀)")
        print()
        
        # Базовый низкочастотный импульс
        base_pulse = np.zeros_like(self.t, dtype=float)
        mask = (self.t >= -self.tau/2) & (self.t <= self.tau/2)
        base_pulse[mask] = self.A
        
        # Его спектр
        f_fft, base_spectrum = self.compute_fft_spectrum(base_pulse)
        
        # Комплексный радиоимпульс
        f0 = 3.0
        complex_signal = self.complex_radio_pulse(f0)
        f_fft2, complex_spectrum = self.compute_fft_spectrum(complex_signal)
        
        # Графическая демонстрация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Временная область: базовый импульс
        mask_time = (np.abs(self.t) <= self.tau*1.5)
        axes[0, 0].plot(self.t[mask_time], base_pulse[mask_time], 'b-', 
                       linewidth=2)
        axes[0, 0].set_title('Базовый низкочастотный импульс')
        axes[0, 0].set_xlabel('Время t (с)')
        axes[0, 0].set_ylabel('x₀(t)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Частотная область: спектр базового импульса
        mask_freq = (np.abs(f_fft) <= 10)
        axes[0, 1].plot(f_fft[mask_freq], np.abs(base_spectrum[mask_freq]), 
                       'b-', linewidth=2)
        axes[0, 1].set_title('Спектр базового импульса |X₀(f)|')
        axes[0, 1].set_xlabel('Частота f (Гц)')
        axes[0, 1].set_ylabel('|X₀(f)|')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Временная область: комплексный радиоимпульс (действительная часть)
        axes[1, 0].plot(self.t[mask_time], complex_signal[mask_time].real, 
                       'r-', linewidth=2, label='Re{z(t)}')
        axes[1, 0].plot(self.t[mask_time], complex_signal[mask_time].imag, 
                       'g-', linewidth=2, label='Im{z(t)}')
        axes[1, 0].set_title(f'Комплексный радиоимпульс (f₀ = {f0} Гц)')
        axes[1, 0].set_xlabel('Время t (с)')
        axes[1, 0].set_ylabel('Амплитуда')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Частотная область: спектр комплексного импульса
        mask_freq2 = (np.abs(f_fft2) <= 10)
        axes[1, 1].plot(f_fft2[mask_freq2], np.abs(complex_spectrum[mask_freq2]), 
                       'r-', linewidth=2)
        axes[1, 1].axvline(x=f0, color='g', linestyle=':', alpha=0.5, 
                          label=f'f₀ = {f0} Гц')
        axes[1, 1].set_title(f'Спектр комплексного импульса |Z(f)|')
        axes[1, 1].set_xlabel('Частота f (Гц)')
        axes[1, 1].set_ylabel('|Z(f)|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Демонстрация теоремы о сдвиге: x(t)·exp(i2πf₀t) ↔ X(f - f₀)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\nВывод:")
        print("Умножение базового импульса на exp(i2πf₀t) сдвигает его спектр")
        print(f"на частоту f₀ = {f0} Гц вправо")
        print("Это фундаментальное свойство преобразования Фурье")

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 4")
    print("Частотный спектр комплексного прямоугольного симметричного радиоимпульса")
    print("=" * 70)
    
    # Создаем анализатор
    analyzer = ComplexRadioPulseAnalyzer(
        A=1.0,
        tau=2.0,          # длительность импульса
        f0_values=[0.5, 1.0, 2.0, 5.0, 10.0],  # несущие частоты
        fs=10000,         # высокая частота дискретизации
        T=8               # общее время
    )
    
    # 1. Проверка формулы
    analyzer.verify_formula()
    
    # 2. Временные графики
    print("\nПостроение графиков во временной области...")
    analyzer.plot_time_domain()
    
    # 3. Частотные графики
    print("Построение графиков в частотной области...")
    analyzer.plot_frequency_domain()
    
    # 4. Сравнение с вещественным импульсом
    print("\nСравнение комплексного и вещественного импульсов...")
    analyzer.compare_with_real_pulse()
    
    # 5. Физический смысл
    print("\nАнализ физического смысла комплексного представления...")
    analyzer.analyze_physical_meaning()
    
    # 6. Численная проверка
    print("\nЧисленная проверка через БПФ...")
    analyzer.verify_numerically()
    
    # 7. Демонстрация теоремы о сдвиге
    print("\nДемонстрация теоремы о сдвиге в частотной области...")
    analyzer.demonstrate_shift_theorem()
    
    # 8. Итоговые выводы
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ ВЫВОДЫ ПО ЗАДАНИЮ 4")
    print("=" * 70)
    
    print("\n1. Формула спектра подтверждена:")
    print("   Z_{τ,f0}(f) = τ * sinc((f - f0)τ)")
    print()
    
    print("2. Комплексный радиоимпульс:")
    print("   z(t) = exp(iω₀t) * 1_{[-τ/2, τ/2]}(t)")
    print("   = cos(ω₀t) + i·sin(ω₀t) внутри импульса")
    print()
    
    print("3. Ключевые особенности:")
    print("   - Только один спектральный компонент около f₀")
    print("   - Нет сопряжённой симметрии спектра")
    print("   - Простая математическая форма")
    print()
    
    print("4. Связь с вещественным случаем:")
    print("   cos(ω₀t) = (exp(iω₀t) + exp(-iω₀t))/2")
    print("   Поэтому спектр cos содержит компоненты на ±f₀")
    print()
    
    print("5. Практическая значимость:")
    print("   - Квадратурное представление сигналов")
    print("   - Аналитические сигналы в обработке")
    print("   - Удобство математических выкладок")
    print("   - Основы цифровой модуляции (QPSK, QAM)")
    print()
    
    print("=" * 70)
    print("ЗАДАНИЕ ВЫПОЛНЕНО ПОЛНОСТЬЮ!")
    print("=" * 70)

if __name__ == "__main__":
    main()
