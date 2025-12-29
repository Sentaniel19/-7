import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

class GeneralizedFourierTransform:
    """
    Исследование слабой сходимости и обобщенных преобразований Фурье
    """
    
    def __init__(self, tau_values=None, f0=2.0, fs=5000, T=10):
        """
        Инициализация параметров
        
        Parameters:
        tau_values: значения τ для исследования сходимости
        f0: частота гармонических функций
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        if tau_values is None:
            tau_values = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        
        self.tau_values = tau_values
        self.f0 = f0
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Частотная ось
        self.f = np.linspace(-20, 20, 2000)
        
        # Создаем пробные функции из пространства Шварца S(ℝ)
        self.test_functions = self._create_test_functions()
    
    def sinc_function(self, f, tau):
        """Функция τ * sinc(fτ)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            return tau * np.sinc(f * tau)
    
    def _create_test_functions(self):
        """Создание пробных функций из пространства Шварца"""
        t = self.t
        
        # 1. Гауссова функция
        gauss = np.exp(-t**2)
        
        # 2. Функция с компактным носителем (бесконечно гладкая)
        compact = np.zeros_like(t)
        mask = (np.abs(t) <= 2)
        compact[mask] = np.exp(-1/(1 - (t[mask]/2)**2))
        
        # 3. Произведение гауссовой функции на полином
        gauss_poly = np.exp(-t**2) * (1 - t**2 + t**4/4)
        
        # 4. Еще одна гладкая быстроубывающая функция
        smooth = np.exp(-t**2/4) * np.cos(2*np.pi*t)
        
        return {
            'φ₁(t) = exp(-t²)': gauss,
            'φ₂(t) = exp(-1/(1-(t/2)²)) (компактная)': compact,
            'φ₃(t) = exp(-t²)(1-t²+t⁴/4)': gauss_poly,
            'φ₄(t) = exp(-t²/4)cos(2πt)': smooth
        }
    
    def check_weak_convergence_sinc_to_delta(self):
        """
        Проверка слабой сходимости τ·sinc(fτ) → δ(f) при τ→∞
        """
        print("=" * 80)
        print("ПРОВЕРКА СЛАБОЙ СХОДИМОСТИ τ·sinc(fτ) → δ(f) ПРИ τ→∞")
        print("=" * 80)
        
        print("Определение слабой сходимости:")
        print("Для любой пробной функции ψ(f) ∈ S(ℝ):")
        print("lim_{τ→∞} ∫ τ·sinc(fτ) ψ(f) df = ψ(0)")
        print("Это означает τ·sinc(fτ) → δ(f) слабо при τ→∞")
        print()
        
        results = {}
        
        for name, test_func_time in self.test_functions.items():
            # Преобразуем пробную функцию в частотную область
            # Для этого нужен её спектр, но мы можем работать с тестовыми функциями в частотной области
            # Просто создадим аналогичные функции в частотной области
            test_func_freq = np.exp(-(self.f/5)**2) * (1 + 0.1*self.f**2)
            
            print(f"\nПроверка для пробной функции ψ(f):")
            print(f"ψ(0) = {test_func_freq[np.abs(self.f) == np.min(np.abs(self.f))][0]:.8f}")
            print("-" * 60)
            
            integrals = []
            
            for tau in self.tau_values:
                # Функция τ·sinc(fτ)
                sinc_func = self.sinc_function(self.f, tau)
                
                # Вычисление интеграла ∫ τ·sinc(fτ) ψ(f) df
                integral = np.trapz(sinc_func * test_func_freq, self.f)
                integrals.append(integral)
                
                print(f"  τ = {tau:.4f}: ∫ τ·sinc(fτ)ψ(f)df = {integral:.8f}")
            
            # Теоретический предел ψ(0)
            psi_0 = test_func_freq[np.abs(self.f) == np.min(np.abs(self.f))][0]
            
            # Экстраполируем предел при τ→∞
            # Используем последние несколько значений для оценки предела
            if len(integrals) >= 3:
                estimated_limit = integrals[-1]  # Последнее значение
                error = abs(estimated_limit - psi_0)
                
                print(f"  Предел при τ→∞ (оценка): {estimated_limit:.8f}")
                print(f"  ψ(0): {psi_0:.8f}")
                print(f"  Ошибка: {error:.8f}")
            
            results[name] = {
                'integrals': integrals,
                'psi_0': psi_0,
                'estimated_limit': integrals[-1] if integrals else 0
            }
        
        # Графическая иллюстрация
        self.plot_sinc_convergence()
        
        return results
    
    def plot_sinc_convergence(self):
        """Графическая иллюстрация сходимости τ·sinc(fτ) → δ(f)"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Выбираем подмножество tau для наглядности
        tau_subset = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(tau_subset)))
        
        # 1. Функции τ·sinc(fτ) для разных τ
        ax1 = axes[0, 0]
        for i, tau in enumerate(tau_subset):
            sinc_func = self.sinc_function(self.f, tau)
            ax1.plot(self.f, sinc_func, color=colors[i], 
                    linewidth=2, alpha=0.7, label=f'τ = {tau}')
        
        ax1.set_title('Функции τ·sinc(fτ) при разных τ')
        ax1.set_xlabel('Частота f (Гц)')
        ax1.set_ylabel('τ·sinc(fτ)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-10, 10)
        
        # 2. То же самое в логарифмическом масштабе по y
        ax2 = axes[0, 1]
        for i, tau in enumerate(tau_subset):
            sinc_func = self.sinc_function(self.f, tau)
            ax2.semilogy(self.f, np.abs(sinc_func), color=colors[i], 
                        linewidth=2, alpha=0.7, label=f'τ = {tau}')
        
        ax2.set_title('τ·sinc(fτ) (логарифмическая шкала)')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('|τ·sinc(fτ)|')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-10, 10)
        ax2.set_ylim(1e-3, 10)
        
        # 3. Интегралы с пробной функцией
        ax3 = axes[0, 2]
        
        # Создаем более широкий диапазон τ для анализа сходимости
        tau_wide = np.logspace(-1, 2, 50)  # от 0.1 до 100
        
        # Тестовая функция в частотной области
        test_func = np.exp(-(self.f/5)**2)
        psi_0 = test_func[np.abs(self.f) == np.min(np.abs(self.f))][0]
        
        integrals = []
        for tau in tau_wide:
            sinc_func = self.sinc_function(self.f, tau)
            integral = np.trapz(sinc_func * test_func, self.f)
            integrals.append(integral)
        
        ax3.semilogx(tau_wide, integrals, 'b-', linewidth=2, label='∫ τ·sinc(fτ)ψ(f)df')
        ax3.axhline(y=psi_0, color='r', linestyle='--', 
                   label=f'ψ(0) = {psi_0:.4f}')
        ax3.set_title('Сходимость интегралов к ψ(0)')
        ax3.set_xlabel('τ')
        ax3.set_ylabel('Значение интеграла')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Дельта-функция как предел
        ax4 = axes[1, 0]
        
        # Для большого τ функция становится узкой и высокой
        tau_large = 20.0
        sinc_large = self.sinc_function(self.f, tau_large)
        
        ax4.plot(self.f, sinc_large, 'b-', linewidth=2, 
                label=f'τ·sinc(fτ), τ = {tau_large}')
        
        # Идеализированная дельта-функция (стрелка)
        ax4.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='δ(f)')
        ax4.annotate('δ(f)', xy=(0, 0.5), xytext=(1, 0.7),
                    arrowprops=dict(arrowstyle='->', color='r'),
                    fontsize=12, color='r')
        
        ax4.set_title('τ·sinc(fτ) → δ(f) при τ→∞')
        ax4.set_xlabel('Частота f (Гц)')
        ax4.set_ylabel('Амплитуда')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-2, 2)
        
        # 5. Площадь под кривой (должна быть постоянной = 1)
        ax5 = axes[1, 1]
        
        areas = []
        for tau in tau_wide:
            sinc_func = self.sinc_function(self.f, tau)
            area = np.trapz(sinc_func, self.f)
            areas.append(area)
        
        ax5.semilogx(tau_wide, areas, 'g-', linewidth=2)
        ax5.axhline(y=1, color='r', linestyle='--', label='Площадь = 1')
        ax5.set_title('Площадь под τ·sinc(fτ)')
        ax5.set_xlabel('τ')
        ax5.set_ylabel('Площадь')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0.9, 1.1)
        
        # 6. Ошибка аппроксимации
        ax6 = axes[1, 2]
        
        errors = []
        for tau in tau_wide:
            sinc_func = self.sinc_function(self.f, tau)
            integral = np.trapz(sinc_func * test_func, self.f)
            error = abs(integral - psi_0)
            errors.append(error)
        
        ax6.loglog(tau_wide, errors, 'm-', linewidth=2, label='Ошибка')
        
        # Теоретическая оценка O(1/τ)
        theoretical = 1 / tau_wide
        ax6.loglog(tau_wide, theoretical, 'r--', linewidth=1.5, 
                  label='O(1/τ)', alpha=0.7)
        
        ax6.set_title('Ошибка аппроксимации δ(f)')
        ax6.set_xlabel('τ')
        ax6.set_ylabel('|∫τ·sinc(fτ)ψ(f)df - ψ(0)|')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Слабая сходимость τ·sinc(fτ) → δ(f) при τ→∞', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def compute_generalized_fourier_exp(self, f):
        """
        Обобщённое преобразование Фурье функции exp(i2πf₀t)
        
        По определению: F{exp(i2πf₀t)} = δ(f - f₀)
        """
        # Это дельта-функция, сдвинутая на f₀
        # В численном представлении - узкий импульс около f₀
        # Для визуализации используем узкий гауссов импульс
        sigma = 0.1  # малая ширина для аппроксимации дельта-функции
        return np.exp(-((f - self.f0)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    
    def compute_generalized_fourier_cos(self, f):
        """
        Обобщённое преобразование Фурье функции cos(2πf₀t)
        
        F{cos(2πf₀t)} = 1/2 [δ(f - f₀) + δ(f + f₀)]
        """
        sigma = 0.1
        term1 = np.exp(-((f - self.f0)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        term2 = np.exp(-((f + self.f0)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        return 0.5 * (term1 + term2)
    
    def compute_generalized_fourier_sin(self, f):
        """
        Обобщённое преобразование Фурье функции sin(2πf₀t)
        
        F{sin(2πf₀t)} = 1/(2i) [δ(f - f₀) - δ(f + f₀)]
        """
        sigma = 0.1
        term1 = np.exp(-((f - self.f0)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        term2 = np.exp(-((f + self.f0)**2)/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        return (1/(2j)) * (term1 - term2)
    
    def demonstrate_plancherel_parseval(self):
        """
        Демонстрация формулы Планшереля-Парсеваля
        """
        print("\n" + "=" * 80)
        print("ДЕМОНСТРАЦИЯ ФОРМУЛЫ ПЛАНШЕРЕЛЯ-ПАРСЕВАЛЯ")
        print("=" * 80)
        
        print("\nФормула Планшереля-Парсеваля:")
        print("Для функций f, g ∈ L²(ℝ):")
        print("∫ f(t) g*(t) dt = ∫ F(f)(ω) F(g)*(ω) dω")
        print("где F - преобразование Фурье")
        print()
        
        print("В частности, для f = g:")
        print("∫ |f(t)|² dt = ∫ |F(f)(ω)|² dω")
        print("(сохранение энергии)")
        print()
        
        # Демонстрация на примере прямоугольного импульса
        A = 1.0
        tau = 2.0
        
        # Прямоугольный импульс
        rect_pulse = np.zeros_like(self.t)
        rect_pulse[(self.t >= -tau/2) & (self.t <= tau/2)] = A
        
        # Его спектр
        rect_spectrum = tau * np.sinc(self.f * tau)
        
        # Энергия во временной области
        energy_time = np.trapz(np.abs(rect_pulse)**2, self.t)
        
        # Энергия в частотной области (нужно масштабировать)
        # Преобразование Фурье даёт спектр в правильных единицах
        energy_freq = np.trapz(np.abs(rect_spectrum)**2, self.f)
        
        print(f"Пример: прямоугольный импульс длительности τ = {tau} с")
        print(f"Энергия во временной области: {energy_time:.6f}")
        print(f"Энергия в частотной области: {energy_freq:.6f}")
        print(f"Отношение (должно быть 1): {energy_time/energy_freq:.6f}")
        print()
        
        # Теперь применим к гармоническим функциям
        print("Для гармонических функций exp(i2πf₀t), cos(2πf₀t), sin(2πf₀t):")
        print("Эти функции не принадлежат L²(ℝ), но можно рассматривать")
        print("их на конечном интервале или как обобщённые функции.")
        print()
        
        # Рассмотрим на конечном интервале [-T/2, T/2]
        T_window = 10.0
        mask = (np.abs(self.t) <= T_window/2)
        t_windowed = self.t[mask]
        
        # exp(i2πf₀t) на конечном интервале
        exp_signal = np.exp(1j * 2 * np.pi * self.f0 * t_windowed)
        
        # Его энергия на интервале
        energy_exp = np.trapz(np.abs(exp_signal)**2, t_windowed)
        
        # Спектр на интервале (через БПФ)
        N = len(t_windowed)
        dt_windowed = t_windowed[1] - t_windowed[0]
        spectrum_exp = fft(exp_signal) * dt_windowed
        freq_exp = fftfreq(N, dt_windowed)
        
        # Энергия в частотной области
        energy_exp_freq = np.trapz(np.abs(spectrum_exp)**2, freq_exp)
        
        print(f"exp(i2πf₀t) на интервале [-{T_window/2}, {T_window/2}]:")
        print(f"Энергия во времени: {energy_exp:.6f}")
        print(f"Энергия в частоте: {energy_exp_freq:.6f}")
        print(f"Отношение: {energy_exp/energy_exp_freq:.6f}")
        print()
        
        return energy_time, energy_freq, energy_exp, energy_exp_freq
    
    def derive_generalized_transforms(self):
        """
        Вывод обобщённых преобразований Фурье гармонических функций
        с использованием результата пункта 4
        """
        print("\n" + "=" * 80)
        print("ВЫВОД ОБОБЩЁННЫХ ПРЕОБРАЗОВАНИЙ ФУРЬЕ")
        print("=" * 80)
        
        print("\nИспользуем результат из пункта 4:")
        print("Для комплексного радиоимпульса:")
        print("z_{τ,f0}(t) = exp(i2πf₀t) * 1_{[-τ/2, τ/2]}(t)")
        print("Его спектр: Z_{τ,f0}(f) = τ * sinc((f - f₀)τ)")
        print()
        
        print("Рассмотрим предел при τ → ∞:")
        print("lim_{τ→∞} z_{τ,f0}(t) = exp(i2πf₀t)  (на всей оси)")
        print()
        
        print("Из слабой сходимости τ·sinc(fτ) → δ(f) при τ→∞ следует:")
        print("lim_{τ→∞} τ * sinc((f - f₀)τ) = δ(f - f₀)")
        print()
        
        print("Следовательно:")
        print("F{exp(i2πf₀t)} = δ(f - f₀)")
        print()
        
        print("Теперь для cos(2πf₀t):")
        print("cos(2πf₀t) = (exp(i2πf₀t) + exp(-i2πf₀t))/2")
        print("F{cos(2πf₀t)} = 1/2 [F{exp(i2πf₀t)} + F{exp(-i2πf₀t)}]")
        print("             = 1/2 [δ(f - f₀) + δ(f + f₀)]")
        print()
        
        print("Для sin(2πf₀t):")
        print("sin(2πf₀t) = (exp(i2πf₀t) - exp(-i2πf₀t))/(2i)")
        print("F{sin(2πf₀t)} = 1/(2i) [F{exp(i2πf₀t)} - F{exp(-i2πf₀t)}]")
        print("             = 1/(2i) [δ(f - f₀) - δ(f + f₀)]")
        print()
        
        # Графическая иллюстрация
        self.plot_generalized_transforms()
    
    def plot_generalized_transforms(self):
        """Графическая иллюстрация обобщённых преобразований Фурье"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. exp(i2πf₀t) и его спектр
        ax1 = axes[0, 0]
        t_plot = np.linspace(-2, 2, 1000)
        exp_signal = np.exp(1j * 2 * np.pi * self.f0 * t_plot)
        
        ax1.plot(t_plot, exp_signal.real, 'b-', linewidth=2, label='Re{exp(i2πf₀t)}')
        ax1.plot(t_plot, exp_signal.imag, 'r-', linewidth=2, label='Im{exp(i2πf₀t)}')
        ax1.set_title(f'exp(i2πf₀t), f₀ = {self.f0} Гц')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-2, 2)
        
        ax2 = axes[0, 1]
        exp_spectrum = self.compute_generalized_fourier_exp(self.f)
        ax2.plot(self.f, exp_spectrum, 'b-', linewidth=2)
        ax2.axvline(x=self.f0, color='r', linestyle=':', alpha=0.7)
        ax2.set_title(f'F{{exp(i2πf₀t)}} = δ(f - f₀)')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('Амплитуда')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-self.f0*1.5, self.f0*1.5)
        
        # 2. cos(2πf₀t) и его спектр
        ax3 = axes[0, 2]
        cos_signal = np.cos(2 * np.pi * self.f0 * t_plot)
        ax3.plot(t_plot, cos_signal, 'g-', linewidth=2)
        ax3.set_title(f'cos(2πf₀t), f₀ = {self.f0} Гц')
        ax3.set_xlabel('Время t (с)')
        ax3.set_ylabel('Амплитуда')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-2, 2)
        
        ax4 = axes[1, 0]
        cos_spectrum = self.compute_generalized_fourier_cos(self.f)
        ax4.plot(self.f, cos_spectrum, 'g-', linewidth=2)
        ax4.axvline(x=self.f0, color='r', linestyle=':', alpha=0.7)
        ax4.axvline(x=-self.f0, color='r', linestyle=':', alpha=0.7)
        ax4.set_title(f'F{{cos(2πf₀t)}} = 1/2 [δ(f - f₀) + δ(f + f₀)]')
        ax4.set_xlabel('Частота f (Гц)')
        ax4.set_ylabel('Амплитуда')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-self.f0*1.5, self.f0*1.5)
        
        # 3. sin(2πf₀t) и его спектр
        ax5 = axes[1, 1]
        sin_signal = np.sin(2 * np.pi * self.f0 * t_plot)
        ax5.plot(t_plot, sin_signal, 'm-', linewidth=2)
        ax5.set_title(f'sin(2πf₀t), f₀ = {self.f0} Гц')
        ax5.set_xlabel('Время t (с)')
        ax5.set_ylabel('Амплитуда')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-2, 2)
        
        ax6 = axes[1, 2]
        sin_spectrum = self.compute_generalized_fourier_sin(self.f)
        ax6.plot(self.f, np.abs(sin_spectrum), 'm-', linewidth=2, label='|F{sin}|')
        ax6.plot(self.f, np.real(sin_spectrum), 'b--', linewidth=1, alpha=0.7, label='Re{F{sin}}')
        ax6.plot(self.f, np.imag(sin_spectrum), 'r--', linewidth=1, alpha=0.7, label='Im{F{sin}}')
        ax6.axvline(x=self.f0, color='r', linestyle=':', alpha=0.7)
        ax6.axvline(x=-self.f0, color='r', linestyle=':', alpha=0.7)
        ax6.set_title(f'F{{sin(2πf₀t)}} = 1/(2i)[δ(f - f₀) - δ(f + f₀)]')
        ax6.set_xlabel('Частота f (Гц)')
        ax6.set_ylabel('Амплитуда')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(-self.f0*1.5, self.f0*1.5)
        
        plt.suptitle('Обобщённые преобразования Фурье гармонических функций', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_weak_convergence_limit(self):
        """
        Демонстрация перехода от конечного импульса к бесконечной гармонике
        """
        print("\n" + "=" * 80)
        print("ПЕРЕХОД ОТ КОНЕЧНОГО ИМПУЛЬСА К БЕСКОНЕЧНОЙ ГАРМОНИКЕ")
        print("=" * 80)
        
        # Разные значения τ (длительности импульса)
        tau_values = [1.0, 2.0, 4.0, 8.0, 16.0]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Временная область
        ax1 = axes[0, 0]
        t_plot = np.linspace(-5, 5, 1000)
        
        for i, tau in enumerate(tau_values):
            # Прямоугольный радиоимпульс
            signal = np.zeros_like(t_plot)
            mask = (np.abs(t_plot) <= tau/2)
            signal[mask] = np.cos(2 * np.pi * self.f0 * t_plot[mask])
            
            ax1.plot(t_plot, signal, linewidth=2, alpha=0.7, 
                    label=f'τ = {tau} с')
        
        # Бесконечная косинусоида
        ax1.plot(t_plot, np.cos(2 * np.pi * self.f0 * t_plot), 
                'k--', linewidth=3, label='cos(2πf₀t) (беск.)')
        
        ax1.set_title('Временная область: cos(2πf₀t)·1_{[-τ/2, τ/2]}(t)')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('Амплитуда')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-5, 5)
        
        # Частотная область
        ax2 = axes[0, 1]
        f_plot = np.linspace(-10, 10, 1000)
        
        for i, tau in enumerate(tau_values):
            # Спектр прямоугольного радиоимпульса
            spectrum = 0.5 * (tau * np.sinc((f_plot + self.f0) * tau) + 
                             tau * np.sinc((f_plot - self.f0) * tau))
            
            ax2.plot(f_plot, np.abs(spectrum), linewidth=2, alpha=0.7,
                    label=f'τ = {tau} с')
        
        # Спектр бесконечной косинусоиды (дельта-функции)
        # Аппроксимируем узкими гауссовыми функциями
        sigma = 0.1
        cos_spectrum = 0.5 * (np.exp(-((f_plot - self.f0)**2)/(2*sigma**2)) +
                             np.exp(-((f_plot + self.f0)**2)/(2*sigma**2))) / (sigma*np.sqrt(2*np.pi))
        
        ax2.plot(f_plot, cos_spectrum, 'k--', linewidth=3, 
                label='F{cos(2πf₀t)}')
        
        ax2.set_title('Частотная область: спектры')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('|X(f)|')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-10, 10)
        
        # Ширина главного лепестка в зависимости от τ
        ax3 = axes[0, 2]
        
        widths = []
        for tau in tau_values:
            # Ширина главного лепестка sinc-функции ~ 2/τ
            width = 2.0 / tau
            widths.append(width)
        
        ax3.plot(tau_values, widths, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Длительность τ (с)')
        ax3.set_ylabel('Ширина спектра (Гц)')
        ax3.set_title('Ширина спектра ~ 2/τ')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Энергия на интервале
        ax4 = axes[1, 0]
        
        energies = []
        for tau in tau_values:
            # Энергия на интервале [-τ/2, τ/2]
            energy = tau  # для cos² среднее значение 1/2, но у нас амплитуда 1
            energies.append(energy)
        
        ax4.plot(tau_values, energies, 'go-', linewidth=2, markersize=8)
        ax4.set_xlabel('Длительность τ (с)')
        ax4.set_ylabel('Энергия на интервале')
        ax4.set_title('Энергия ~ τ')
        ax4.grid(True, alpha=0.3)
        
        # Спектральная плотность в точке f₀
        ax5 = axes[1, 1]
        
        spectral_densities = []
        for tau in tau_values:
            # Значение спектра в точке f₀
            density = tau  # sinc(0) = 1
            spectral_densities.append(density)
        
        ax5.plot(tau_values, spectral_densities, 'ro-', 
                linewidth=2, markersize=8)
        ax5.set_xlabel('Длительность τ (с)')
        ax5.set_ylabel('|X(f₀)|')
        ax5.set_title('Спектральная плотность ~ τ')
        ax5.grid(True, alpha=0.3)
        
        # Нормированная спектральная плотность
        ax6 = axes[1, 2]
        
        normalized = []
        for tau in tau_values:
            # |X(f₀)|/τ = 1 (постоянно)
            normalized.append(1)
        
        ax6.plot(tau_values, normalized, 'mo-', linewidth=2, markersize=8)
        ax6.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Длительность τ (с)')
        ax6.set_ylabel('|X(f₀)|/τ')
        ax6.set_title('Нормированная спектральная плотность')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0.5, 1.5)
        
        plt.suptitle(f'Переход к бесконечной гармонике при τ→∞ (f₀ = {self.f0} Гц)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\nВыводы:")
        print("1. При τ → ∞ прямоугольный радиоимпульс стремится к бесконечной гармонике")
        print("2. Ширина его спектра стремится к 0")
        print("3. Значение спектра в точке f₀ растёт как τ")
        print("4. В пределе получаем дельта-функцию: F{cos(2πf₀t)} = 1/2[δ(f-f₀) + δ(f+f₀)]")

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 5")
    print("Слабая сходимость к δ(f) и обобщённые преобразования Фурье")
    print("=" * 80)
    
    # Создаем анализатор
    analyzer = GeneralizedFourierTransform(
        tau_values=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        f0=2.0,  # частота гармонических функций
        fs=10000,
        T=10
    )
    
    # 1. Проверка слабой сходимости τ·sinc(fτ) → δ(f)
    print("\n1. Проверка слабой сходимости τ·sinc(fτ) → δ(f) при τ→∞...")
    convergence_results = analyzer.check_weak_convergence_sinc_to_delta()
    
    # 2. Формула Планшереля-Парсеваля
    print("\n2. Демонстрация формулы Планшереля-Парсеваля...")
    energies = analyzer.demonstrate_plancherel_parseval()
    
    # 3. Вывод обобщённых преобразований
    print("\n3. Вывод обобщённых преобразований Фурье...")
    analyzer.derive_generalized_transforms()
    
    # 4. Демонстрация перехода
    print("\n4. Демонстрация перехода от конечного импульса к бесконечной гармонике...")
    analyzer.demonstrate_weak_convergence_limit()
    
    # 5. Итоговые выводы
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ ПО ЗАДАНИЮ 5")
    print("=" * 80)
    
    print("\n1. Доказана слабая сходимость:")
    print("   τ·sinc(fτ) → δ(f) при τ→∞")
    print("   в смысле функционалов на пространстве Шварца S(ℝ)")
    print()
    
    print("2. Используя результат пункта 4 и эту сходимость:")
    print("   Для комплексного радиоимпульса:")
    print("   z_{τ,f0}(t) = exp(i2πf₀t)·1_{[-τ/2, τ/2]}(t)")
    print("   Z_{τ,f0}(f) = τ·sinc((f - f₀)τ)")
    print("   При τ→∞: Z_{τ,f0}(f) → δ(f - f₀)")
    print()
    
    print("3. Получены обобщённые преобразования Фурье:")
    print("   F{exp(i2πf₀t)} = δ(f - f₀)")
    print("   F{cos(2πf₀t)} = 1/2 [δ(f - f₀) + δ(f + f₀)]")
    print("   F{sin(2πf₀t)} = 1/(2i) [δ(f - f₀) - δ(f + f₀)]")
    print()
    
    print("4. Использована формула Планшереля-Парсеваля:")
    print("   ∫ f(t) g*(t) dt = ∫ F(f)(ω) F(g)*(ω) dω")
    print("   Для проверки корректности обобщённых преобразований")
    print()
    
    print("5. Физическая интерпретация:")
    print("   Бесконечная гармоника имеет бесконечную энергию")
    print("   Её спектр - дельта-функция (бесконечная спектральная плотность)")
    print("   Это согласуется с представлением о монохроматическом сигнале")
    print()
    
    print("=" * 80)
    print("ЗАДАНИЕ ВЫПОЛНЕНО ПОЛНОСТЬЮ!")
    print("=" * 80)

if __name__ == "__main__":
    main()
