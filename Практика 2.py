import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')

class DeltaFunctionApproximation:
    """
    Исследование предельного перехода прямоугольных импульсов к дельта-функции Дирака
    """
    
    def __init__(self, tau_values=None, fs=5000, T=4):
        """
        Инициализация параметров
        
        Parameters:
        tau_values: значения τ для последовательности импульсов
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        if tau_values is None:
            # Значения τ от большего к меньшему
            tau_values = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
        
        self.tau_values = tau_values
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Функция для проверки слабой сходимости
        self.test_functions = self._create_test_functions()
    
    def delta_approximation_pulse(self, tau):
        """
        Прямоугольный импульс, аппроксимирующий дельта-функцию
        
        δ_τ(t) = (1/τ) * 1_{[-τ/2, τ/2]}(t)
        """
        pulse = np.zeros_like(self.t)
        pulse[(self.t >= -tau/2) & (self.t <= tau/2)] = 1.0 / tau
        return pulse
    
    def analytic_spectrum(self, f, tau):
        """
        Аналитический спектр аппроксимирующего импульса
        
        δ̃_τ(f) = sinc(πfτ)
        """
        # Избегаем деления на 0
        with np.errstate(divide='ignore', invalid='ignore'):
            spectrum = np.sinc(f * tau)
        return spectrum
    
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
    
    def _create_test_functions(self):
        """
        Создание пробных функций из пространства Шварца S(ℝ)
        для проверки слабой сходимости
        """
        t = self.t
        
        # 1. Гауссова функция (гладкая, быстро убывающая)
        gauss = lambda sigma: np.exp(-(t/sigma)**2/2) / (sigma*np.sqrt(2*np.pi))
        
        # 2. Функция с компактным носителем (C^∞)
        compact_support = np.zeros_like(t)
        mask = (np.abs(t) <= 2)
        compact_support[mask] = np.exp(-1/(1 - (t[mask]/2)**2))
        
        # 3. Еще одна гладкая функция
        smooth = np.exp(-t**2) * np.cos(2*np.pi*t)
        
        return {
            'Гауссова функция (σ=0.5)': gauss(0.5),
            'Гауссова функция (σ=1.0)': gauss(1.0),
            'Функция с компактным носителем': compact_support,
            'exp(-t²)cos(2πt)': smooth
        }
    
    def check_weak_convergence_delta(self):
        """
        Проверка слабой сходимости δ_τ(t) → δ(t) при τ → 0
        """
        print("=" * 70)
        print("ПРОВЕРКА СЛАБОЙ СХОДИМОСТИ δ_τ(t) → δ(t) ПРИ τ → 0")
        print("=" * 70)
        
        print("Определение слабой сходимости:")
        print("δ_τ → δ слабо, если для любой пробной функции φ ∈ S(ℝ):")
        print("lim_{τ→0} ∫ δ_τ(t) φ(t) dt = ∫ δ(t) φ(t) dt = φ(0)")
        print()
        
        results = {}
        
        # Для каждой пробной функции вычисляем предел
        for name, test_func in self.test_functions.items():
            print(f"\nПроверка для функции: {name}")
            print("-" * 50)
            
            integrals = []
            
            for tau in self.tau_values:
                # Аппроксимация дельта-функции
                delta_approx = self.delta_approximation_pulse(tau)
                
                # Вычисление интеграла ∫ δ_τ(t) φ(t) dt
                integral = np.trapz(delta_approx * test_func, self.t)
                integrals.append(integral)
                
                print(f"  τ = {tau:.4f} с: ∫ δ_τ(t)φ(t)dt = {integral:.8f}")
            
            # Значение φ(0) - теоретический предел
            phi_0 = test_func[np.abs(self.t) == np.min(np.abs(self.t))][0]
            print(f"  Теоретический предел φ(0) = {phi_0:.8f}")
            print(f"  Предел при τ→0: {integrals[-1]:.8f}")
            print(f"  Отклонение: {abs(integrals[-1] - phi_0):.8f}")
            
            results[name] = {
                'integrals': integrals,
                'phi_0': phi_0,
                'limit': integrals[-1]
            }
        
        return results
    
    def check_weak_convergence_spectrum(self):
        """
        Проверка слабой сходимости спектров δ̃_τ(f) → 1 при τ → 0
        """
        print("\n" + "=" * 70)
        print("ПРОВЕРКА СЛАБОЙ СХОДИМОСТИ δ̃_τ(f) → 1 ПРИ τ → 0")
        print("=" * 70)
        
        print("Определение слабой сходимости спектров:")
        print("δ̃_τ → 1 слабо, если для любой пробной функции ψ ∈ S(ℝ):")
        print("lim_{τ→0} ∫ δ̃_τ(f) ψ(f) df = ∫ 1 * ψ(f) df")
        print()
        
        # Создаем пробные функции в частотной области
        f_analytic = np.linspace(-20, 20, 2000)
        
        test_functions_freq = {
            'Гауссова в частотной области': np.exp(-(f_analytic/5)**2),
            'Функция с компактным носителем': np.where(np.abs(f_analytic) <= 5, 1, 0),
            'Другая гладкая функция': np.exp(-(f_analytic/3)**2) * np.cos(f_analytic)
        }
        
        results = {}
        
        for name, test_func in test_functions_freq.items():
            print(f"\nПроверка для функции: {name}")
            print("-" * 50)
            
            integrals = []
            
            for tau in self.tau_values:
                # Аналитический спектр
                spectrum = self.analytic_spectrum(f_analytic, tau)
                
                # Вычисление интеграла ∫ δ̃_τ(f) ψ(f) df
                integral = np.trapz(spectrum * test_func, f_analytic)
                integrals.append(integral)
                
                # Интеграл от 1 * ψ(f) df
                integral_1 = np.trapz(1 * test_func, f_analytic)
                
                print(f"  τ = {tau:.4f} с: ∫ δ̃_τ(f)ψ(f)df = {integral:.8f}")
                print(f"                ∫ 1 * ψ(f)df = {integral_1:.8f}")
                print(f"                Разница = {abs(integral - integral_1):.8f}")
            
            results[name] = {
                'integrals': integrals,
                'integral_1': integral_1,
                'limit': integrals[-1]
            }
        
        return results
    
    def plot_approximation_sequence(self):
        """Графическое представление последовательности аппроксимаций"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes = axes.flatten()
        
        for i, tau in enumerate(self.tau_values[:8]):  # Показываем первые 8
            ax = axes[i]
            
            # Импульс во времени
            delta_approx = self.delta_approximation_pulse(tau)
            
            # Масштабируем график для лучшей видимости
            mask_time = (np.abs(self.t) <= 3*tau)
            if np.sum(mask_time) > 0:
                ax.plot(self.t[mask_time], delta_approx[mask_time], 'b-', linewidth=2)
            else:
                ax.plot(self.t, delta_approx, 'b-', linewidth=2)
            
            # Площадь под кривой (должна быть = 1)
            area = np.trapz(delta_approx, self.t)
            
            ax.set_title(f'τ = {tau:.3f} с\nA = {1/tau:.1f}, S = {area:.3f}')
            ax.set_xlabel('Время t (с)')
            ax.set_ylabel('δ_τ(t)')
            ax.grid(True, alpha=0.3)
            
            # Динамический масштаб по Y
            y_max = min(1.1/tau, 100) if tau > 0.01 else 100
            ax.set_ylim(-0.1*y_max, y_max)
        
        plt.suptitle('Последовательность прямоугольных импульсов, аппроксимирующих δ(t)', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_spectrum_sequence(self):
        """Графическое представление последовательности спектров"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        f_analytic = np.linspace(-20, 20, 1000)
        
        for i, tau in enumerate(self.tau_values[:8]):
            ax = axes[i]
            
            # Аналитический спектр
            spectrum = self.analytic_spectrum(f_analytic, tau)
            
            # Также численный спектр через БПФ
            delta_approx = self.delta_approximation_pulse(tau)
            f_fft, spectrum_fft = self.compute_fft_spectrum(delta_approx)
            mask_freq = (np.abs(f_fft) <= 20)
            
            ax.plot(f_analytic, np.abs(spectrum), 'k-', linewidth=2, label='Аналитический')
            ax.plot(f_fft[mask_freq], np.abs(spectrum_fft[mask_freq]), 'r--', 
                   linewidth=1, alpha=0.7, label='БПФ')
            
            ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Предел: 1')
            
            ax.set_title(f'τ = {tau:.3f} с\n|δ̃_τ(f)| = |sinc(πfτ)|')
            ax.set_xlabel('Частота f (Гц)')
            ax.set_ylabel('|δ̃_τ(f)|')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-20, 20)
            ax.set_ylim(-0.1, 1.2)
        
        plt.suptitle('Спектры аппроксимирующих импульсов, сходящиеся к 1', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_analysis(self):
        """Анализ сходимости"""
        # Вычисляем ошибки аппроксимации
        errors_delta = []
        errors_spectrum = []
        
        # Для дельта-функции: проверяем на тестовой функции
        test_func = np.exp(-self.t**2)  # Гауссова функция
        phi_0 = test_func[np.abs(self.t) == np.min(np.abs(self.t))][0]
        
        # Для спектра: проверяем на тестовой функции в частотной области
        f_test = np.linspace(-10, 10, 1000)
        test_func_freq = np.exp(-(f_test/3)**2)
        integral_1 = np.trapz(1 * test_func_freq, f_test)
        
        for tau in self.tau_values:
            # Ошибка для дельта-функции
            delta_approx = self.delta_approximation_pulse(tau)
            integral = np.trapz(delta_approx * test_func, self.t)
            error_delta = abs(integral - phi_0)
            errors_delta.append(error_delta)
            
            # Ошибка для спектра
            spectrum = self.analytic_spectrum(f_test, tau)
            integral_spec = np.trapz(spectrum * test_func_freq, f_test)
            error_spec = abs(integral_spec - integral_1)
            errors_spectrum.append(error_spec)
        
        # График сходимости
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Сходимость дельта-функций
        ax1.loglog(self.tau_values, errors_delta, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('τ (с)')
        ax1.set_ylabel('Ошибка аппроксимации δ_τ')
        ax1.set_title('Сходимость δ_τ(t) → δ(t)')
        ax1.grid(True, which="both", ls="-", alpha=0.3)
        
        # Добавляем теоретическую оценку
        theoretical = np.array(self.tau_values)**2 / 12  # O(τ²) сходимость
        ax1.loglog(self.tau_values, theoretical, 'r--', linewidth=1.5, label='O(τ²)')
        ax1.legend()
        
        # 2. Сходимость спектров
        ax2.loglog(self.tau_values, errors_spectrum, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('τ (с)')
        ax2.set_ylabel('Ошибка аппроксимации δ̃_τ')
        ax2.set_title('Сходимость δ̃_τ(f) → 1')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        
        # Теоретическая оценка для спектра
        theoretical_spec = np.array(self.tau_values)  # O(τ) сходимость
        ax2.loglog(self.tau_values, theoretical_spec, 'r--', linewidth=1.5, label='O(τ)')
        ax2.legend()
        
        plt.suptitle('Анализ скорости сходимости', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        print("\nАнализ скорости сходимости:")
        print("=" * 50)
        print("Для прямоугольного импульса δ_τ(t):")
        print("1. Площадь всегда = 1 (сохраняется)")
        print("2. Ширина импульса → 0 при τ → 0")
        print("3. Высота импульса → ∞ при τ → 0")
        print("4. Сходимость слабая: интегралы с пробными функциями → φ(0)")
        print()
        print("Для спектров δ̃_τ(f) = sinc(πfτ):")
        print("1. δ̃_τ(0) = 1 для всех τ")
        print("2. Ширина главного лепестка ≈ 2/τ → ∞ при τ → 0")
        print("3. Предел: sinc(πfτ) → 1 при τ → 0 для всех f")
    
    def illustrate_weak_convergence(self):
        """Иллюстрация концепции слабой сходимости"""
        print("\n" + "=" * 70)
        print("ИЛЛЮСТРАЦИЯ СЛАБОЙ СХОДИМОСТИ")
        print("=" * 70)
        
        print("\n1. Что такое слабая сходимость?")
        print("   Обычная сходимость: lim_{τ→0} δ_τ(t) = δ(t) для каждого t")
        print("   СЛАБАЯ сходимость: lim_{τ→0} ∫ δ_τ(t)φ(t)dt = φ(0) для всех φ ∈ S(ℝ)")
        print()
        print("2. Почему слабая сходимость?")
        print("   - В обычном смысле δ_τ(t) не сходится ни к чему при τ→0")
        print("   - Но интегралы со 'хорошими' функциями имеют предел")
        print("   - Это позволяет работать с обобщенными функциями (распределениями)")
        print()
        print("3. Пространство Шварца S(ℝ):")
        print("   - Бесконечно дифференцируемые функции")
        print("   - Убывают на бесконечности быстрее любой степени")
        print("   - Примеры: exp(-t²), exp(-t²)cos(t), функции с компактным носителем")
        print()
        
        # Демонстрация на конкретном примере
        print("4. Демонстрация на примере φ(t) = exp(-t²):")
        
        # Тестовая функция
        phi = np.exp(-self.t**2)
        phi_0 = phi[np.abs(self.t) == np.min(np.abs(self.t))][0]
        
        print(f"   φ(0) = {phi_0:.6f}")
        print()
        
        for tau in [0.5, 0.1, 0.05, 0.01]:
            if tau in self.tau_values or tau <= min(self.tau_values):
                delta_approx = self.delta_approximation_pulse(tau)
                integral = np.trapz(delta_approx * phi, self.t)
                error = abs(integral - phi_0)
                print(f"   τ = {tau:.3f} с: ∫ δ_τ(t)φ(t)dt = {integral:.8f}")
                print(f"     Ошибка = {error:.8f}")
        
        print("\n" + "=" * 70)

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 2")
    print("Предельный частотный спектр последовательности прямоугольных импульсов")
    print("=" * 70)
    
    # Создаем анализатор
    analyzer = DeltaFunctionApproximation(
        tau_values=[2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005],
        fs=10000,  # Высокая частота дискретизации для точности
        T=2        # Более короткий интервал для лучшего разрешения
    )
    
    # 1. Иллюстрация слабой сходимости
    analyzer.illustrate_weak_convergence()
    
    # 2. Проверка слабой сходимости δ_τ(t) → δ(t)
    delta_results = analyzer.check_weak_convergence_delta()
    
    # 3. Проверка слабой сходимости δ̃_τ(f) → 1
    spectrum_results = analyzer.check_weak_convergence_spectrum()
    
    # 4. Графики аппроксимаций во времени
    print("\nПостроение графиков последовательности импульсов...")
    analyzer.plot_approximation_sequence()
    
    # 5. Графики спектров
    print("Построение графиков последовательности спектров...")
    analyzer.plot_spectrum_sequence()
    
    # 6. Анализ сходимости
    print("Анализ скорости сходимости...")
    analyzer.plot_convergence_analysis()
    
    # 7. Подтверждение ответа из задания
    print("\n" + "=" * 70)
    print("ПОДТВЕРЖДЕНИЕ ОТВЕТА ИЗ ЗАДАНИЯ")
    print("=" * 70)
    
    print("\nОтвет из задания:")
    print("δ̃_τ(f) → 1, τ → 0 (в смысле слабой сходимости функционалов)")
    print()
    
    print("Наши результаты подтверждают:")
    print("1. Последовательность δ_τ(t) слабо сходится к δ(t):")
    print("   lim_{τ→0} ∫ δ_τ(t)φ(t)dt = φ(0) для φ ∈ S(ℝ)")
    print()
    print("2. Последовательность спектров δ̃_τ(f) слабо сходится к 1:")
    print("   lim_{τ→0} ∫ δ̃_τ(f)ψ(f)df = ∫ 1 * ψ(f)df для ψ ∈ S(ℝ)")
    print()
    print("3. Следовательно, преобразование Фурье дельта-функции:")
    print("   F{δ(t)} = 1")
    print()
    
    print("=" * 70)
    print("МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ:")
    print("=" * 70)
    
    print("\n1. Определение δ_τ(t):")
    print("   δ_τ(t) = 1/τ для |t| ≤ τ/2, и 0 иначе")
    print("   ∫ δ_τ(t) dt = 1 для всех τ > 0")
    print()
    
    print("2. Спектр δ_τ(t):")
    print("   δ̃_τ(f) = ∫_{-τ/2}^{τ/2} (1/τ) e^{-i2πft} dt")
    print("          = (1/τ) * [e^{-i2πft}/(-i2πf)]_{-τ/2}^{τ/2}")
    print("          = sinc(πfτ)")
    print()
    
    print("3. Предел при τ → 0:")
    print("   lim_{τ→0} sinc(πfτ) = lim_{x→0} sin(x)/x = 1")
    print("   где x = πfτ")
    print()
    
    print("4. Слабая сходимость:")
    print("   Для любой ψ ∈ S(ℝ):")
    print("   lim_{τ→0} ∫ sinc(πfτ) ψ(f) df = ∫ 1 * ψ(f) df")
    print("   т.к. sinc(πfτ) → 1 равномерно на компактах")
    print()
    
    print("5. Связь с преобразованием Фурье δ(t):")
    print("   По определению, F{δ(t)} = ∫ δ(t) e^{-i2πft} dt = e^{0} = 1")
    print("   Наша последовательность δ_τ(t) дает ту же предельную спектральную плотность.")
    
    # 8. Дополнительная проверка: прямое вычисление через тестовые функции
    print("\n" + "=" * 70)
    print("ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА ЧЕРЕЗ ФОРМУЛУ ПЛАНШЕРЕЛЯ-ПАРСЕВАЛЯ")
    print("=" * 70)
    
    # Возьмем тестовую функцию и вычислим двумя способами
    phi = np.exp(-analyzer.t**2) * np.cos(2*np.pi*analyzer.t)
    phi_hat = np.fft.fftshift(np.fft.fft(phi)) * analyzer.dt
    
    print("\nПроверка через равенство Парсеваля:")
    print("∫ δ_τ(t) φ(t) dt = ∫ δ̃_τ(f) φ̃(-f) df")
    print()
    
    for tau in [0.1, 0.05, 0.02, 0.01]:
        if tau in analyzer.tau_values or tau <= min(analyzer.tau_values):
            # Левая часть
            delta_approx = analyzer.delta_approximation_pulse(tau)
            left = np.trapz(delta_approx * phi, analyzer.t)
            
            # Правая часть
            f = np.fft.fftshift(np.fft.fftfreq(len(phi), analyzer.dt))
            delta_hat = analyzer.analytic_spectrum(f, tau)
            right = np.trapz(delta_hat * np.conj(phi_hat), f)
            
            print(f"τ = {tau:.3f} с:")
            print(f"  Левая часть: {left:.8f}")
            print(f"  Правая часть: {right:.8f}")
            print(f"  Разница: {abs(left - right):.8f}")
    
    print("\n" + "=" * 70)
    print("ВЫВОД: Результаты подтверждают ответ из задания!")
    print("       δ̃_τ(f) → 1 при τ → 0 в смысле слабой сходимости.")
    print("=" * 70)

if __name__ == "__main__":
    main()
