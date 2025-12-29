import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

class GaussianPulseFourierTransform:
    """
    Исследование преобразования Фурье гауссовского импульса
    и принципа дуальности
    """
    
    def __init__(self, tau_values=None, fs=5000, T=20):
        """
        Инициализация параметров
        
        Parameters:
        tau_values: значения τ для исследования
        fs: частота дискретизации (Гц)
        T: общее время наблюдения (с)
        """
        if tau_values is None:
            tau_values = [0.5, 1.0, 2.0, 4.0, 8.0]
        
        self.tau_values = tau_values
        self.fs = fs
        self.T = T
        
        # Временная ось
        self.t = np.linspace(-T/2, T/2, int(T*fs), endpoint=False)
        self.dt = self.t[1] - self.t[0]
        
        # Частотная ось
        self.f = np.linspace(-10, 10, 2000)
    
    def gaussian_pulse(self, t, tau):
        """
        Гауссовский импульс
        
        g_τ(t) = exp(-π(t/τ)²)
        """
        return np.exp(-np.pi * (t/tau)**2)
    
    def analytic_fourier_gaussian(self, f, tau):
        """
        Аналитическое преобразование Фурье гауссовского импульса
        
        ĝ_τ(f) = τ * exp(-π(fτ)²)
        """
        return tau * np.exp(-np.pi * (f * tau)**2)
    
    def demonstrate_fourier_transform(self):
        """
        Демонстрация преобразования Фурье гауссовского импульса
        """
        print("=" * 80)
        print("ПРЕОБРАЗОВАНИЕ ФУРЬЕ ГАУССОВСКОГО ИМПУЛЬСА")
        print("=" * 80)
        
        print("\nУказание из задания:")
        print("Подынтегральную функцию прямого преобразования Фурье")
        print("представить в виде exp(-π(t/τ - iτf)²) * exp(-π(τf)²)")
        print("и воспользоваться условием нормировки гауссова:")
        print("∫ exp(-π(t/τ - α)²) dt = τ, для любого α ∈ ℂ")
        print()
        
        print("Гауссовский импульс:")
        print("g_τ(t) = exp(-π(t/τ)²)")
        print()
        
        print("Его преобразование Фурье:")
        print("ĝ_τ(f) = τ * exp(-π(fτ)²)")
        print()
        
        print("Математический вывод:")
        print("1. Прямое преобразование Фурье:")
        print("   ĝ_τ(f) = ∫ exp(-π(t/τ)²) * exp(-i2πft) dt")
        print()
        
        print("2. Выделение полного квадрата:")
        print("   -π(t/τ)² - i2πft = -π[(t/τ)² + i2τft]")
        print("   = -π[(t/τ + iτf)² + (τf)²]")
        print("   = -π(t/τ + iτf)² - π(τf)²")
        print()
        
        print("3. Подстановка:")
        print("   ĝ_τ(f) = exp(-π(τf)²) * ∫ exp(-π(t/τ + iτf)²) dt")
        print()
        
        print("4. Использование нормировки гауссова:")
        print("   ∫ exp(-π(t/τ + iτf)²) dt = τ")
        print("   (интеграл не зависит от мнимого сдвига)")
        print()
        
        print("5. Результат:")
        print("   ĝ_τ(f) = τ * exp(-π(τf)²)")
        print()
        
        # Проверка для нескольких значений τ
        print("Численная проверка:")
        print("-" * 60)
        print(f"{'τ (с)':<10} {'g_τ(0)':<15} {'ĝ_τ(0)':<15} {'Соотношение':<15}")
        print("-" * 60)
        
        for tau in self.tau_values:
            g0 = self.gaussian_pulse(0, tau)
            g_hat0 = self.analytic_fourier_gaussian(0, tau)
            ratio = g_hat0 / g0
            
            print(f"{tau:<10.3f} {g0:<15.6f} {g_hat0:<15.6f} {ratio:<15.6f}")
        
        print("-" * 60)
        print("Примечание: g_τ(0) = 1, ĝ_τ(0) = τ")
        
        # Графическая демонстрация
        self.plot_gaussian_fourier_transform()
    
    def plot_gaussian_fourier_transform(self):
        """Графическая демонстрация преобразования Фурье"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.tau_values)))
        
        # 1. Гауссовские импульсы во времени
        ax1 = axes[0, 0]
        
        for i, tau in enumerate(self.tau_values):
            gaussian = self.gaussian_pulse(self.t, tau)
            mask = (np.abs(self.t) <= 3*tau)
            ax1.plot(self.t[mask], gaussian[mask], color=colors[i], 
                    linewidth=2, label=f'τ = {tau} с')
        
        ax1.set_title('Гауссовские импульсы во времени')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('g_τ(t) = exp(-π(t/τ)²)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-10, 10)
        
        # 2. Их преобразования Фурье
        ax2 = axes[0, 1]
        
        for i, tau in enumerate(self.tau_values):
            gaussian_fourier = self.analytic_fourier_gaussian(self.f, tau)
            ax2.plot(self.f, gaussian_fourier, color=colors[i], 
                    linewidth=2, label=f'τ = {tau} с')
        
        ax2.set_title('Преобразования Фурье')
        ax2.set_xlabel('Частота f (Гц)')
        ax2.set_ylabel('ĝ_τ(f) = τ·exp(-π(fτ)²)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-5, 5)
        
        # 3. Ширина по уровню 1/e
        ax3 = axes[0, 2]
        
        time_widths = []
        freq_widths = []
        
        for tau in self.tau_values:
            # Ширина во времени: g_τ(t) = exp(-π(t/τ)²) = 1/e при t = τ/√π
            time_width = tau / np.sqrt(np.pi)
            time_widths.append(time_width)
            
            # Ширина в частоте: ĝ_τ(f) = τ·exp(-π(fτ)²) = τ/e при f = 1/(τ√π)
            freq_width = 1/(tau * np.sqrt(np.pi))
            freq_widths.append(freq_width)
        
        ax3.loglog(self.tau_values, time_widths, 'bo-', 
                  linewidth=2, markersize=8, label='Ширина во времени')
        ax3.loglog(self.tau_values, freq_widths, 'ro-', 
                  linewidth=2, markersize=8, label='Ширина в частоте')
        ax3.set_xlabel('τ (с)')
        ax3.set_ylabel('Ширина по уровню 1/e')
        ax3.set_title('Ширина импульса и его спектра')
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
        
        # 4. Произведение ширины во времени и частоте
        ax4 = axes[1, 0]
        
        uncertainty_product = []
        for tw, fw in zip(time_widths, freq_widths):
            uncertainty_product.append(tw * fw)
        
        ax4.plot(self.tau_values, uncertainty_product, 'go-', 
                linewidth=2, markersize=8)
        ax4.axhline(y=1/np.pi, color='r', linestyle='--', 
                   label=f'Теоретическое: 1/π ≈ {1/np.pi:.4f}')
        ax4.set_xlabel('τ (с)')
        ax4.set_ylabel('Δt·Δf')
        ax4.set_title('Произведение неопределённостей')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Нормированные импульсы
        ax5 = axes[1, 1]
        
        t_norm = np.linspace(-3, 3, 1000)
        
        for i, tau in enumerate(self.tau_values[:3]):  # Первые 3 для наглядности
            # Нормировка по площади
            gaussian_norm = self.gaussian_pulse(t_norm * tau, tau)
            area = np.trapz(gaussian_norm, t_norm * tau)
            gaussian_norm = gaussian_norm / area
            
            ax5.plot(t_norm, gaussian_norm, color=colors[i], 
                    linewidth=2, label=f'τ = {tau} с')
        
        ax5.set_title('Нормированные импульсы (единичная площадь)')
        ax5.set_xlabel('Нормированное время t/τ')
        ax5.set_ylabel('g_τ(t) / ∫g_τ(t)dt')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-3, 3)
        
        # 6. Энергия (площадь под квадратом)
        ax6 = axes[1, 2]
        
        energies = []
        
        for tau in self.tau_values:
            gaussian = self.gaussian_pulse(self.t, tau)
            energy = np.trapz(gaussian**2, self.t)
            energies.append(energy)
        
        ax6.plot(self.tau_values, energies, 'mo-', 
                linewidth=2, markersize=8)
        ax6.set_xlabel('τ (с)')
        ax6.set_ylabel('Энергия ∫g_τ²(t)dt')
        ax6.set_title('Энергия гауссовских импульсов')
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        
        plt.suptitle('Преобразование Фурье гауссовского импульса', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def verify_norm_invariance(self):
        """
        Проверка инвариантности нормировки при комплексном сдвиге
        """
        print("\n" + "=" * 80)
        print("ПРОВЕРКА ИНВАРИАНТНОСТИ НОРМИРОВКИ ГАУССОВА")
        print("=" * 80)
        
        print("\nУсловие нормировки из указания:")
        print("∫ exp(-π(t/τ - α)²) dt = τ, для любого α ∈ ℂ")
        print()
        
        print("Проверим численно для различных α:")
        print("-" * 70)
        print(f"{'α':<15} {'Действительная часть':<25} {'Мнимая часть':<25}")
        print("-" * 70)
        
        tau = 2.0
        t_vals = np.linspace(-20, 20, 100000)
        dt = t_vals[1] - t_vals[0]
        
        alpha_values = [
            0.0,                    # вещественный 0
            1.0,                    # вещественный положительный
            -1.0,                   # вещественный отрицательный
            0.5 + 0.5j,            # комплексный с малой мнимой частью
            2.0 + 1.0j,            # комплексный с заметной мнимой частью
            0.0 + 3.0j             # чисто мнимый
        ]
        
        for alpha in alpha_values:
            integrand = np.exp(-np.pi * ((t_vals/tau - alpha)**2))
            integral = np.trapz(integrand, t_vals)
            
            print(f"{str(alpha):<15} {integral.real:<25.6f} {integral.imag:<25.6f}")
        
        print("-" * 70)
        print(f"Теоретическое значение: {tau}")
        print()
        
        print("Физический смысл:")
        print("Интеграл гауссовой функции с комплексным сдвигом")
        print("не зависит от сдвига α. Это ключевое свойство,")
        print("используемое при выводе преобразования Фурье.")
        
        # Графическая иллюстрация
        self.plot_norm_invariance()
    
    def plot_norm_invariance(self):
        """Графическая иллюстрация инвариантности нормировки"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        tau = 2.0
        t_vals = np.linspace(-10, 10, 1000)
        
        # 1. Вещественные сдвиги
        ax1 = axes[0, 0]
        
        real_shifts = [0.0, 0.5, 1.0, 2.0]
        colors_real = ['b', 'g', 'r', 'm']
        
        for shift, color in zip(real_shifts, colors_real):
            integrand = np.exp(-np.pi * ((t_vals/tau - shift)**2))
            ax1.plot(t_vals, integrand, color=color, 
                    linewidth=2, label=f'α = {shift}')
        
        ax1.set_title('Гауссовы функции с вещественным сдвигом')
        ax1.set_xlabel('Время t (с)')
        ax1.set_ylabel('exp(-π(t/τ - α)²)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-6, 6)
        
        # 2. Мнимые сдвиги (действительная часть)
        ax2 = axes[0, 1]
        
        imag_shifts = [0.0, 0.5j, 1.0j, 2.0j]
        colors_imag = ['b', 'g', 'r', 'm']
        
        for shift, color in zip(imag_shifts, colors_imag):
            integrand = np.exp(-np.pi * ((t_vals/tau - shift)**2))
            ax2.plot(t_vals, integrand.real, color=color, 
                    linewidth=2, label=f'α = {shift}')
        
        ax2.set_title('Гауссовы функции с мнимым сдвигом (Re)')
        ax2.set_xlabel('Время t (с)')
        ax2.set_ylabel('Re[exp(-π(t/τ - α)²)]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-6, 6)
        
        # 3. Мнимые сдвиги (мнимая часть)
        ax3 = axes[1, 0]
        
        for shift, color in zip(imag_shifts, colors_imag):
            integrand = np.exp(-np.pi * ((t_vals/tau - shift)**2))
            ax3.plot(t_vals, integrand.imag, color=color, 
                    linewidth=2, label=f'α = {shift}')
        
        ax3.set_title('Гауссовы функции с мнимым сдвигом (Im)')
        ax3.set_xlabel('Время t (с)')
        ax3.set_ylabel('Im[exp(-π(t/τ - α)²)]')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-6, 6)
        
        # 4. Площади под кривыми
        ax4 = axes[1, 1]
        
        # Проверим для разных α
        alpha_test = np.linspace(-3, 3, 50)
        integrals_real = []
        integrals_imag = []
        
        for alpha in alpha_test:
            # Комплексный α с небольшой мнимой частью
            alpha_complex = alpha + 0.5j
            integrand = np.exp(-np.pi * ((t_vals/tau - alpha_complex)**2))
            integral = np.trapz(integrand, t_vals)
            integrals_real.append(integral.real)
            integrals_imag.append(integral.imag)
        
        ax4.plot(alpha_test, integrals_real, 'b-', 
                linewidth=2, label='Действительная часть')
        ax4.plot(alpha_test, integrals_imag, 'r-', 
                linewidth=2, label='Мнимая часть')
        ax4.axhline(y=tau, color='k', linestyle='--', 
                   label=f'Теоретическое: τ = {tau}')
        ax4.set_xlabel('Re(α)')
        ax4.set_ylabel('Значение интеграла')
        ax4.set_title('Инвариантность интеграла относительно α')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Инвариантность нормировки гауссовой функции', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_duality_principle(self):
        """
        Демонстрация принципа дуальности частоты и времени
        """
        print("\n" + "=" * 80)
        print("ПРИНЦИП ДУАЛЬНОСТИ ЧАСТОТЫ И ВРЕМЕНИ")
        print("=" * 80)
        
        print("\nЗамечание из задания:")
        print("Если в обозначениях переменные t и f, обозначающие")
        print("время и частоту, всюду поменять местами, то получим")
        print("спектральные представления новых сигналов.")
        print()
        
        print("Это отражает принцип дуальности частоты и времени")
        print("(другое проявление этого принципа выражается")
        print("теоремой Парсеваля-Планшереля).")
        print()
        
        print("Применим этот принцип к полученным формулам:")
        print()
        
        # Перепишем формулы с заменой t ↔ f
        print("1. Исходная формула (гауссовский импульс):")
        print("   g_τ(t) = exp(-π(t/τ)²)")
        print("   ĝ_τ(f) = τ·exp(-π(fτ)²)")
        print()
        
        print("2. После замены t ↔ f:")
        print("   Пусть τ' = 1/τ (новая константа)")
        print("   g̃_τ'(f) = exp(-π(f/τ')²)")
        print("   ǧ̃_τ'(t) = τ'·exp(-π(tτ')²)")
        print()
        
        print("3. Новая интерпретация:")
        print("   g̃_τ'(f) - спектральная характеристика")
        print("   ǧ̃_τ'(t) - соответствующий временной сигнал")
        print()
        
        print("4. Свойство самодуальности гауссова импульса:")
        print("   Если τ = 1/√(2π), то форма одинакова:")
        print("   g(t) = exp(-2πt²)")
        print("   ĝ(f) = exp(-2πf²)")
        print("   (с точностью до нормировочного множителя)")
        print()
        
        # Демонстрация самодуальности
        print("Численная проверка самодуальности:")
        print("-" * 60)
        
        tau_special = 1/np.sqrt(2*np.pi)
        t_vals = np.linspace(-3, 3, 1000)
        f_vals = np.linspace(-3, 3, 1000)
        
        # Гауссов в времени
        g_t = self.gaussian_pulse(t_vals, tau_special)
        
        # Гауссов в частоте
        g_f = self.analytic_fourier_gaussian(f_vals, tau_special)
        
        # Нормируем для сравнения
        g_t_norm = g_t / np.max(g_t)
        g_f_norm = g_f / np.max(g_f)
        
        # Сравнение в нескольких точках
        test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        print(f"{'Координата':<15} {'g(t) (норм.)':<15} {'ĝ(f) (норм.)':<15} {'Разница':<15}")
        print("-" * 60)
        
        for point in test_points:
            idx_t = np.argmin(np.abs(t_vals - point))
            idx_f = np.argmin(np.abs(f_vals - point))
            
            val_t = g_t_norm[idx_t]
            val_f = g_f_norm[idx_f]
            diff = abs(val_t - val_f)
            
            print(f"{point:<15.3f} {val_t:<15.6f} {val_f:<15.6f} {diff:<15.6f}")
        
        print("-" * 60)
        
        # Графическая демонстрация
        self.plot_duality_principle()
    
    def plot_duality_principle(self):
        """Графическая демонстрация принципа дуальности"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Исходный гауссов и его преобразование
        ax1 = axes[0, 0]
        
        tau = 2.0
        t_vals = np.linspace(-5, 5, 1000)
        f_vals = np.linspace(-5, 5, 1000)
        
        g_t = self.gaussian_pulse(t_vals, tau)
        g_f = self.analytic_fourier_gaussian(f_vals, tau)
        
        ax1.plot(t_vals, g_t, 'b-', linewidth=2, label='g_τ(t)')
        ax1.plot(f_vals, g_f, 'r-', linewidth=2, label='ĝ_τ(f)')
        ax1.set_title(f'Исходный импульс и его спектр (τ = {tau} с)')
        ax1.set_xlabel('Время/частота')
        ax1.set_ylabel('Амплитуда')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-3, 3)
        
        # 2. После замены t ↔ f
        ax2 = axes[0, 1]
        
        tau_prime = 1/tau
        g_tilde_f = self.gaussian_pulse(f_vals, tau_prime)
        g_tilde_t = self.analytic_fourier_gaussian(t_vals, tau_prime)
        
        ax2.plot(f_vals, g_tilde_f, 'b-', linewidth=2, label='g̃_τ\'(f)')
        ax2.plot(t_vals, g_tilde_t, 'r-', linewidth=2, label='ǧ̃_τ\'(t)')
        ax2.set_title(f'После замены t↔f (τ\' = 1/τ = {tau_prime:.3f} с)')
        ax2.set_xlabel('Частота/время')
        ax2.set_ylabel('Амплитуда')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-3, 3)
        
        # 3. Сравнение форм
        ax3 = axes[0, 2]
        
        # Нормированные для сравнения
        g_t_norm = g_t / np.max(g_t)
        g_f_norm = g_f / np.max(g_f)
        g_tilde_f_norm = g_tilde_f / np.max(g_tilde_f)
        g_tilde_t_norm = g_tilde_t / np.max(g_tilde_t)
        
        ax3.plot(t_vals, g_t_norm, 'b-', linewidth=2, label='g_τ(t) (норм.)')
        ax3.plot(f_vals, g_f_norm, 'r--', linewidth=2, label='ĝ_τ(f) (норм.)')
        ax3.plot(f_vals, g_tilde_f_norm, 'g:', linewidth=2, label='g̃_τ\'(f) (норм.)')
        ax3.plot(t_vals, g_tilde_t_norm, 'm-.', linewidth=2, label='ǧ̃_τ\'(t) (норм.)')
        
        ax3.set_title('Сравнение нормированных форм')
        ax3.set_xlabel('Аргумент')
        ax3.set_ylabel('Нормированная амплитуда')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-3, 3)
        
        # 4. Самодуальный случай
        ax4 = axes[1, 0]
        
        tau_selfdual = 1/np.sqrt(2*np.pi)
        g_selfdual_t = self.gaussian_pulse(t_vals, tau_selfdual)
        g_selfdual_f = self.analytic_fourier_gaussian(f_vals, tau_selfdual)
        
        # Нормируем к максимуму = 1
        g_selfdual_t_norm = g_selfdual_t / np.max(g_selfdual_t)
        g_selfdual_f_norm = g_selfdual_f / np.max(g_selfdual_f)
        
        ax4.plot(t_vals, g_selfdual_t_norm, 'b-', linewidth=3, label='g(t) (норм.)')
        ax4.plot(f_vals, g_selfdual_f_norm, 'r--', linewidth=2, label='ĝ(f) (норм.)')
        ax4.set_title(f'Самодуальный случай (τ = {tau_selfdual:.4f} с)')
        ax4.set_xlabel('Время/частота')
        ax4.set_ylabel('Нормированная амплитуда')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-2, 2)
        
        # 5. Разница между нормированными функциями
        ax5 = axes[1, 1]
        
        # Интерполируем на общую сетку для сравнения
        common_grid = np.linspace(-2, 2, 1000)
        
        # Интерполяция
        from scipy.interpolate import interp1d
        interp_t = interp1d(t_vals, g_selfdual_t_norm, kind='cubic')
        interp_f = interp1d(f_vals, g_selfdual_f_norm, kind='cubic')
        
        g_t_interp = interp_t(common_grid)
        g_f_interp = interp_f(common_grid)
        
        diff = np.abs(g_t_interp - g_f_interp)
        
        ax5.semilogy(common_grid, diff, 'g-', linewidth=2)
        ax5.set_title('Разница между g(t) и ĝ(f) (нормир.)')
        ax5.set_xlabel('Аргумент')
        ax5.set_ylabel('|g(t) - ĝ(f)|')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-2, 2)
        
        # 6. Принцип неопределённости
        ax6 = axes[1, 2]
        
        tau_range = np.logspace(-1, 1, 50)  # от 0.1 до 10
        time_widths = []
        freq_widths = []
        products = []
        
        for tau_val in tau_range:
            # Ширина по уровню 1/e во времени
            time_width = tau_val / np.sqrt(np.pi)
            time_widths.append(time_width)
            
            # Ширина по уровню 1/e в частоте
            freq_width = 1/(tau_val * np.sqrt(np.pi))
            freq_widths.append(freq_width)
            
            # Произведение
            products.append(time_width * freq_width)
        
        ax6.loglog(tau_range, time_widths, 'b-', linewidth=2, label='Δt')
        ax6.loglog(tau_range, freq_widths, 'r-', linewidth=2, label='Δf')
        ax6.loglog(tau_range, products, 'g-', linewidth=3, label='Δt·Δf')
        ax6.axhline(y=1/np.pi, color='k', linestyle='--', 
                   label='1/π ≈ 0.3183')
        ax6.set_xlabel('τ (с)')
        ax6.set_ylabel('Ширина')
        ax6.set_title('Принцип неопределённости для гауссова')
        ax6.legend()
        ax6.grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Принцип дуальности частоты и времени', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
    
    def rewrite_all_formulas_with_duality(self):
        """
        Переписывание всех полученных формул с заменой t ↔ f
        в соответствии с замечанием
        """
        print("\n" + "=" * 80)
        print("ПЕРЕПИСЫВАНИЕ ФОРМУЛ С УЧЁТОМ ДУАЛЬНОСТИ")
        print("=" * 80)
        
        print("\nЗамечание из задания:")
        print("Если в обозначениях переменные t и f поменять местами,")
        print("то получим спектральные представления новых сигналов.")
        print()
        
        print("Перепишем все полученные формулы:")
        print()
        
        print("1. Прямоугольный импульс (симметричный):")
        print("   Исходно: x₀(t) = 1_{[-τ/2, τ/2]}(t)")
        print("            X₀(f) = τ·sinc(fτ)")
        print("   После замены: X̃₀(t) = 1_{[-τ/2, τ/2]}(f)")
        print("               x̃₀(f) = τ·sinc(tτ)")
        print()
        
        print("2. Дельта-функция:")
        print("   Исходно: δ(t) ↔ 1")
        print("   После замены: 1 ↔ δ(f)")
        print("   (это известное свойство)")
        print()
        
        print("3. Комплексная экспонента:")
        print("   Исходно: exp(i2πf₀t) ↔ δ(f - f₀)")
        print("   После замены: δ(t - t₀) ↔ exp(-i2πft₀)")
        print("   (теорема о сдвиге)")
        print()
        
        print("4. Знаковая функция:")
        print("   Исходно: sgn(t) ↔ -i/(πf)")
        print("   После замены: -i/(πt) ↔ sgn(f)")
        print()
        
        print("5. Функция Хевисайда:")
        print("   Исходно: θ(t) ↔ 1/2[δ(f) + 1/(iπf)]")
        print("   После замены: 1/2[δ(t) + 1/(iπt)] ↔ θ(f)")
        print()
        
        print("6. Гауссовский импульс:")
        print("   Исходно: g_τ(t) = exp(-π(t/τ)²)")
        print("            ĝ_τ(f) = τ·exp(-π(fτ)²)")
        print("   После замены: g̃_τ'(f) = exp(-π(f/τ')²)")
        print("               ǧ̃_τ'(t) = τ'·exp(-π(tτ')²)")
        print("   где τ' = 1/τ")
        print()
        
        print("Физический смысл:")
        print("Новые формулы описывают сигналы, которые являются")
        print("преобразованиями Фурье исходных сигналов.")
        print("Это проявление дуальности между временным")
        print("и частотным представлениями сигналов.")
        print()
        
        # Демонстрация на примере
        print("Пример: прямоугольный спектр ↔ sinc-импульс")
        print("Если X₀(f) - прямоугольная спектральная характеристика,")
        print("то соответствующий сигнал x₀(t) имеет форму sinc.")
        print("Это основа многих методов цифровой обработки сигналов.")

# Основная программа
def main():
    print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ 8")
    print("Преобразование Фурье гауссовского импульса и принцип дуальности")
    print("=" * 80)
    
    # Создаем анализатор
    analyzer = GaussianPulseFourierTransform(
        tau_values=[0.5, 1.0, 2.0, 4.0, 8.0],
        fs=10000,
        T=20
    )
    
    # 1. Демонстрация преобразования Фурье
    print("\n1. Демонстрация преобразования Фурье гауссовского импульса...")
    analyzer.demonstrate_fourier_transform()
    
    # 2. Проверка инвариантности нормировки
    print("\n2. Проверка инвариантности нормировки гауссова...")
    analyzer.verify_norm_invariance()
    
    # 3. Демонстрация принципа дуальности
    print("\n3. Демонстрация принципа дуальности частоты и времени...")
    analyzer.demonstrate_duality_principle()
    
    # 4. Переписывание формул с учётом дуальности
    print("\n4. Переписывание формул с учётом дуальности...")
    analyzer.rewrite_all_formulas_with_duality()
    
    # 5. Итоговые выводы
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ ПО ЗАДАНИЮ 8")
    print("=" * 80)
    
    print("\n1. Преобразование Фурье гауссовского импульса:")
    print("   g_τ(t) = exp(-π(t/τ)²)")
    print("   ĝ_τ(f) = τ·exp(-π(fτ)²)")
    print()
    
    print("2. Ключевое свойство для вывода:")
    print("   ∫ exp(-π(t/τ - α)²) dt = τ, для любого α ∈ ℂ")
    print("   (инвариантность относительно комплексного сдвига)")
    print()
    
    print("3. Метод выделения полного квадрата:")
    print("   -π(t/τ)² - i2πft = -π[(t/τ + iτf)² + (τf)²]")
    print("   Это позволяет свести интеграл к нормировочному.")
    print()
    
    print("4. Принцип дуальности частоты и времени:")
    print("   Замена t ↔ f даёт новые пары преобразований Фурье.")
    print("   Для гауссова: g(t) и ĝ(f) имеют одинаковую форму")
    print("   (с точностью до масштабирования).")
    print()
    
    print("5. Свойство самодуальности:")
    print("   При τ = 1/√(2π) гауссов сохраняет форму")
    print("   при преобразовании Фурье (с точностью до множителя).")
    print()
    
    print("6. Принцип неопределённости:")
    print("   Δt·Δf ≥ 1/(4π)  (для гауссова достигается равенство)")
    print("   Гауссовский импульс имеет минимальное произведение")
    print("   неопределённостей.")
    print()
    
    print("7. Практическое значение:")
    print("   - Гауссовы импульсы используются в оптике и связи")
    print("   - Минимальная неопределённость делает их оптимальными")
    print("   - Самодуальность важна в квантовой механике")
    print("   - Принцип дуальности лежит в основе многих методов")
    print("     обработки сигналов")
    print()
    
    print("=" * 80)
    print("ЗАДАНИЕ ВЫПОЛНЕНО ПОЛНОСТЬЮ!")
    print("=" * 80)
    
    # Дополнительно: математический вывод
    print("\n" + "=" * 80)
    print("МАТЕМАТИЧЕСКИЙ ВЫВОД (для справки)")
    print("=" * 80)
    
    print("\nПолный вывод преобразования Фурье гауссова импульса:")
    print("1. Исходный интеграл:")
    print("   ĝ_τ(f) = ∫_{-∞}^{∞} exp(-π(t/τ)²) exp(-i2πft) dt")
    print()
    
    print("2. Выделение полного квадрата:")
    print("   -π(t/τ)² - i2πft")
    print("   = -π[(t/τ)² + i2τft]")
    print("   = -π[(t/τ + iτf)² - (iτf)²]")
    print("   = -π(t/τ + iτf)² - π(τf)²")
    print()
    
    print("3. Подстановка в интеграл:")
    print("   ĝ_τ(f) = exp(-π(τf)²) ∫_{-∞}^{∞} exp(-π(t/τ + iτf)²) dt")
    print()
    
    print("4. Замена переменной:")
    print("   u = t/τ + iτf,  du = dt/τ")
    print("   ∫ exp(-π(t/τ + iτf)²) dt = τ ∫ exp(-πu²) du")
    print()
    
    print("5. Использование нормировки гауссова:")
    print("   ∫_{-∞}^{∞} exp(-πu²) du = 1")
    print("   (стандартный гауссов интеграл)")
    print()
    
    print("6. Окончательный результат:")
    print("   ĝ_τ(f) = τ exp(-π(τf)²)")
    print()

if __name__ == "__main__":
    main()
