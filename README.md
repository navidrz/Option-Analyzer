**شروع**

هر ایده‌ی بزرگ از جرقه‌ای کوچک آغاز می‌شود، و این مدول نیز از همین باور برخاسته است؛ تلاشی برای پیوند میان دنیای پرچالش اختیار معامله‌ها و قدرت بی‌بدیل تحلیل داده‌ها. در طراحی این ابزار، دغدغه‌ی اصلی من این بود که با بهره‌گیری از دانش روز مالی و قابلیت‌های فناوری، ابزاری فراهم کنم که نه‌تنها مرزهای تحلیل را گسترش دهد، بلکه کاربردی، دقیق، و به‌روز باشد. این مدول، از ابتدا تا انتها، داستانی از تلفیق دانش مالی و برنامه‌نویسی است؛ قصه‌ای از جست‌وجو برای درک بهتر بازار و بهینه‌سازی تصمیم‌گیری‌های سرمایه‌گذاری.

---

## 📂 **مدول تحلیل اختیار معامله**

این مدول به‌منظور تحلیل و ارزیابی **اختیار معامله** از طریق شبیه‌سازی مونت کارلو و محاسبه شاخص‌های مالی مختلف طراحی شده است. در ادامه، به بررسی بخش‌های مختلف این کد می‌پردازیم. 


### 📚 **وارد کردن کتابخانه‌های مورد نیاز**

```python
import numpy as np
import pandas as pd
import multiprocessing
from typing import Dict, Tuple, Optional, Any
from scipy.stats import norm, skew, kurtosis, skewnorm
from scipy.optimize import minimize
from functools import partial
import logging
from gldpy import GLD  # اطمینان از نصب و وارد کردن صحیح gldpy
import matplotlib.pyplot as plt
```

**توضیحات:**

- **`numpy` و `pandas`**: برای پردازش داده‌های عددی و جدولی.
- **`multiprocessing`**: برای اجرای موازی شبیه‌سازی‌ها و افزایش کارایی.
- **`scipy.stats`**: برای توزیع‌های آماری مختلف.
- **`logging`**: برای ثبت لاگ‌ها و پیگیری فرآیندها.
- **`gldpy`**: برای تطبیق توزیع‌های عمومی (GLD).
- **`matplotlib.pyplot`**: برای ترسیم نمودارها.

### 📝 **پیکربندی لاگینگ**

```python
# پیکربندی لاگر
logger = logging.getLogger("OptionAnalysisLogger")
logger.setLevel(logging.DEBUG)  # تنظیم سطح لاگینگ به DEBUG برای ثبت دقیق‌تر
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```

**توضیحات:**

- **تعریف یک Logger**: با نام `"OptionAnalysisLogger"` برای ثبت لاگ‌ها.
- **سطح لاگینگ**: تنظیم شده به `DEBUG` برای دریافت تمامی جزئیات.
- **Handler و Formatter**: برای ارسال لاگ‌ها به خروجی استاندارد با قالب مشخص.

### 🌐 **تعریف سناریوهای بازار**

```python
# تعریف سناریوهای بازار
MARKET_SCENARIOS = {
    'بازار عادی': {'skewness': 0.0, 'kurtosis': 3.0},
    'بازار صعودی': {'skewness': -0.5, 'kurtosis': 3.5},
    'بازار نزولی': {'skewness': 0.5, 'kurtosis': 3.5},
    'بازار پرنوسان': {'skewness': 0.0, 'kurtosis': 5.0},
    'سقوط بازار': {'skewness': 2.0, 'kurtosis': 9.0},
    'رالی بازار': {'skewness': -2.0, 'kurtosis': 9.0}
}
```

**توضیحات:**

- **`skewness` (چولگی)**: عدم تقارن توزیع داده‌ها.
- **`kurtosis` (کشیدگی)**: کشیدگی توزیع نسبت به توزیع نرمال.

---

## 🏛️ **کلاس `OptionAnalyzer`**

این کلاس هسته اصلی تحلیل **اختیار معامله** را تشکیل می‌دهد و شامل متدهای مختلفی برای بارگذاری داده‌ها، شبیه‌سازی قیمت‌ها، محاسبه شاخص‌های مالی و ارائه توصیه‌ها است.

### 🔧 **تعریف کلاس و متد سازنده**

```python
class OptionAnalyzer:
    def __init__(self,
                 data_loader,  # نمونه‌ای از کلاس DataLoader
                 risk_free_rate: float = 0.01,
                 market_scenario: str = 'بازار عادی',
                 distribution_type: str = 'GLD'):
        """
        مقداردهی اولیه OptionAnalyzer با استفاده از DataLoader و پارامترهای تحلیل.
        
        Args:
            data_loader: نمونه‌ای از کلاس DataLoader.
            risk_free_rate (float): نرخ بهره بدون ریسک.
            market_scenario (str): سناریوی بازار برای شبیه‌سازی.
            distribution_type (str): نوع توزیع برای شبیه‌سازی قیمت‌ها ('GLD', 'SkewNormal', 'CornishFisher', و غیره).
        """
        self.data_loader = data_loader
        self.risk_free_rate = risk_free_rate
        self.market_scenario = market_scenario
        self.distribution_type = distribution_type
        self.cleaned_data = pd.DataFrame()
        self.historical_data = {}
        self.simulation_results = {}
        self.greeks = {}
        self.pop_results = {}
        self.breakeven_points = {}
        self.sharpe_ratios = {}
        self.var = {}
        self.cvar = {}
        self.recommendations_df = pd.DataFrame()
        self.target_date_distribution = {}
        self.cash_flows = {}
        self.variance = {}
        self.payout_ratios = {}
        self.market_views = {}
        self.strategy = 'default'
        self.metrics_min = {}
        self.metrics_max = {}
        self.thresholds = {}
        self.scenario_analysis_results = {}
        self.market_scenarios = MARKET_SCENARIOS
```

**توضیحات:**

- **پارامترهای ورودی**:
  - **`data_loader`**: برای بارگذاری و پردازش داده‌های اولیه.
  - **`risk_free_rate`**: نرخ بهره بدون ریسک.
  - **`market_scenario`**: سناریوی بازار انتخاب‌شده.
  - **`distribution_type`**: نوع توزیع مورد استفاده برای شبیه‌سازی قیمت‌ها.
  
- **متغیرهای داخلی**: شامل داده‌های پاک‌شده، داده‌های تاریخی، نتایج شبیه‌سازی، شاخص‌های یونانی، و سایر شاخص‌های مالی.

### 📦 **متد `prepare_data`**

```python
def prepare_data(self):
    """
    آماده‌سازی داده‌ها با بارگذاری داده‌های پاک‌شده و تاریخی از DataLoader.
    """
    try:
        # بارگذاری داده‌ها با استفاده از DataLoader
        self.cleaned_data, self.historical_data = self.data_loader.load_all_data()
        logger.info("آماده‌سازی داده‌ها تکمیل شد.")
        
        # محاسبه شاخص‌های اضافی
        self.calculate_volatility()
        self.calculate_moneyness()
    except Exception as e:
        logger.error(f"آماده‌سازی داده‌ها شکست خورد: {e}")
        raise
```

**توضیحات:**

- **بارگذاری داده‌ها**: با استفاده از `DataLoader`, داده‌های پاک‌شده و تاریخی بارگذاری می‌شوند.
- **محاسبه شاخص‌های اضافی**: نوسان (`Volatility`) و مونی‌نِس (`Moneyness`) محاسبه می‌شوند.

### 📈 **متد `calculate_volatility`**

```python
def calculate_volatility(self):
    """
    محاسبه نوسان تاریخی برای هر کد معامله منحصر به فرد.
    """
    logger.info("محاسبه نوسان تاریخی برای هر کد معامله.")
    volatility_dict = {}
    for ua_tse_code, df in self.historical_data.items():
        if df.empty:
            logger.warning(f"داده‌های تاریخی برای کد معامله {ua_tse_code} موجود نیست.")
            volatility_dict[ua_tse_code] = np.nan
            continue
        df = df.sort_values('date')
        df['daily_return'] = df['close'].pct_change()
        daily_std = df['daily_return'].std()
        annual_volatility = daily_std * np.sqrt(252)
        volatility_dict[ua_tse_code] = annual_volatility
        logger.debug(f"نوسان برای {ua_tse_code}: {annual_volatility:.4f}")
    
    self.cleaned_data['Volatility'] = self.cleaned_data['ua_tse_code'].map(volatility_dict)
    missing_vol = self.cleaned_data['Volatility'].isna().sum()
    if missing_vol > 0:
        mean_vol = self.cleaned_data['Volatility'].mean()
        self.cleaned_data['Volatility'].fillna(mean_vol, inplace=True)
        logger.warning(f"نوسان‌های ناقص با میانگین نوسان تکمیل شدند: {mean_vol:.4f}")
```

**محاسبه بازده روزانه:**

بازده روزانه = (قیمت پایانی امروز - قیمت پایانی دیروز) / قیمت پایانی دیروز

**تفسیر:**

این فرمول درصد تغییرات قیمت یک دارایی (مانند سهام) را از یک روز به روز دیگر نشان می‌دهد. به عبارت دیگر، نشان می‌دهد که قیمت دارایی در طول یک روز چه مقدار افزایش یا کاهش یافته است.

- **قیمت پایانی امروز**: قیمت نهایی دارایی در پایان روز مورد نظر
- **قیمت پایانی دیروز**: قیمت نهایی دارایی در پایان روز قبل

**محاسبه نوسان سالانه:**

نوسان سالانه = نوسان روزانه × √252

**تفسیر:**

این فرمول برای تبدیل نوسان روزانه یک دارایی به نوسان سالانه استفاده می‌شود. عدد 252 نشان‌دهنده تعداد روزهای معاملاتی تقریبی در یک سال است.

- **نوسان سالانه**: اندازه‌گیری میزان نوسانات قیمت یک دارایی در طول یک سال
- **نوسان روزانه**: اندازه‌گیری میزان نوسانات قیمت یک دارایی در طول یک روز

**چرا از √252 استفاده می‌کنیم؟**

از آنجایی که بازده‌های روزانه معمولاً با هم همبستگی کمی دارند، می‌توانیم از ویژگی‌های ریاضی برای تبدیل نوسان روزانه به نوسان سالانه استفاده کنیم. ریشه دوم تعداد روزهای معاملاتی در سال (√252) یک تقریب معمول برای این تبدیل است.

**مدیریت داده‌های ناقص:**

اگر برای برخی از کدهای معامله، نوسان محاسبه نشود (مثلاً به دلیل نبود داده کافی)، به جای آن از میانگین نوسان کلی استفاده می‌شود. این کار به این دلیل انجام می‌شود که از دست دادن اطلاعات برای برخی کدهای معامله، باعث کاهش دقت تحلیل کلی نشود. با استفاده از میانگین، یک تخمین محافظه‌کارانه از نوسان برای آن کدهای معامله در نظر گرفته می‌شود.

### 📉 **متد `calculate_moneyness`**

```python
def calculate_moneyness(self):
    """
    محاسبه مونی‌نِس بر اساس نوع اختیار معامله.
    """
    logger.info("محاسبه مونی‌نِس بر اساس نوع اختیار معامله.")
    self.cleaned_data['moneyness'] = np.where(
        self.cleaned_data['option_type'].str.upper() == 'CALL',
        self.cleaned_data['last_spot_price'] / self.cleaned_data['strike_price'],
        self.cleaned_data['strike_price'] / self.cleaned_data['last_spot_price']
    )
```

**توضیحات:**

**مفهوم مونی‌نِس:**

مونی‌نِس در بازارهای مالی، به ویژه در حوزه اختیار معامله، به رابطه بین قیمت فعلی یک دارایی پایه (مانند سهام، کالا، ارز و ...) و قیمت اجرای یک اختیار معامله اشاره دارد. این مفهوم به ما کمک می‌کند تا وضعیت یک اختیار معامله را از نظر سودآوری بالقوه ارزیابی کنیم.

**محاسبه مونی‌نِس:**

برای **اختیار معامله خرید (Call)**:

مونی‌نِس = قیمت دارایی پایه / قیمت اجرا

- اگر مقدار مونی‌نِس بزرگتر از 1 باشد، اختیار معامله "در سود" (In-the-Money) است. به این معنی که اگر اختیار معامله اعمال شود، خریدار سود خواهد کرد.
- اگر مقدار مونی‌نِس برابر با 1 باشد، اختیار معامله "در نقطه سربه‌سر" (At-the-Money) است. به این معنی که اگر اختیار معامله اعمال شود، خریدار نه سود می‌کند و نه ضرر می‌کند.
- اگر مقدار مونی‌نِس کمتر از 1 باشد، اختیار معامله "خارج از سود" (Out-of-the-Money) است. به این معنی که اگر اختیار معامله اعمال شود، خریدار ضرر خواهد کرد.

برای **اختیار معامله فروش (Put)**:

مونی‌نِس = قیمت اجرا / قیمت دارایی پایه

- اگر مقدار مونی‌نِس بزرگتر از 1 باشد، اختیار معامله "خارج از سود" (Out-of-the-Money) است.
- اگر مقدار مونی‌نِس برابر با 1 باشد، اختیار معامله "در نقطه سربه‌سر" (At-the-Money) است.
- اگر مقدار مونی‌نِس کمتر از 1 باشد، اختیار معامله "در سود" (In-the-Money) است.

**اهمیت مونی‌نِس:**

- **ارزیابی سودآوری**: مونی‌نِس به ما کمک می‌کند تا بفهمیم یک اختیار معامله چقدر بالقوه سودآور است.
- **استراتژی معاملاتی**: معامله‌گران از مونی‌نِس برای طراحی استراتژی‌های مختلف معاملاتی استفاده می‌کنند.
- **قیمت‌گذاری اختیار معامله**: مونی‌نِس یکی از عوامل مهم در مدل‌های قیمت‌گذاری اختیار معامله است.

### 🎲 **متد `simulate_price`**

```python
def simulate_price(self, S0: float, K: float, r: float, sigma: float, T: float, num_simulations: int,
                  distribution_type: str = 'GLD', skewness: float = 0.0, kurtosis_val: float = 3.0) -> Optional[np.ndarray]:
    """
    شبیه‌سازی قیمت آینده دارایی پایه با استفاده از توزیع مشخص.
    
    Args:
        S0 (float): قیمت فعلی دارایی.
        K (float): قیمت اجرای اختیار معامله.
        r (float): نرخ بهره بدون ریسک.
        sigma (float): نوسان.
        T (float): زمان باقی‌مانده تا سررسید (بر حسب سال).
        num_simulations (int): تعداد شبیه‌سازی‌ها.
        distribution_type (str): نوع توزیع برای شبیه‌سازی ('GLD', 'SkewNormal', 'CornishFisher', و غیره).
        skewness (float): چولگی مورد نظر.
        kurtosis_val (float): کشیدگی مورد نظر.
    
    Returns:
        Optional[np.ndarray]: قیمت‌های شبیه‌سازی‌شده در زمان سررسید.
    """
    try:
        if distribution_type == 'GLD':
            # استفاده از بازده‌های تاریخی برای تطبیق GLD
            # به دلیل پیچیدگی، از توزیع نرمال به عنوان جایگزین استفاده می‌شود
            Z = np.random.normal(0, 1, num_simulations) * sigma * np.sqrt(T)
            logger.debug("استفاده از توزیع نرمال به عنوان جایگزین برای GLD.")
        elif distribution_type == 'SkewNormal':
            a = skewness
            Z = skewnorm.rvs(a, size=num_simulations)
            logger.debug(f"تولید متغیرهای تصادفی SkewNormal با پارامتر چولگی: a={a}")
        elif distribution_type == 'CornishFisher':
            Z = np.random.normal(0, 1, num_simulations)
            Z = self.cornish_fisher_quantile(Z, skewness, kurtosis_val)
            logger.debug(f"تعدیل متغیرهای نرمال با استفاده از توسعه کرونیش-فیشر با چولگی={skewness} و کشیدگی={kurtosis_val}")
        else:
            # پیش‌فرض به توزیع نرمال
            Z = np.random.normal(0, 1, num_simulations)
            logger.debug("تولید متغیرهای تصادفی نرمال به عنوان پیش‌فرض.")
    
        # فرمول حرکت براونی هندسی
        drift = (r - 0.5 * sigma**2) * T
        diffusion = Z  # Z به‌طور مستقیم مقیاس‌دهی شده به sigma * sqrt(T)
        S_T = S0 * np.exp(drift + diffusion)
    
        return S_T
    except Exception as e:
        logger.error(f"خطا در شبیه‌سازی قیمت: {e}")
        return None
```

**توضیحات:**

**روش‌های توزیع‌شده برای شبیه‌سازی:**

در شبیه‌سازی‌های مالی، انتخاب توزیع مناسب برای متغیرهای تصادفی بسیار مهم است. در این بخش، چند روش رایج برای شبیه‌سازی متغیرهای تصادفی با ویژگی‌های خاص معرفی شده‌اند:

1. **GLD (Generalized Lambda Distribution):** این توزیع بسیار انعطاف‌پذیر است و می‌تواند طیف وسیعی از توزیع‌های احتمالی را مدل کند. با این حال، پیاده‌سازی آن پیچیده است و به همین دلیل، در برخی موارد از توزیع نرمال به عنوان جایگزینی برای آن استفاده می‌شود.
2. **SkewNormal (توزیع نرمال چول):** این توزیع، یک تعمیم از توزیع نرمال است که امکان مدل‌سازی چولگی را فراهم می‌کند. چولگی به معنای عدم تقارن توزیع است.
3. **CornishFisher:** این روش، یک توسعه بر اساس سری تیلور است که به شما اجازه می‌دهد تا توزیع‌های با چولگی و کشیدگی دلخواه را ایجاد کنید. کشیدگی به میزان پهن بودن یا باریک بودن دم‌های توزیع اشاره دارد.

**فرمول حرکت براونی هندسی:**

این فرمول یکی از مهم‌ترین فرمول‌ها در مدل‌سازی قیمت دارایی‌ها است و به طور گسترده در مدل‌سازی قیمت سهام استفاده می‌شود. این فرمول بیان می‌کند که قیمت یک دارایی در آینده (S_T) برابر است با قیمت فعلی آن (S_0) ضرب در یک عامل رشد نمایی که شامل نرخ بدون ریسک (r)، نوسان‌پذیری (σ) و یک متغیر تصادفی استاندارد نرمال (Z) است.

- **S_T:** قیمت دارایی در زمان T
- **S_0:** قیمت اولیه دارایی
- **r:** نرخ بدون ریسک
- **σ:** نوسان‌پذیری (انحراف استاندارد بازده‌های لگاریتمی)
- **T:** زمان
- **Z:** متغیر تصادفی استاندارد نرمال

**مدیریت خطا:**

در فرایند شبیه‌سازی، ممکن است خطاهایی رخ دهد. برای مدیریت این خطاها، معمولاً مقدار `None` بازگردانده می‌شود و خطا ثبت می‌گردد. این کار به شما اجازه می‌دهد تا به راحتی خطاها را شناسایی کرده و آنها را برطرف کنید.

**کاربردها:**

- **قیمت‌گذاری اختیار معامله:** مدل بلک-شولز از حرکت براونی هندسی برای قیمت‌گذاری اختیار معامله استفاده می‌کند.
- **شبیه‌سازی پرتفوی:** برای شبیه‌سازی بازده یک پرتفوی سرمایه‌گذاری از این فرمول استفاده می‌شود.
- **مدیریت ریسک:** برای ارزیابی ریسک مرتبط با سرمایه‌گذاری‌ها از این فرمول استفاده می‌شود.

**نکات مهم:**

- **انتخاب توزیع مناسب برای متغیر تصادفی Z در فرمول حرکت براونی هندسی، تأثیر زیادی بر نتایج شبیه‌سازی دارد.**
- **نوسان‌پذیری یک پارامتر کلیدی در مدل‌سازی قیمت دارایی‌ها است و بر میزان نوسانات قیمت تأثیر می‌گذارد.**
- **مدیریت خطا در شبیه‌سازی بسیار مهم است تا از نتایج نادرست جلوگیری شود.**

### 🔍 **متد `cornish_fisher_quantile`**

```python
def cornish_fisher_quantile(self, z: np.ndarray, skewness: float = 0.0, kurtosis_val: float = 3.0) -> np.ndarray:
    """
    تنظیم کوانتایل‌های نرمال استاندارد با استفاده از توسعه کرونیش-فیشر.
    
    Args:
        z (np.ndarray): کوانتایل‌های نرمال استاندارد.
        skewness (float): چولگی مورد نظر.
        kurtosis_val (float): کشیدگی مورد نظر.
    
    Returns:
        np.ndarray: کوانتایل‌های تنظیم‌شده.
    """
    excess_kurtosis = kurtosis_val - 3.0
    z_adj = (z +
             (z**2 - 1) * skewness / 6 +
             (z**3 - 3 * z) * excess_kurtosis / 24 -
             (2 * z**3 - 5 * z) * skewness**2 / 36)
    return z_adj
```

**توضیحات:**

**فرمول توزیع کرونیش-فیشر:**

فرمول:

z تعدیل‌شده = z + ((z² - 1) × چولگی) / 6 + ((z³ - 3z) × (کشیدگی - 3)) / 24 - ((2z³ - 5z) × چولگی²) / 36

**تفسیر:**

این فرمول یک روش برای تبدیل یک متغیر تصادفی با توزیع نرمال استاندارد به یک متغیر تصادفی با توزیع دلخواه با چولگی و کشیدگی مشخص است. به عبارت دیگر، این فرمول به ما اجازه می‌دهد تا یک متغیر تصادفی با توزیع نرمال را طوری تغییر دهیم که شکل آن به توزیع مورد نظر ما نزدیک‌تر شود.

- **z:** کوانتایل‌های توزیع نرمال استاندارد هستند. این مقادیر نشان می‌دهند که چه درصدی از داده‌ها کمتر از یک مقدار خاص هستند.
- **z تعدیل‌شده:** کوانتایل‌های توزیع تعدیل‌شده هستند که چولگی و کشیدگی مورد نظر را دارند.
- **کشیدگی (skewness):** معیاری برای اندازه‌گیری عدم تقارن یک توزیع است. اگر چولگی مثبت باشد، دم سمت راست توزیع بلندتر است و اگر چولگی منفی باشد، دم سمت چپ بلندتر است.
- **چولگی (kurtosis):** معیاری برای اندازه‌گیری تیزی قله و پهنای دم‌های یک توزیع است. اگر کشیدگی بزرگتر از 3 باشد، توزیع نسبت به توزیع نرمال نوک‌تیزتر و دم‌های بلندتری دارد.

**کاربردها:**

- **شبیه‌سازی داده‌ها:** برای ایجاد داده‌های شبیه‌سازی شده با توزیع‌های خاص، می‌توان از این فرمول استفاده کرد.
- **تبدیل متغیرهای تصادفی:** اگر یک متغیر تصادفی توزیع نرمال نداشته باشد، می‌توان با استفاده از این فرمول آن را به یک متغیر تصادفی با توزیع نزدیک به نرمال تبدیل کرد.
- **تعیین مقدار بحرانی در آزمون‌های آماری:** در برخی از آزمون‌های آماری، نیاز به تعیین مقدار بحرانی بر اساس توزیع‌های خاصی است که از این فرمول می‌توان برای محاسبه آن استفاده کرد.

**محدودیت‌ها:**

- **این فرمول یک تقریب است و ممکن است برای توزیع‌های با چولگی و کشیدگی بسیار زیاد دقت کافی نداشته باشد.**
- **این فرمول برای توزیع‌هایی که دارای چندین قله هستند یا بسیار پیچیده هستند، مناسب نیست.**

**به طور خلاصه:**

فرمول توسعه کرونیش-فیشر یک ابزار قدرتمند برای کار با توزیع‌های احتمالی است و به ما اجازه می‌دهد تا توزیع‌های پیچیده‌تری را بر اساس توزیع نرمال استاندارد مدل‌سازی کنیم.

### 🧮 **متد `monte_carlo_simulation_worker`**

```python
def monte_carlo_simulation_worker(self, option_data: Dict[str, Any], num_simulations: int = 10000) -> Dict[str, Any]:
    """
    تابع کارگر برای شبیه‌سازی مونت کارلو. برای استفاده در multiprocessing.
    
    Args:
        option_data (Dict[str, Any]): دیکشنری حاوی داده‌های اختیار معامله.
        num_simulations (int): تعداد شبیه‌سازی‌ها.
    
    Returns:
        Dict[str, Any]: نتایج شبیه‌سازی برای اختیار معامله.
    """
    try:
        # استخراج داده‌های لازم
        S0 = option_data['last_spot_price']
        K = option_data['strike_price']
        option_type = option_data['option_type'].upper()
        T = option_data['days'] / 365
        r = self.risk_free_rate
        sigma = option_data['Volatility']
        premium_long = option_data['ask_price']
        premium_short = option_data['bid_price']
        contract_size = option_data['contract_size']
        option_name = option_data['option_name']
        ua_tse_code = option_data['ua_tse_code']

        # تعیین چولگی و کشیدگی بر اساس سناریوی بازار و نوع اختیار معامله
        market_view = self.market_views.get(option_name, 'neutral')
        skewness, kurtosis_val = self.get_skew_kurtosis(option_type, market_view)

        # شبیه‌سازی قیمت دارایی
        Z = self.simulate_price(S0, K, r, sigma, T, num_simulations,
                                distribution_type=self.distribution_type,
                                skewness=skewness, kurtosis_val=kurtosis_val)
        if Z is None or len(Z) == 0:
            return {'option_name': option_name, 'pl_long': np.nan, 'pl_short': np.nan, 'S_T': np.nan}

        # محاسبه پرداخت‌ها
        if option_type == 'CALL':
            payoff = np.maximum(Z - K, 0)
        elif option_type == 'PUT':
            payoff = np.maximum(K - Z, 0)
        else:
            payoff = np.nan
            logger.warning(f"نوع اختیار معامله {option_type} برای {option_name} پشتیبانی نمی‌شود.")

        # محاسبه سود/ضرر
        pl_long = (payoff - premium_long) * contract_size
        pl_short = (premium_short - payoff) * contract_size

        return {'option_name': option_name, 'pl_long': pl_long, 'pl_short': pl_short, 'S_T': Z}
    except Exception as e:
        logger.error(f"خطا در شبیه‌سازی برای {option_data.get('option_name', 'ناشناس')}: {e}")
        return {'option_name': option_data.get('option_name', 'ناشناس'), 'pl_long': None, 'pl_short': None, 'S_T': None}
```

**توضیحات:**

- **استخراج داده‌های ضروری**: شامل قیمت دارایی پایه، قیمت اجرای اختیار معامله، نوع اختیار معامله، زمان تا سررسید، نرخ بهره بدون ریسک، نوسان، قیمت‌های خرید و فروش، اندازه قرارداد و نام اختیار معامله.
  
- **تعیین چولگی و کشیدگی**: بر اساس سناریوی بازار و نوع اختیار معامله، چولگی و کشیدگی مناسب تعیین می‌شود.

**محاسبه پرداخت (Payoff):**

پرداخت در اختیار معامله، مبلغی است که در زمان سررسید اختیار معامله به دارنده آن پرداخت می‌شود. این مقدار به نوع اختیار معامله (خرید یا فروش) و رابطه بین قیمت دارایی پایه در زمان سررسید (S_T) و قیمت اجرای اختیار معامله (K) بستگی دارد.

**اختیار معامله خرید (Call):**

پرداخت = حداکثر(S_T - K, 0)

به این معنی که اگر قیمت دارایی پایه در زمان سررسید (S_T) بیشتر از قیمت اجرا (K) باشد، دارنده اختیار می‌تواند دارایی را به قیمت پایین‌تر (K) بخرد و سپس با قیمت بازار (S_T) بفروشد و سود کند. در غیر این صورت، اختیار معامله بی‌ارزش می‌شود.

**اختیار معامله فروش (Put):**

پرداخت = حداکثر(K - S_T, 0)

به این معنی که اگر قیمت دارایی پایه در زمان سررسید (S_T) کمتر از قیمت اجرا (K) باشد، دارنده اختیار می‌تواند دارایی را با قیمت بازار (S_T) بخرد و سپس به قیمت بالاتر (K) بفروشد و سود کند. در غیر این صورت، اختیار معامله بی‌ارزش می‌شود.

**محاسبه سود/ضرر (Profit/Loss):**

- **موقعیت Long:**

  سود/ضرر = (پرداخت - پرمیوم خرید) × اندازه قرارداد

  این فرمول سود یا ضرر کلی یک موقعیت خرید اختیار معامله را نشان می‌دهد. پرداخت همانطور که در بالا توضیح داده شد، پرداخت نهایی است و پرمیوم خرید (Premium_Long) پرمیومی است که در ابتدا برای خرید اختیار معامله پرداخت شده است. اندازه قرارداد به تعداد قراردادهای خریداری شده اشاره دارد.

- **موقعیت Short:**

  سود/ضرر = (پرمیوم فروش - پرداخت) × اندازه قرارداد

  این فرمول سود یا ضرر کلی یک موقعیت فروش اختیار معامله را نشان می‌دهد. پرمیوم فروش (Premium_Short) پرمیومی است که در ابتدا از فروش اختیار معامله دریافت شده است.

**مدیریت خطا:**

در صورت بروز هرگونه خطا در محاسبات، به جای مقدار صحیح، مقدار `NaN` (Not a Number) بازگردانده می‌شود و خطا ثبت می‌گردد. این کار به شما کمک می‌کند تا خطاها را شناسایی و برطرف کنید.

**کاربردهای این فرمول‌ها:**

- **ارزیابی سود و زیان:** با استفاده از این فرمول‌ها می‌توانید سود یا ضرر بالقوه یک موقعیت اختیار معامله را قبل از انجام معامله تخمین بزنید.
- **ساخت استراتژی‌های معاملاتی:** این فرمول‌ها پایه و اساس بسیاری از استراتژی‌های معاملاتی مبتنی بر اختیار معامله هستند.
- **مدلسازی قیمت‌گذاری اختیار معامله:** این فرمول‌ها در مدل‌های پیچیده‌تر قیمت‌گذاری اختیار معامله استفاده می‌شوند.

**نکات مهم:**

- **پرمیوم:** پرمیوم قیمتی است که برای خرید یا فروش یک اختیار معامله پرداخت می‌شود.
- **اندازه قرارداد:** اندازه قرارداد به تعداد واحدهای دارایی پایه که هر قرارداد اختیار معامله به آن اشاره دارد، گفته می‌شود.
- **عوامل موثر بر پرداخت و سود/ضرر:** علاوه بر قیمت دارایی پایه و قیمت اجرا، عواماتی مانند نوسان‌پذیری، نرخ بهره و زمان تا سررسید نیز بر پرداخت و سود/ضرر اختیار معامله تأثیرگذار هستند.

### 🏭 **متد `monte_carlo_simulation`**

```python
def monte_carlo_simulation(self, num_simulations: int = 10000):
    """
    اجرای شبیه‌سازی‌های مونت کارلو برای تمامی اختیار معامله‌ها.
    
    Args:
        num_simulations (int): تعداد شبیه‌سازی‌ها برای هر اختیار معامله.
    """
    logger.info("شروع شبیه‌سازی‌های مونت کارلو.")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(self.monte_carlo_simulation_worker, num_simulations=num_simulations)
    try:
        # تبدیل هر ردیف به دیکشنری برای ارسال به کارگر
        options_data = self.cleaned_data.to_dict(orient='records')
        results = pool.map(func, options_data)
    except Exception as e:
        logger.error(f"خطا در multiprocessing: {e}")
        pool.close()
        pool.join()
        raise
    pool.close()
    pool.join()

    for result in results:
        option_name = result['option_name']
        pl_long = result['pl_long']
        pl_short = result['pl_short']
        S_T = result['S_T']
        if isinstance(pl_long, np.ndarray) and isinstance(pl_short, np.ndarray) and isinstance(S_T, np.ndarray):
            self.simulation_results[option_name] = {'long': pl_long, 'short': pl_short}
            self.target_date_distribution[option_name] = S_T
            self.cash_flows[option_name] = {
                'long': {
                    'initial': -self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'ask_price'].values[0] *
                               self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'contract_size'].values[0],
                    'final': pl_long
                },
                'short': {
                    'initial': self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'bid_price'].values[0] *
                               self.cleaned_data.loc[self.cleaned_data['option_name'] == option_name, 'contract_size'].values[0],
                    'final': pl_short
                }
            }
        else:
            self.simulation_results[option_name] = {'long': np.nan, 'short': np.nan}
            self.target_date_distribution[option_name] = np.nan
            self.cash_flows[option_name] = {
                'long': {'initial': np.nan, 'final': np.nan},
                'short': {'initial': np.nan, 'final': np.nan}
            }
    logger.info("شبیه‌سازی‌های مونت کارلو تکمیل شد.")
```

**توضیحات:**

- **اجرای موازی شبیه‌سازی‌ها:** با استفاده از `multiprocessing.Pool`, شبیه‌سازی‌ها به‌صورت موازی برای تمامی اختیار معامله‌ها اجرا می‌شوند.
- **جمع‌آوری نتایج:** نتایج شبیه‌سازی‌ها در متغیرهای داخلی ذخیره می‌شوند.
- **مدیریت خطا:** در صورت بروز خطا در multiprocessing, پردازش‌ها به‌درستی خاتمه یافته و خطا ثبت می‌شود.

---

### 📊 **متد `calculate_pop`**

```python
def calculate_pop(self):
    """
    محاسبه احتمال سود (Probability of Profit) برای هر اختیار معامله و موقعیت.
    """
    logger.info("محاسبه احتمال سود (PoP) برای هر اختیار معامله و موقعیت.")
    for option, results in self.simulation_results.items():
        pl_long = results.get('long', np.nan)
        pl_short = results.get('short', np.nan)
        pop_long = (np.sum(pl_long > 0) / len(pl_long)) * 100 if self.is_valid_array(pl_long) else np.nan
        pop_short = (np.sum(pl_short > 0) / len(pl_short)) * 100 if self.is_valid_array(pl_short) else np.nan
        self.pop_results[option] = {'long': pop_long, 'short': pop_short}
        logger.debug(f"PoP برای {option} - Long: {pop_long:.2f}%, Short: {pop_short:.2f}%")
```

**توضیحات:**

**احتمال سود (PoP):**

احتمال سود (Probability of Profit یا PoP) شاخصی است که در تحلیل‌های شبیه‌سازی معاملاتی به کار می‌رود و نشان می‌دهد که یک معامله یا استراتژی معاملاتی با چه احتمالی به سود خواهد رسید. این شاخص به معامله‌گران کمک می‌کند تا قبل از اجرای یک استراتژی در بازار واقعی، از احتمال موفقیت آن آگاه شوند و تصمیم‌گیری بهتری داشته باشند.

**محاسبه PoP:**

برای محاسبه PoP، تعداد دفعاتی که یک معامله در شبیه‌سازی‌ها به سود رسیده است را بر تعداد کل شبیه‌سازی‌ها تقسیم کرده و سپس نتیجه را در 100 ضرب می‌کنیم تا به درصد تبدیل شود. به‌طور رسمی‌تر، فرمول محاسبه PoP به صورت زیر است:

PoP = (تعداد شبیه‌سازی‌های سودآور / تعداد کل شبیه‌سازی‌ها) × 100%

**کاربردهای PoP:**

- **ارزیابی استراتژی‌ها:** با استفاده از PoP می‌توان استراتژی‌های مختلف معاملاتی را با هم مقایسه کرده و استراتژی‌هایی که بیشترین احتمال موفقیت را دارند، انتخاب کرد.
- **مدیریت ریسک:** PoP به معامله‌گران کمک می‌کند تا ریسک مرتبط با هر استراتژی را بهتر درک کرده و تصمیمات آگاهانه‌تری در مورد مدیریت ریسک بگیرند.
- **بهینه‌سازی پارامترها:** می‌توان از PoP برای تنظیم و بهینه‌سازی پارامترهای یک مدل یا استراتژی استفاده کرد تا به بالاترین احتمال سود دست یافت.

### 📐 **متد `calculate_greeks`**

```python
def calculate_greeks(self):
    """
    محاسبه شاخص‌های یونانی (Greeks) برای هر اختیار معامله.
    """
    logger.info("محاسبه شاخص‌های یونانی (Greeks) برای هر اختیار معامله.")
    for idx, row in self.cleaned_data.iterrows():
        S0 = row['last_spot_price']
        K = row['strike_price']
        r = self.risk_free_rate
        sigma = row['Volatility']
        T = row['days'] / 365
        option_type = row['option_type'].upper()
        option_name = row['option_name']

        if S0 <= 0 یا K <= 0 یا sigma <= 0 یا T <= 0:
            self.greeks[option_name] = {key: np.nan for key in
                                        ['Delta_long', 'Gamma_long', 'Theta_long', 'Vega_long', 'Rho_long',
                                         'Delta_short', 'Gamma_short', 'Theta_short', 'Vega_short', 'Rho_short']}
            logger.warning(f"پارامترهای نامعتبر برای محاسبه شاخص‌های یونانی برای {option_name}.")
            continue

        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'CALL':
            delta_long = norm.cdf(d1)
            theta_long = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
                          r * K * np.exp(-r * T) * norm.cdf(d2))
            rho_long = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'PUT':
            delta_long = norm.cdf(d1) - 1
            theta_long = (- (S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
                          r * K * np.exp(-r * T) * norm.cdf(-d2))
            rho_long = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            delta_long = theta_long = rho_long = np.nan
            logger.warning(f"نوع اختیار معامله {option_type} برای {option_name} پشتیبانی نمی‌شود.")

        gamma_long = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        vega_long = S0 * norm.pdf(d1) * np.sqrt(T)

        self.greeks[option_name] = {
            'Delta_long': delta_long,
            'Gamma_long': gamma_long,
            'Theta_long': theta_long,
            'Vega_long': vega_long,
            'Rho_long': rho_long,
            'Delta_short': -delta_long,
            'Gamma_short': -gamma_long,
            'Theta_short': -theta_long,
            'Vega_short': -vega_long,
            'Rho_short': -rho_long
        }
        logger.debug(f"شاخص‌های یونانی برای {option_name}: {self.greeks[option_name]}")
```

**توضیحات:**

- **تعریف شاخص‌های یونانی:**

  - **Delta (Δ):** حساسیت قیمت اختیار معامله نسبت به تغییرات قیمت دارایی پایه.
  
  - **Gamma (Γ):** حساسیت Delta نسبت به تغییرات قیمت دارایی پایه.
  
  - **Theta (Θ):** حساسیت قیمت اختیار معامله نسبت به تغییرات زمان باقی‌مانده تا سررسید.
  
  - **Vega (V):** حساسیت قیمت اختیار معامله نسبت به تغییرات نوسان.
  
  - **Rho (ρ):** حساسیت قیمت اختیار معامله نسبت به تغییرات نرخ بهره بدون ریسک.

- **فرمول‌های محاسبه:**

  - **اختیار معامله خرید (CALL):**

    - Δ_L = N(d1)
    - Θ_L = - (S0 * N'(d1) * σ) / (2 * √T) - r * K * e^{-rT} * N(d2)
    - ρ_L = K * T * e^{-rT} * N(d2)

  - **اختیار معامله فروش (PUT):**

    - Δ_L = N(d1) - 1
    - Θ_L = - (S0 * N'(d1) * σ) / (2 * √T) + r * K * e^{-rT} * N(-d2)
    - ρ_L = -K * T * e^{-rT} * N(-d2)

  - **Gamma و Vega:**

    - Γ_L = N'(d1) / (S0 * σ * √T)
    - V_L = S0 * N'(d1) * √T

**مدیریت خطا:**

در صورت داشتن پارامترهای نامعتبر یا نوع اختیار معامله پشتیبانی‌نشده، مقادیر `NaN` تنظیم شده و هشدار ثبت می‌شود.

### 📍 **متد `calculate_breakeven`**

```python
def calculate_breakeven(self):
    """
    محاسبه نقاط سر به سر برای هر اختیار معامله و موقعیت.
    """
    logger.info("محاسبه نقاط سر به سر برای هر اختیار معامله و موقعیت.")
    for idx, row in self.cleaned_data.iterrows():
        option = row['option_name']
        option_type = row['option_type'].upper()
        S0 = row['last_spot_price']
        K = row['strike_price']
        premium_long = row['ask_price']
        premium_short = row['bid_price']
        r = self.risk_free_rate
        T = row['days'] / 365

        if option_type == 'CALL':
            breakeven_long = K + premium_long
            breakeven_short = K + premium_short
        elif option_type == 'PUT':
            breakeven_long = K - premium_long
            breakeven_short = K - premium_short
        else:
            breakeven_long = breakeven_short = np.nan
            logger.warning(f"نوع اختیار معامله {option_type} برای محاسبه نقاط سر به سر پشتیبانی نمی‌شود.")

        adjusted_breakeven_long = breakeven_long * np.exp(r * T) if not np.isnan(breakeven_long) else np.nan
        adjusted_breakeven_short = breakeven_short * np.exp(r * T) if not np.isnan(breakeven_short) else np.nan

        breakeven_long_pct = ((adjusted_breakeven_long - S0) / S0) * 100 if not np.isnan(adjusted_breakeven_long) else np.nan
        breakeven_short_pct = ((adjusted_breakeven_short - S0) / S0) * 100 if not np.isnan(adjusted_breakeven_short) else np.nan

        self.breakeven_points[option] = {
            'long': adjusted_breakeven_long,
            'short': adjusted_breakeven_short,
            'long_pct': breakeven_long_pct,
            'short_pct': breakeven_short_pct
        }
        logger.debug(f"نقاط سر به سر برای {option}: {self.breakeven_points[option]}")
```

**توضیحات:**

- **تعریف نقاط سر به سر:**

  - **اختیار معامله خرید (CALL):**

    نقاط سر به سر برای موقعیت Long:
    
    نقاط سر به سر = قیمت اجرا + پرمیوم خرید

    نقاط سر به سر برای موقعیت Short:
    
    نقاط سر به سر = قیمت اجرا + پرمیوم فروش

  - **اختیار معامله فروش (PUT):**

    نقاط سر به سر برای موقعیت Long:
    
    نقاط سر به سر = قیمت اجرا - پرمیوم خرید

    نقاط سر به سر برای موقعیت Short:
    
    نقاط سر به سر = قیمت اجرا - پرمیوم فروش

- **تنظیم نقاط سر به سر بر اساس نرخ بهره:**

  نقاط سر به سر تعدیل‌شده = نقاط سر به سر × e^{rT}

- **محاسبه درصد نقاط سر به سر نسبت به قیمت دارایی پایه:**

  درصد نقاط سر به سر = ((نقاط سر به سر تعدیل‌شده - قیمت دارایی پایه) / قیمت دارایی پایه) × 100%

- **مدیریت خطا:**

  در صورت داشتن نوع اختیار معامله پشتیبانی‌نشده، مقادیر `NaN` تنظیم شده و هشدار ثبت می‌شود.

### 📈 **متد `calculate_sharpe_ratio`**

```python
def calculate_sharpe_ratio(self, risk_free_rate: float = 0.01):
    """
    محاسبه نسبت شارپ برای هر اختیار معامله و موقعیت.
    
    Args:
        risk_free_rate (float): نرخ بهره بدون ریسک برای محاسبه نسبت شارپ.
    """
    logger.info("محاسبه نسبت شارپ برای هر اختیار معامله و موقعیت.")
    for option, positions in self.simulation_results.items():
        pl_long = positions.get('long', np.nan)
        pl_short = positions.get('short', np.nan)

        option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
        if option_row.empty:
            logger.warning(f"داده‌ای برای اختیار معامله {option} در محاسبه نسبت شارپ یافت نشد.")
            self.sharpe_ratios[option] = {'long': np.nan, 'short': np.nan}
            continue

        option_row = option_row.iloc[0]
        days_to_maturity = option_row['days']

        if self.is_valid_array(pl_long):
            initial_investment_long = option_row['ask_price'] * option_row['contract_size']
            if initial_investment_long > 0:
                returns_long = pl_long / initial_investment_long
                mean_return_long = np.mean(returns_long) * (365 / days_to_maturity)
                std_return_long = np.std(returns_long) * np.sqrt(365 / days_to_maturity)
                sharpe_long = (mean_return_long - risk_free_rate) / std_return_long if std_return_long > 0 else np.nan
            else:
                sharpe_long = np.nan
        else:
            sharpe_long = np.nan

        if self.is_valid_array(pl_short):
            initial_premium_short = option_row['bid_price'] * option_row['contract_size']
            if initial_premium_short > 0:
                returns_short = pl_short / initial_premium_short
                mean_return_short = np.mean(returns_short) * (365 / days_to_maturity)
                std_return_short = np.std(returns_short) * np.sqrt(365 / days_to_maturity)
                sharpe_short = (mean_return_short - risk_free_rate) / std_return_short if std_return_short > 0 else np.nan
            else:
                sharpe_short = np.nan
        else:
            sharpe_short = np.nan

        self.sharpe_ratios[option] = {'long': sharpe_long, 'short': sharpe_short}
        logger.debug(f"نسبت شارپ برای {option}: {self.sharpe_ratios[option]}")
```

**توضیحات:**

- **تعریف نسبت شارپ:**

  نسبت شارپ = (بازده مورد انتظار سرمایه‌گذاری - نرخ بهره بدون ریسک) / انحراف معیار بازده سرمایه‌گذاری

  که در آن:
  
  - **E[R_p]:** بازده مورد انتظار سرمایه‌گذاری.
  - **R_f:** نرخ بهره بدون ریسک.
  - **σ_p:** انحراف معیار بازده سرمایه‌گذاری.

- **محاسبه بازده و انحراف معیار:**

  برای **موقعیت Long** و **Short** به‌طور جداگانه محاسبه می‌شود.

- **مدیریت خطا:**

  در صورت نداشتن داده‌های لازم، مقادیر `NaN` تنظیم شده و هشدار ثبت می‌شود.

### 📉 **متد `calculate_var_cvar`**

```python
def calculate_var_cvar(self, confidence_level: float = 0.95):
    """
    محاسبه Value at Risk (VaR) و Conditional Value at Risk (CVaR) برای هر اختیار معامله و موقعیت.
    
    Args:
        confidence_level (float): سطح اطمینان برای محاسبه VaR و CVaR.
    """
    logger.info(f"محاسبه VaR و CVaR با سطح اطمینان {confidence_level*100:.0f}%.")
    for option, positions in self.simulation_results.items():
        pl_long = positions.get('long', np.nan)
        pl_short = positions.get('short', np.nan)

        if self.is_valid_array(pl_long):
            var_long = np.percentile(pl_long, (1 - confidence_level) * 100)
            cvar_long = pl_long[pl_long <= var_long].mean()
        else:
            var_long = cvar_long = np.nan

        if self.is_valid_array(pl_short):
            var_short = np.percentile(pl_short, (1 - confidence_level) * 100)
            cvar_short = pl_short[pl_short <= var_short].mean()
        else:
            var_short = cvar_short = np.nan

        self.var[option] = {'long': var_long, 'short': var_short}
        self.cvar[option] = {'long': cvar_long, 'short': cvar_short}
        logger.debug(f"VaR و CVaR برای {option}: VaR_Long={var_long}, CVaR_Long={cvar_long}, VaR_Short={var_short}, CVaR_Short={cvar_short}")
```

**توضیحات:**

- **تعریف VaR و CVaR:**

  - **Value at Risk (VaR):** حد ضرر محتمل در سطح اطمینان مشخص.
    
    VaR = Percentile((1 - α) × 100)%

    که در آن α سطح اطمینان (مثلاً ۹۵٪) است.

  - **Conditional Value at Risk (CVaR):** میانگین ضررهای فراتر از VaR.
    
    CVaR = میانگین ضررهایی که بیشتر از VaR هستند.

- **محاسبه VaR و CVaR:**

  برای **موقعیت Long** و **Short** به‌طور جداگانه محاسبه می‌شود.

- **مدیریت خطا:**

  در صورت نداشتن داده‌های لازم، مقادیر `NaN` تنظیم می‌شود.

### 📈 **متد `calculate_variance`**

```python
def calculate_variance(self):
    """
    محاسبه واریانس سود برای هر اختیار معامله و موقعیت.
    """
    logger.info("محاسبه واریانس برای هر اختیار معامله و موقعیت.")
    for option, positions in self.simulation_results.items():
        pl_long = positions.get('long', np.nan)
        pl_short = positions.get('short', np.nan)

        option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
        if option_row.empty:
            logger.warning(f"داده‌ای برای اختیار معامله {option} در محاسبه واریانس یافت نشد.")
            self.variance[option] = {'long': np.nan, 'short': np.nan}
            continue

        option_row = option_row.iloc[0]
        days_to_maturity = option_row['days']

        if self.is_valid_array(pl_long):
            variance_long = np.var(pl_long) * (252 / days_to_maturity)
        else:
            variance_long = np.nan

        if self.is_valid_array(pl_short):
            variance_short = np.var(pl_short) * (252 / days_to_maturity)
        else:
            variance_short = np.nan

        self.variance[option] = {'long': variance_long, 'short': variance_short}
        logger.debug(f"واریانس برای {option}: واریانس_Long={variance_long}, واریانس_Short={variance_short}")
```

**توضیحات:**

- **تعریف واریانس:**

  واریانس = (252 / T) × واریانس سود

  که در آن T زمان باقی‌مانده تا سررسید به روز است.

- **مدیریت خطا:**

  در صورت نداشتن داده‌های لازم، مقادیر `NaN` تنظیم می‌شود.

### 💰 **متد `calculate_payout_ratio`**

```python
def calculate_payout_ratio(self):
    """
    محاسبه نسبت‌های پرداخت و کارایی پرمیوم برای هر اختیار معامله و موقعیت.
    """
    logger.info("محاسبه نسبت‌های پرداخت و کارایی پرمیوم برای هر اختیار معامله و موقعیت.")
    for option, cash_flows in self.cash_flows.items():
        option_row = self.cleaned_data[self.cleaned_data['option_name'] == option]
        if option_row.empty:
            logger.warning(f"داده‌ای برای اختیار معامله {option} در محاسبه نسبت پرداخت یافت نشد.")
            self.payout_ratios[option] = {
                'long': np.nan,
                'short': np.nan,
                'premium_efficiency_long': np.nan,
                'premium_efficiency_short': np.nan
            }
            continue

        option_row = option_row.iloc[0]
        contract_size = option_row['contract_size']
        premium_long = option_row['ask_price']
        premium_short = option_row['bid_price']
        pl_long = cash_flows['long']['final']
        pl_short = cash_flows['short']['final']

        if self.is_valid_array(pl_long):
            average_pl_long = np.mean(pl_long)
            initial_investment_long = premium_long * contract_size
            payout_long = average_pl_long / initial_investment_long if initial_investment_long > 0 else np.nan
            premium_efficiency_long = 1 / initial_investment_long if initial_investment_long > 0 else 0
        else:
            payout_long = premium_efficiency_long = np.nan

        if self.is_valid_array(pl_short):
            average_pl_short = np.mean(pl_short)
            initial_premium_short = premium_short * contract_size
            payout_short = average_pl_short / initial_premium_short if initial_premium_short > 0 else np.nan
            premium_efficiency_short = initial_premium_short
        else:
            payout_short = premium_efficiency_short = np.nan

        self.payout_ratios[option] = {
            'long': payout_long,
            'short': payout_short,
            'premium_efficiency_long': premium_efficiency_long,
            'premium_efficiency_short': premium_efficiency_short
        }
        logger.debug(f"نسبت‌های پرداخت برای {option}: {self.payout_ratios[option]}")
```

**توضیحات:**

- **تعریف نسبت‌های پرداخت:**

  - **موقعیت Long:**

    نسبت پرداخت = میانگین سود/ضرر Long / سرمایه اولیه Long

    کارایی پرمیوم Long = 1 / سرمایه اولیه Long

  - **موقعیت Short:**

    نسبت پرداخت = میانگین سود/ضرر Short / پرمیوم اولیه Short

    کارایی پرمیوم Short = پرمیوم اولیه Short

- **مدیریت خطا:**

  در صورت نداشتن داده‌های لازم، مقادیر `NaN` تنظیم می‌شود.

### 🧭 **متد `determine_market_views`**

```python
def determine_market_views(self):
    """
    تعیین دیدگاه بازار بر اساس مونی‌نِس هر اختیار معامله.
    """
    logger.info("تعیین دیدگاه بازار بر اساس مونی‌نِس.")
    for _, row in self.cleaned_data.iterrows():
        option = row['option_name']
        option_type = row['option_type'].upper()
        S = row['last_spot_price']
        K = row['strike_price']
        if option_type == 'CALL':
            self.market_views[option] = 'صعودی' if S > K else 'نزولی'
        elif option_type == 'PUT':
            self.market_views[option] = 'نزولی' if S < K else 'صعودی'
        else:
            self.market_views[option] = 'بی‌طرف'
            logger.warning(f"نوع اختیار معامله {option_type} برای تعیین دیدگاه بازار پشتیبانی نمی‌شود.")
```

**توضیحات:**

- **تعریف دیدگاه بازار:**

  - **اختیار معامله خرید (CALL):**

    دیدگاه بازار = صعودی اگر قیمت دارایی پایه > قیمت اجرا، در غیر این صورت نزولی

  - **اختیار معامله فروش (PUT):**

    دیدگاه بازار = نزولی اگر قیمت دارایی پایه < قیمت اجرا، در غیر این صورت صعودی

  که در آن \( S \) قیمت دارایی پایه و \( K \) قیمت اجرای اختیار معامله است.

- **مدیریت خطا:**

  در صورت داشتن نوع اختیار معامله پشتیبانی‌نشده، دیدگاه بازار به‌صورت `بی‌طرف` تنظیم شده و هشدار ثبت می‌شود.

### 📏 **متد `calculate_metrics_min_max`**

```python
def calculate_metrics_min_max(self):
    """
    محاسبه حداقل و حداکثر مقادیر شاخص‌های مختلف برای هر موقعیت جهت استانداردسازی.
    """
    metrics = ['sharpe_ratios', 'pop_results', 'var', 'cvar', 'payout_ratios', 'breakeven_pct', 'premium_efficiency']
    positions = ['long', 'short']
    for metric in metrics:
        self.metrics_min[metric] = {}
        self.metrics_max[metric] = {}
        for position in positions:
            if metric == 'sharpe_ratios':
                values = [self.sharpe_ratios.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'pop_results':
                values = [self.pop_results.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'payout_ratios':
                values = [self.payout_ratios.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'premium_efficiency':
                key = f'premium_efficiency_{position}'
                values = [self.payout_ratios.get(opt, {}).get(key, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'breakeven_pct':
                key = f'breakeven_{position}_pct'
                values = [self.breakeven_points.get(opt, {}).get(key, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'var':
                values = [self.var.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
            elif metric == 'cvar':
                values = [self.cvar.get(opt, {}).get(position, np.nan) for opt in self.cleaned_data['option_name']]
            else:
                values = []

            values = np.array(values)
            values = values[~np.isnan(values)]
            if len(values) > 0:
                self.metrics_min[metric][position] = np.min(values)
                self.metrics_max[metric][position] = np.max(values)
                logger.debug(f"شاخص '{metric}' برای موقعیت '{position}': min={self.metrics_min[metric][position]}, max={self.metrics_max[metric][position]}")
            else:
                self.metrics_min[metric][position] = 0
                self.metrics_max[metric][position] = 1
                logger.debug(f"شاخص '{metric}' برای موقعیت '{position}' مقادیر معتبر ندارد. تنظیم min=0 و max=1.")
```

**توضیحات:**

- **محاسبه حداقل و حداکثر مقادیر:**

  برای هر شاخص و موقعیت، حداقل و حداکثر مقادیر از داده‌های شبیه‌سازی‌شده استخراج می‌شوند تا در فرآیند استانداردسازی استفاده شوند.

- **مدیریت داده‌های ناقص:**

  در صورت نداشتن داده‌های معتبر برای یک شاخص و موقعیت، حداقل و حداکثر به‌صورت پیش‌فرض برابر با ۰ و ۱ تنظیم می‌شوند.

### 📏 **متد `standardize_metric`**

```python
def standardize_metric(self, value: float, metric_name: str, position: str) -> float:
    """
    استانداردسازی مقدار یک شاخص بر اساس حداقل و حداکثر آن.
    
    Args:
        value (float): مقدار شاخص برای استانداردسازی.
        metric_name (str): نام شاخص.
        position (str): 'long' یا 'short'.
    
    Returns:
        float: مقدار استاندارد شده شاخص.
    """
    if metric_name in ['sharpe_ratios', 'pop_results', 'payout_ratios', 'premium_efficiency']:
        min_value = self.metrics_min[metric_name][position]
        max_value = self.metrics_max[metric_name][position]
        standardized = (value - min_value) / (max_value - min_value) if max_value != min_value else 0
    elif metric_name in ['var', 'cvar', 'breakeven_pct']:
        min_value = self.metrics_min[metric_name][position]
        max_value = self.metrics_max[metric_name][position]
        standardized = (max_value - value) / (max_value - min_value) if max_value != min_value else 0
    else:
        standardized = 0
        logger.warning(f"شناخته نشدن شاخص '{metric_name}' در هنگام استانداردسازی.")
    return standardized
```

**توضیحات:**

- **استانداردسازی شاخص‌ها:**

  - **شاخص‌های مثبت** (`sharpe_ratios`, `pop_results`, `payout_ratios`, `premium_efficiency`):

    استاندارد شده = (مقدار - حداقل) / (حداکثر - حداقل)

  - **شاخص‌های منفی** (`var`, `cvar`, `breakeven_pct`):

    استاندارد شده = (حداکثر - مقدار) / (حداکثر - حداقل)

- **مدیریت خطا:**

  در صورت شناسایی نشدن شاخص، مقدار استاندارد شده به‌صورت ۰ تنظیم شده و هشدار ثبت می‌شود.

### 📊 **متد `calculate_composite_score`**

```python
def calculate_composite_score(self, option: str, position: str) -> float:
    """
    محاسبه امتیاز ترکیبی برای موقعیت یک اختیار معامله بر اساس شاخص‌های مختلف.
    
    Args:
        option (str): نام اختیار معامله.
        position (str): 'long' یا 'short'.
    
    Returns:
        float: امتیاز ترکیبی.
    """
    metrics = ['sharpe_ratios', 'pop_results', 'var', 'cvar', 'payout_ratios', 'breakeven_pct', 'premium_efficiency']
    base_weights = {
        'sharpe_ratios': 0.10,
        'pop_results': 0.15,
        'var': 0.10,
        'cvar': 0.10,
        'payout_ratios': 0.15,
        'breakeven_pct': 0.25,
        'premium_efficiency': 0.15
    }
    adjusted_weights = base_weights.copy()
    if self.strategy == 'aggressive':
        adjusted_weights['payout_ratios'] += 0.05
        adjusted_weights['var'] -= 0.05
    elif self.strategy == 'conservative':
        adjusted_weights['var'] += 0.05
        adjusted_weights['payout_ratios'] -= 0.05

    # نرمال‌سازی وزن‌ها
    total_weight = sum(adjusted_weights.values())
    adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

    score = 0
    for key in metrics:
        if key == 'premium_efficiency':
            value = self.payout_ratios.get(option, {}).get(f'premium_efficiency_{position}', np.nan)
        elif key == 'breakeven_pct':
            value = self.breakeven_points.get(option, {}).get(f'breakeven_{position}_pct', np.nan)
        else:
            metric_dict = getattr(self, key, {})
            value = metric_dict.get(option, {}).get(position, np.nan)

        if not np.isnan(value):
            standardized = self.standardize_metric(value, key, position)
            score += adjusted_weights[key] * standardized
            logger.debug(f"شاخص '{key}' برای {option} موقعیت {position}: مقدار={value}, استاندارد شده={standardized}, وزن={adjusted_weights[key]}, سهم امتیاز={adjusted_weights[key] * standardized}")
        else:
            logger.debug(f"شاخص '{key}' برای {option} موقعیت {position} نامعتبر است و در محاسبه امتیاز ترکیبی نادیده گرفته می‌شود.")

    logger.debug(f"امتیاز ترکیبی برای {option} موقعیت {position}: {score}")
    return score
```

**توضیحات:**

- **تعریف وزن‌های پایه و تنظیم‌شده:**

  وزن‌های اولیه شاخص‌ها بر اساس استراتژی (خشن یا محافظه‌کار) تنظیم می‌شوند.

  - **استراتژی خشن (Aggressive):**

    - افزایش وزن `payout_ratios` به میزان 0.05
    - کاهش وزن `var` به میزان 0.05

  - **استراتژی محافظه‌کار (Conservative):**

    - افزایش وزن `var` به میزان 0.05
    - کاهش وزن `payout_ratios` به میزان 0.05

- **نرمال‌سازی وزن‌ها:**

  مجموع وزن‌ها برابر با ۱ تنظیم می‌شود.

- **محاسبه امتیاز ترکیبی:**

  هر شاخص استانداردسازی شده با وزن مربوطه ضرب شده و به امتیاز کل اضافه می‌شود.

- **مدیریت خطا:**

  در صورت نامعتبر بودن یک شاخص، آن شاخص در محاسبه امتیاز ترکیبی نادیده گرفته می‌شود.

### 📑 **متد `compile_metrics_data`**

```python
def compile_metrics_data(self) -> pd.DataFrame:
    """
    جمع‌آوری تمامی شاخص‌های محاسبه‌شده در یک DataFrame ساختاریافته.
    
    Returns:
        pd.DataFrame: DataFrame حاوی تمامی شاخص‌ها برای هر اختیار معامله.
    """
    try:
        data = []
        for option in self.cleaned_data['option_name']:
            metrics = {
                'OptionName': option,
                'Sharpe_Long': self.sharpe_ratios.get(option, {}).get('long', np.nan),
                'Sharpe_Short': self.sharpe_ratios.get(option, {}).get('short', np.nan),
                'PoP_Long': self.pop_results.get(option, {}).get('long', np.nan),
                'PoP_Short': self.pop_results.get(option, {}).get('short', np.nan),
                'VaR_Long': self.var.get(option, {}).get('long', np.nan),
                'VaR_Short': self.var.get(option, {}).get('short', np.nan),
                'CVaR_Long': self.cvar.get(option, {}).get('long', np.nan),
                'CVaR_Short': self.cvar.get(option, {}).get('short', np.nan),
                'Breakeven_Long_Pct': self.breakeven_points.get(option, {}).get('long_pct', np.nan),
                'Breakeven_Short_Pct': self.breakeven_points.get(option, {}).get('short_pct', np.nan),
                'Premium_Efficiency_Long': self.payout_ratios.get(option, {}).get('premium_efficiency_long', np.nan),
                'Premium_Efficiency_Short': self.payout_ratios.get(option, {}).get('premium_efficiency_short', np.nan),
                'Variance_Long': self.variance.get(option, {}).get('long', np.nan),
                'Variance_Short': self.variance.get(option, {}).get('short', np.nan),
                # افزودن شاخص‌های بیشتر در صورت نیاز
            }
            data.append(metrics)
        metrics_df = pd.DataFrame(data)
        logger.info("جمع‌آوری داده‌های شاخص‌ها در DataFrame تکمیل شد.")
        return metrics_df
    except Exception as e:
        logger.error(f"خطا در جمع‌آوری داده‌های شاخص‌ها: {e}")
        return pd.DataFrame()
```

**توضیحات:**

- **جمع‌آوری شاخص‌ها:**

  تمامی شاخص‌های محاسبه‌شده برای هر اختیار معامله در یک دیکشنری جمع‌آوری و سپس به یک DataFrame تبدیل می‌شوند.

- **مدیریت خطا:**

  در صورت بروز خطا، خطا ثبت شده و یک DataFrame خالی بازگردانده می‌شود.

---

## 🔍 **نتیجه‌گیری**

با استفاده از این مدول، شما قادر به تحلیل دقیق **اختیار معامله‌ها** با بهره‌گیری از روش‌های شبیه‌سازی پیشرفته و محاسبه شاخص‌های مالی مختلف خواهید بود. این ابزار امکانات زیر را فراهم می‌کند:

1. **بارگذاری و پردازش داده‌ها:** با استفاده از کلاس `DataLoader`, داده‌های اولیه به‌صورت دقیق بارگذاری و پردازش می‌شوند.
2. **شبیه‌سازی مونت کارلو:** با استفاده از توزیع‌های مختلف و تنظیمات چولگی و کشیدگی، شبیه‌سازی‌های متنوعی از قیمت‌های آینده دارایی‌ها انجام می‌شود.
3. **محاسبه شاخص‌های مالی:** شاخص‌هایی مانند نسبت شارپ، PoP، VaR، CVaR، و شاخص‌های یونانی برای ارزیابی ریسک و بازده اختیار معامله‌ها محاسبه می‌شوند.
4. **تحلیل‌های تعاملی:** با استفاده از تحلیل‌های سناریویی و حساسیت، تأثیر تغییرات مختلف در شرایط بازار و پارامترهای مالی بر عملکرد اختیار معامله‌ها بررسی می‌شود.
5. **مدیریت کارآمد فرآیند:** با بهره‌گیری از پردازش موازی (`multiprocessing`), محاسبات پیچیده به‌صورت کارآمد و سریع انجام می‌شوند تا زمان اجرای کد کاهش یابد.
6. **ثبت دقیق لاگ‌ها:** با استفاده از کتابخانه‌ی `logging`, تمامی مراحل و خطاها به‌صورت دقیق ثبت شده و امکان پیگیری و رفع مشکلات فراهم شده است.


---

**پایان**

این مدول، فراتر از یک ابزار تحلیلی، تلاشی است برای ارائه‌ی دیدگاهی نوین در مواجهه با پیچیدگی‌های دنیای اختیار معامله. هر خط از کد، هر فرمول محاسباتی، و هر شبیه‌سازی انجام‌شده، گواهی است بر اشتیاق بی‌پایان من برای ایجاد راه‌حلی که درک ریسک و بازده را شفاف‌تر کند. امید دارم این اثر، برای تحلیل‌گران مالی و علاقه‌مندان به بازار سرمایه، نه‌تنها به‌عنوان ابزاری مفید، بلکه به‌عنوان دعوتی برای جست‌وجوی خلاقانه‌تر در مرزهای دانش و فناوری، مورد استقبال قرار گیرد. آینده، میدان بازی ماست؛ بیایید آن را با ابزارهای بهتر و نگاهی ژرف‌تر فتح کنیم.
