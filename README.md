# Safe Driving with DQN in MetaDrive

پیاده‌سازی کامل تکلیف «رانندگی ایمن با Deep Q-Network در شبیه‌ساز MetaDrive».



## اعضای تیم
## حسام علیزاده 

## ساختار فایل‌ها

```
metadrive_dqn/
├── env_utils.py       # Task 2: flatten_obs, discrete_to_continuous_action
├── make_env.py        # Task 1: ساخت و تست محیط MetaDriveEnv
├── dqn_network.py      # Task 3: شبکه Q (DQNNetwork)
├── replay_buffer.py    # Task 4: بافر تجربه (ReplayBuffer)
├── dqn_agent.py         # Task 5: ایجنت DQN (epsilon-greedy + Bellman update)
├── train.py             # Task 6 & 7: حلقه آموزش
├── evaluate.py           # Task 8: رسم نمودار و ارزیابی + ویژوالایز
├── requirements.txt
├── models/               # مدل ذخیره‌شده اینجا قرار می‌گیرد -> dqn_trained.pt
├── logs/                 # لاگ آموزش (training_log.json)
└── plots/                # نمودارهای reward / length / crash rate
```

## ۱. نصب کتابخانه‌ها

توصیه می‌شود از یک محیط مجازی جدا استفاده کنید (Python 3.9 یا 3.10 پیشنهاد MetaDrive است):

```bash
python -m venv venv
source venv/bin/activate      # ویندوز: venv\Scripts\activate

pip install --upgrade pip
pip install metadrive-simulator
pip install torch --index-url https://download.pytorch.org/whl/cu121   # اگر GPU دارید
# یا نسخه CPU:
# pip install torch

pip install numpy matplotlib
```

نکات مهم نصب MetaDrive:
- روی لینوکس/ویندوز/مک کار می‌کند؛ برای رندر ۳بعدی به یک GPU یا حداقل درایور OpenGL نیاز دارد.
- اگر روی سرور بدون صفحه‌نمایش (headless, مثل Colab یا SSH) کار می‌کنید، `use_render=False` بگذارید (پیش‌فرض کد همین است) و فقط برای مرحله‌ی نهایی ویژوالایزیشن آن را روشن کنید.
- بعد از نصب، تست کنید:
```bash
python -c "import metadrive; print(metadrive.__version__)"
```

## ۲. تست اولیه محیط (Task 1)

```bash
python make_env.py
```
این اسکریپت محیط را reset و چند step تصادفی انجام می‌دهد تا مطمئن شوید محیط درست کار می‌کند.

## ۳. اجرای آموزش سریع برای دیباگ (Task 7 - مرحله اول)

```bash
python train.py --episodes 20 --learning_starts 500
```
این اجرا فقط برای اطمینان از صحت پیاده‌سازی است (سرعت مهم نیست، درستی مهم است).

## ۴. اجرای آموزش کامل (Task 7 - مرحله نهایی)

```bash
python train.py --episodes 800 --max_steps 1000 --learning_starts 2000
```
- خروجی مدل نهایی در `models/dqn_trained.pt` ذخیره می‌شود.
- لاگ کامل آموزش (reward, length, crash) در `logs/training_log.json` ذخیره می‌شود.
- هر ۵۰ اپیزود یک چک‌پوینت میانی هم ذخیره می‌شود.

آموزش کامل (۵۰۰–۱۰۰۰ اپیزود) روی CPU ممکن است چند ساعت طول بکشد؛ در صورت داشتن GPU سرعت بیشتر خواهد بود (PyTorch به‌طور خودکار GPU را تشخیص می‌دهد).

## ۵. ارزیابی و رسم نمودار (Task 8)

```bash
python evaluate.py --episodes 10
```
این دستور:
1. نمودارهای `reward_curve.png`، `length_curve.png` و `crash_rate.png` را در پوشه `plots/` می‌سازد.
2. مدل ذخیره‌شده را با سیاست حریصانه (greedy, بدون اکتشاف) روی ۱۰ اپیزود ارزیابی، و میانگین reward و نرخ تصادف را چاپ می‌کند.

برای دیدن ایجنت به‌صورت گرافیکی (۳بعدی یا top-down)، روی سیستمی که صفحه‌نمایش/GPU دارد اجرا کنید:
```bash
python evaluate.py --episodes 3 --render_3d
python evaluate.py --episodes 3 --render_3d --render_topdown
```

## جریان کلی کار (خلاصه مراحل)

1. نصب کتابخانه‌ها طبق بخش ۱.
2. اجرای `make_env.py` برای تست محیط.
3. اجرای `train.py --episodes 20` برای دیباگ.
4. بعد از اطمینان از صحت، اجرای `train.py --episodes 800` (یا بیشتر) برای آموزش نهایی.
5. اجرای `evaluate.py` برای رسم نمودارها و ارزیابی نهایی.
6. بررسی نمودارهای `plots/` و نوشتن پاسخ سوالات تحلیلی (آیا reward افزایش پیدا کرده؟ طول اپیزود چطور تغییر کرده؟ نرخ تصادف کاهش یافته؟).
7. تحویل: کد + `plots/` + `models/dqn_trained.pt` + `logs/training_log.json` + پاسخ سوالات.

## نکات پیاده‌سازی مهم

- **Action space گسسته:** ۶ اکشن طبق جدول PDF در `env_utils.py` تعریف شده‌اند (ترکیب فرمان و شتاب/ترمز).
- **Bellman update:** در `dqn_agent.py`، تابع `learn()` دقیقا فرمول
  `y = r + (1-d) * gamma * max_a' Q_target(s', a')` و loss از نوع MSE را پیاده کرده است.
- **Target network:** هر `target_update_freq` گام یادگیری (پیش‌فرض ۱۰۰۰) سینک می‌شود.
- **Epsilon-greedy:** epsilon به‌صورت خطی از ۱٫۰ به ۰٫۰۵ در طول `epsilon_decay_steps` کاهش می‌یابد.
- اگر ساختار observation در نسخه‌ی MetaDrive شما دیکشنری بود (نه آرایه ساده)، تابع `flatten_obs` به‌طور خودکار همه‌ی زیرآرایه‌ها را concatenate می‌کند، پس نیازی به تغییر دستی نیست.
- اگر خطای مربوط به `info["crash"]` گرفتید، ممکن است در نسخه‌ی نصب‌شده‌ی شما کلید متفاوتی مثل `"crash_vehicle"` یا `"out_of_road"` باشد؛ کد هر دو حالت را چک می‌کند اما بهتر است یک‌بار `print(info)` بگیرید و کلیدهای موجود را ببینید.
