import torch, torch.nn as nn, math, warnings, re, signal, sys, os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Устройство: {device}")

# ── ТОКЕНИЗАТОР ───────────────────────────────────
class Tokenizer:
    def __init__(self):
        chars = ("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                 "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "0123456789 \n\t.,!?;:'\"-()[]{}=+*/\\<>_@#$%^&|~`")
        self.special = ["<PAD>","<UNK>","<BOS>","<EOS>"]
        self.vocab = {}
        for i,t in enumerate(self.special): self.vocab[t]=i
        for i,c in enumerate(chars): self.vocab[c]=len(self.special)+i
        self.inv_vocab={v:k for k,v in self.vocab.items()}
        self.pad_id=self.vocab["<PAD>"]; self.unk_id=self.vocab["<UNK>"]
        self.bos_id=self.vocab["<BOS>"]; self.eos_id=self.vocab["<EOS>"]
        self.vocab_size=len(self.vocab)
    def encode(self,text):
        return [self.bos_id]+[self.vocab.get(c,self.unk_id) for c in text]
    def decode(self,ids):
        return "".join(self.inv_vocab.get(i,"") for i in ids
                       if self.inv_vocab.get(i,"") not in self.special)

# ── МОДЕЛЬ ────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self,e,h):
        super().__init__(); self.h=h; self.d=e//h
        self.q=nn.Linear(e,e,bias=False); self.k=nn.Linear(e,e,bias=False)
        self.v=nn.Linear(e,e,bias=False); self.o=nn.Linear(e,e,bias=False)
    def forward(self,x,mask=None):
        B,T,C=x.shape
        Q=self.q(x).view(B,T,self.h,self.d).transpose(1,2)
        K=self.k(x).view(B,T,self.h,self.d).transpose(1,2)
        V=self.v(x).view(B,T,self.h,self.d).transpose(1,2)
        s=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d)
        if mask is not None: s=s.masked_fill(mask==0,float('-inf'))
        return self.o(torch.matmul(torch.softmax(s,dim=-1),V).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self,e,h):
        super().__init__()
        self.a=SelfAttention(e,h)
        self.f=nn.Sequential(nn.Linear(e,e*4),nn.GELU(),nn.Linear(e*4,e))
        self.n1=nn.LayerNorm(e); self.n2=nn.LayerNorm(e)
    def forward(self,x,mask=None):
        x=x+self.a(self.n1(x),mask)
        return x+self.f(self.n2(x))

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,embed_dim=512,num_heads=8,num_layers=8,max_len=512):
        super().__init__(); self.max_len=max_len
        self.te=nn.Embedding(vocab_size,embed_dim)
        self.pe=nn.Embedding(max_len,embed_dim)
        self.blocks=nn.ModuleList([Block(embed_dim,num_heads) for _ in range(num_layers)])
        self.norm=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,vocab_size,bias=False)
    def forward(self,x):
        B,T=x.shape; pos=torch.arange(T,device=x.device).unsqueeze(0)
        out=self.te(x)+self.pe(pos)
        mask=torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0).unsqueeze(0)
        for b in self.blocks: out=b(out,mask)
        return self.head(self.norm(out))
    def count_params(self): return sum(p.numel() for p in self.parameters())

# ── ДАТАСЕТ ───────────────────────────────────────
raw_data = [
    "Пользователь: Привет!\ndemorg: Привет! Я demorg, готов помочь с кодом, ошибками и вопросами. Что нужно?",
    "Пользователь: Привет\ndemorg: Привет! Чем могу помочь?",
    "Пользователь: Здравствуй\ndemorg: Здравствуй! Задавай вопрос — помогу.",
    "Пользователь: Хай\ndemorg: Привет! Чем могу помочь?",
    "Пользователь: Добрый день\ndemorg: Добрый день! Готов помочь с вопросом.",
    "Пользователь: Доброе утро\ndemorg: Доброе утро! С чего начнём?",
    "Пользователь: Ты кто?\ndemorg: Я demorg — языковая модель HeartAI. Помогаю с кодом, ошибками, формулами и конспектами.",
    "Пользователь: Как тебя зовут?\ndemorg: Меня зовут demorg, я часть проекта HeartAI.",
    "Пользователь: Что ты умеешь?\ndemorg: Я умею писать код на Python и SQL, объяснять ошибки, помогать с формулами Excel и хоткеями VS Code.",
    "Пользователь: Кто тебя создал?\ndemorg: Меня обучили на небольшом датасете диалогов в рамках проекта HeartAI.",
    "Пользователь: Как дела?\ndemorg: Отлично! Готов помочь с кодом или ответить на вопросы.",
    "Пользователь: Как ты?\ndemorg: Работаю без сбоев! Чем могу помочь?",
    "Пользователь: Спасибо!\ndemorg: Пожалуйста! Если ещё что-то понадобится — обращайся.",
    "Пользователь: Спасибо за помощь!\ndemorg: Рад помочь! Удачи с проектом.",
    "Пользователь: Окей\ndemorg: Хорошо! Если появятся вопросы — пиши.",
    "Пользователь: Понял\ndemorg: Отлично! Если что-то непонятно — спрашивай.",
    "Пользователь: Не понимаю\ndemorg: Давай разберём подробнее. Что именно непонятно?",
    "Пользователь: Помоги мне\ndemorg: Конечно! Расскажи что нужно сделать.",
    "Пользователь: Пока!\ndemorg: До свидания! Обращайся если понадобится помощь.",
    "Пользователь: До свидания\ndemorg: Пока! Удачи в работе.",
    "Пользователь: Ты можешь писать код?\ndemorg: Да, могу писать код на Python, SQL и других языках. Что нужно написать?",
    "Пользователь: Напиши функцию суммы списка\ndemorg: def sum_list(lst):\n    return sum(lst)",
    "Пользователь: Напиши функцию переворота строки\ndemorg: def reverse_string(s):\n    return s[::-1]",
    "Пользователь: Напиши функцию факториала\ndemorg: def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "Пользователь: Напиши функцию простого числа\ndemorg: def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
    "Пользователь: Напиши функцию палиндрома\ndemorg: def is_palindrome(s):\n    return s == s[::-1]",
    "Пользователь: Напиши функцию Фибоначчи\ndemorg: def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Пользователь: Напиши функцию чтения файла\ndemorg: def read_file(path):\n    with open(path, 'r', encoding='utf-8') as f:\n        return f.read()",
    "Пользователь: Напиши функцию записи в файл\ndemorg: def write_file(path, text):\n    with open(path, 'w', encoding='utf-8') as f:\n        f.write(text)",
    "Пользователь: Напиши функцию поиска максимума в списке\ndemorg: def find_max(lst):\n    return max(lst)",
    "Пользователь: Напиши функцию сортировки списка\ndemorg: def sort_list(lst):\n    return sorted(lst)",
    "Пользователь: Напиши функцию удаления дубликатов\ndemorg: def remove_duplicates(lst):\n    return list(set(lst))",
    "Пользователь: Напиши декоратор для замера времени\ndemorg: import time\ndef timer(func):\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(time.time() - start)\n        return result\n    return wrapper",
    "Пользователь: Как создать список в Python\ndemorg: Список создаётся так: lst = [] или lst = list(). Добавить элемент: lst.append(значение).",
    "Пользователь: Как создать словарь в Python\ndemorg: Словарь создаётся так: d = {} или d = dict(). Добавить элемент: d['ключ'] = значение.",
    "Пользователь: Как создать множество в Python\ndemorg: Множество создаётся так: s = set(). Добавить элемент: s.add(значение).",
    "Пользователь: Что такое lambda\ndemorg: Lambda — анонимная функция в одну строку. Пример: double = lambda x: x * 2.",
    "Пользователь: Что такое генератор в Python\ndemorg: Генератор — функция с yield, которая возвращает значения по одному, не храня всё в памяти сразу.",
    "Пользователь: Что такое list comprehension\ndemorg: Это краткая запись для создания списков: [x*2 for x in range(10)].",
    "Пользователь: Как обработать исключение\ndemorg: Используй try-except:\ntry:\n    код\nexcept Exception as e:\n    print(e)",
    "Пользователь: Как открыть файл в Python\ndemorg: Используй контекстный менеджер: with open('file.txt') as f:\n    data = f.read()",
    "Пользователь: Напиши SQL выборку всех записей\ndemorg: SELECT * FROM название_таблицы;",
    "Пользователь: Напиши SQL последних 10 записей\ndemorg: SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;",
    "Пользователь: Напиши SQL с условием\ndemorg: SELECT * FROM users WHERE age > 18 AND city = 'Москва';",
    "Пользователь: Напиши SQL с группировкой\ndemorg: SELECT city, COUNT(*) FROM users GROUP BY city;",
    "Пользователь: Напиши SQL join двух таблиц\ndemorg: SELECT * FROM orders JOIN users ON orders.user_id = users.id;",
    "Пользователь: Что означает ошибка IndexError\ndemorg: IndexError — обращение к несуществующему индексу. Проверь длину: len(lst) перед обращением.",
    "Пользователь: Что означает ошибка KeyError\ndemorg: KeyError — ключа нет в словаре. Используй: d.get('ключ', значение_по_умолчанию).",
    "Пользователь: Что означает ошибка TypeError\ndemorg: TypeError — неправильный тип данных. Проверь тип: print(type(переменная)).",
    "Пользователь: Что означает ошибка NameError\ndemorg: NameError — переменная не объявлена. Проверь правильность написания имени.",
    "Пользователь: Что означает ZeroDivisionError\ndemorg: ZeroDivisionError — деление на ноль. Добавь проверку: if divisor != 0.",
    "Пользователь: Что означает ModuleNotFoundError\ndemorg: ModuleNotFoundError — модуль не найден. Установи: pip install название.",
    "Пользователь: Что означает AttributeError\ndemorg: AttributeError — у объекта нет такого атрибута или метода. Проверь имя и тип объекта.",
    "Пользователь: Что означает ValueError\ndemorg: ValueError — правильный тип, но некорректное значение. Например int('abc').",
    "Пользователь: Как открыть терминал в VS Code\ndemorg: Нажми Ctrl+` или выбери Terminal → New Terminal в верхнем меню.",
    "Пользователь: Как найти файл в VS Code\ndemorg: Нажми Ctrl+P и начни вводить название файла.",
    "Пользователь: Как найти текст в VS Code\ndemorg: Ctrl+F — поиск в файле. Ctrl+Shift+F — поиск по всему проекту.",
    "Пользователь: Как удалить строку в VS Code\ndemorg: Ctrl+Shift+K удаляет текущую строку.",
    "Пользователь: Как форматировать код в VS Code\ndemorg: Shift+Alt+F форматирует весь файл.",
    "Пользователь: Как закомментировать строку в VS Code\ndemorg: Ctrl+/ комментирует или раскомментирует текущую строку.",
    "Пользователь: Как открыть несколько терминалов в VS Code\ndemorg: Нажми + в панели терминала или Ctrl+Shift+` для нового терминала.",
    "Пользователь: Как посчитать сумму в Excel\ndemorg: Используй =СУММ(A1:A10) для суммы диапазона.",
    "Пользователь: Как найти среднее в Excel\ndemorg: Используй =СРЗНАЧ(A1:A10) для среднего значения.",
    "Пользователь: Как сделать условие в Excel\ndemorg: Используй =ЕСЛИ(условие, значение_если_истина, значение_если_ложь).",
    "Пользователь: Как найти значение в Excel по условию\ndemorg: Используй =ВПР(искомое, диапазон, номер_столбца, 0).",
    "Пользователь: Как посчитать количество ячеек в Excel\ndemorg: Используй =СЧЁТ(A1:A10) для чисел или =СЧЁТЗ(A1:A10) для всех непустых.",
    "Пользователь: Что такое Git\ndemorg: Git — система контроля версий. Сохраняет историю изменений и позволяет команде работать вместе.",
    "Пользователь: Как создать ветку в Git\ndemorg: git checkout -b название_ветки создаёт и переключает на новую ветку.",
    "Пользователь: Как отменить последний коммит\ndemorg: git reset --soft HEAD~1 отменяет коммит, но сохраняет изменения в staging.",
    "Пользователь: Как посмотреть историю коммитов\ndemorg: git log покажет историю. Добавь --oneline для краткого вида.",
    "Пользователь: Что такое merge conflict\ndemorg: Конфликт слияния — Git не может сам объединить изменения. Нужно вручную отредактировать файл и сделать git add.",
    "Пользователь: Что такое API\ndemorg: API — интерфейс для взаимодействия программ. Один сервис предоставляет его, другие вызывают для получения данных.",
    "Пользователь: Что такое REST API\ndemorg: REST API — архитектурный стиль API поверх HTTP с методами GET, POST, PUT, DELETE.",
    "Пользователь: Что такое JSON\ndemorg: JSON — формат данных. Выглядит как словарь: {'имя': 'Иван', 'возраст': 25}.",
    "Пользователь: Что такое ООП\ndemorg: ООП — объектно-ориентированное программирование. Код организуется в классы с методами и свойствами.",
    "Пользователь: Что такое наследование в ООП\ndemorg: Наследование — когда класс перенимает свойства и методы другого класса: class Cat(Animal).",
    "Пользователь: Что такое инкапсуляция\ndemorg: Инкапсуляция — сокрытие внутренних деталей объекта, доступ только через методы.",
    "Пользователь: Чем список отличается от кортежа\ndemorg: Список изменяемый ([1,2,3]), кортеж неизменяемый ((1,2,3)). Кортеж быстрее и безопаснее для констант.",
    "Пользователь: Чем массив отличается от списка\ndemorg: Массив хранит один тип данных и занимает меньше памяти, список Python может хранить разные типы.",
    "Пользователь: Что такое рекурсия\ndemorg: Рекурсия — функция вызывает саму себя. Нужно условие выхода, иначе будет бесконечный вызов.",
    "Пользователь: Что такое бинарный поиск\ndemorg: Алгоритм поиска в отсортированном массиве делением диапазона пополам, сложность O(log n).",
    "Пользователь: Что такое сложность алгоритма O(n)\ndemorg: O(n) значит время работы растёт линейно с размером входных данных.",
    "Пользователь: Сколько будет 2 плюс 2\ndemorg: 2 + 2 = 4.",
    "Пользователь: Сколько будет 15 умножить на 3\ndemorg: 15 * 3 = 45.",
    "Пользователь: Что такое простое число\ndemorg: Простое число делится только на 1 и на само себя, например 2, 3, 5, 7, 11.",
]

class TextDataset(Dataset):
    def __init__(self,texts,tok,max_len=256):
        self.samples=[]
        for text in texts:
            ids=tok.encode(text)
            for i in range(0,max(1,len(ids)-max_len),max_len//2):
                chunk=ids[i:i+max_len+1]
                if len(chunk)>8: self.samples.append(chunk)
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        ids=self.samples[idx]
        return torch.tensor(ids[:-1],dtype=torch.long),torch.tensor(ids[1:],dtype=torch.long)

def collate_fn(batch):
    xs,ys=zip(*batch); ml=max(x.size(0) for x in xs)
    xp=torch.zeros(len(xs),ml,dtype=torch.long)
    yp=torch.full((len(ys),ml),-100,dtype=torch.long)
    for i,(x,y) in enumerate(zip(xs,ys)):
        xp[i,:x.size(0)]=x; yp[i,:y.size(0)]=y
    return xp,yp

# ── СЭМПЛИРОВАНИЕ ─────────────────────────────────
def sample_next(logits, tokenizer, generated, temperature=0.8, top_k=40, top_p=0.9,
                repetition_penalty=1.3, recent_window=64):
    """Сэмплирование следующего токена вместо жадного argmax.

    Жадный argmax на char-level модели почти всегда сваливается в повтор
    одного слова. temperature + repetition penalty + top-k/top-p дают
    связные слова без зацикливания (та же логика, что и в heartai_space/app.py)."""
    logits = logits.clone()
    for sid in (tokenizer.bos_id, tokenizer.pad_id, tokenizer.unk_id):
        logits[sid] = float("-inf")
    if repetition_penalty and repetition_penalty != 1.0:
        for tid in set(generated[-recent_window:]):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty
    if temperature and temperature > 0:
        logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1).item())

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=120):
    model.eval()
    ids = tokenizer.encode(f"Пользователь: {prompt}\ndemorg:")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for _ in range(max_new_tokens):
        nl = model(x)[0, -1, :]
        nid = sample_next(nl, tokenizer, generated)
        if nid == tokenizer.eos_id: break
        generated.append(nid)
        x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
    model.train()
    answer = tokenizer.decode(x[0].tolist()[len(ids):])
    for stop in ["Пользователь:", "demorg:"]:
        if stop in answer: answer = answer.split(stop)[0]
    return answer.strip()

# ── ИНИЦИАЛИЗАЦИЯ ─────────────────────────────────
CHECKPOINT = "minigpt_v3.pt"
tokenizer  = Tokenizer()

if os.path.exists(CHECKPOINT):
    print(f"📂 Загружаю чекпоинт {CHECKPOINT}...")
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    cfg   = ckpt["config"]
    model = MiniGPT(vocab_size=tokenizer.vocab_size, **cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    start_epoch = ckpt.get("epoch", 0)
    best_loss   = ckpt.get("best_loss", float('inf'))
    print(f"✅ Продолжаю с эпохи {start_epoch} | Best loss: {best_loss:.4f}")
else:
    print("🆕 Создаю новую модель...")
    model = MiniGPT(vocab_size=tokenizer.vocab_size).to(device)
    start_epoch = 0
    best_loss   = float('inf')

print(f"   Параметров: {model.count_params()/1e6:.1f}M на {device}")

# 90/10 train/val split — чтобы видеть, учится модель или просто запоминает
split = max(1, int(len(raw_data) * 0.9))
shuffled = raw_data[:]
import random; random.seed(42); random.shuffle(shuffled)
train_texts, val_texts = shuffled[:split], shuffled[split:]

train_dataset = TextDataset(train_texts * 6, tokenizer)
val_dataset   = TextDataset(val_texts, tokenizer)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader    = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn) if len(val_dataset) else None

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# ── СОХРАНЕНИЕ ────────────────────────────────────
def save(epoch, loss, reason=""):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": loss,
        "vocab_size": tokenizer.vocab_size,
        "config": {
            "embed_dim":  model.te.weight.shape[1],
            "num_heads":  8,
            "num_layers": len(model.blocks),
            "max_len":    model.max_len,
        }
    }, CHECKPOINT)
    tag = f" [{reason}]" if reason else ""
    print(f"\n💾 Сохранено{tag}: эпоха {epoch} | loss {loss:.4f}")

current_epoch = start_epoch

# ── CTRL+C — сохраняет и выходит ──────────────────
def handle_exit(sig, frame):
    print("\n\n⛔ Остановка по Ctrl+C...")
    save(current_epoch, best_loss, "прерывание")
    print("✅ Прогресс сохранён.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

@torch.no_grad()
def evaluate():
    if val_loader is None: return None
    model.eval()
    total, n = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x).view(-1, tokenizer.vocab_size), y.view(-1))
        total += loss.item(); n += 1
    model.train()
    return total / max(1, n)

# ── ОБУЧЕНИЕ ──────────────────────────────────────
EPOCHS = 300
PRINT_EVERY = 20
PATIENCE = 8   # эпох подряд без улучшения val loss — датасет крошечный, легко переобучиться
DEMO_PROMPTS = ["Привет!", "Напиши функцию факториала", "Что такое Git"]

if __name__ == "__main__":
    print(f"📚 Train: {len(train_dataset)} чанков | Val: {len(val_dataset)} чанков\n")
    best_metric = float('inf')  # выбираем чекпоинт по val loss (train loss почти всегда обманчив)
    bad_evals = 0
    last_epoch = start_epoch
    for epoch in range(start_epoch, EPOCHS):
        current_epoch = last_epoch = epoch
        epoch_loss, steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item(); steps += 1
        train_loss = epoch_loss / max(1, steps)
        best_loss = min(best_loss, train_loss)

        val_loss = evaluate()
        metric = val_loss if val_loss is not None else train_loss

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            val_str = f" | val {val_loss:.4f}" if val_loss is not None else ""
            print(f"эпоха {epoch:4d} | train {train_loss:.4f}{val_str}")
            demo = generate(model, tokenizer, DEMO_PROMPTS[epoch // PRINT_EVERY % len(DEMO_PROMPTS)])
            print(f"   demo » {demo[:120]}")

        if metric < best_metric:
            best_metric, bad_evals = metric, 0
            save(epoch, train_loss, "лучший результат")
        else:
            bad_evals += 1
            if bad_evals >= PATIENCE:
                print(f"\n⏹️  Early stopping на эпохе {epoch}: "
                      f"val loss не улучшается {PATIENCE} эпох подряд.")
                break

    save(last_epoch, best_loss, "финал")
    print(f"\n🏁 Обучение завершено. Чекпоинт: {os.path.abspath(CHECKPOINT)}")

    # Опционально — залить чекпоинт на Hugging Face Hub, чтобы его подхватил
    # heartai_space/app.py (см. HF_HUB_UPLOAD.md). Ничего не делает, если
    # переменная окружения HF_TOKEN не задана.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_repo_id = os.environ.get("HF_REPO_ID", "jfenviejijeijef/heartai-demorg")
        from huggingface_hub import HfApi
        print(f"\n☁️  Заливаю {CHECKPOINT} в {hf_repo_id}...")
        HfApi(token=hf_token).upload_file(
            path_or_fileobj=CHECKPOINT,
            path_in_repo=CHECKPOINT,
            repo_id=hf_repo_id,
        )
        print("✅ Готово — Space подхватит новый чекпоинт при следующем запуске.")
    else:
        print("\nℹ️  Чтобы залить чекпоинт на Hugging Face Hub автоматически, "
              "задай переменную окружения HF_TOKEN и перезапусти скрипт "
              "(или загрузи файл вручную — см. README).")
