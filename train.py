import torch, torch.nn as nn, math, warnings, re, signal, sys
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
    "Пользователь: Ты кто?\ndemorg: Я demorg — языковая модель HeartAI. Помогаю с кодом, ошибками, формулами и конспектами.",
    "Пользователь: Что ты умеешь?\ndemorg: Я умею писать код на Python и SQL, объяснять ошибки, помогать с формулами Excel и хоткеями VS Code.",
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
    "Пользователь: Как создать список в Python\ndemorg: Список создаётся так: lst = [] или lst = list(). Добавить элемент: lst.append(значение).",
    "Пользователь: Как создать словарь в Python\ndemorg: Словарь создаётся так: d = {} или d = dict(). Добавить элемент: d['ключ'] = значение.",
    "Пользователь: Что такое lambda\ndemorg: Lambda — анонимная функция в одну строку. Пример: double = lambda x: x * 2.",
    "Пользователь: Как обработать исключение\ndemorg: Используй try-except:\ntry:\n    код\nexcept Exception as e:\n    print(e)",
    "Пользователь: Напиши SQL выборку всех записей\ndemorg: SELECT * FROM название_таблицы;",
    "Пользователь: Напиши SQL последних 10 записей\ndemorg: SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;",
    "Пользователь: Напиши SQL с условием\ndemorg: SELECT * FROM users WHERE age > 18 AND city = 'Москва';",
    "Пользователь: Что означает ошибка IndexError\ndemorg: IndexError — обращение к несуществующему индексу. Проверь длину: len(lst) перед обращением.",
    "Пользователь: Что означает ошибка KeyError\ndemorg: KeyError — ключа нет в словаре. Используй: d.get('ключ', значение_по_умолчанию).",
    "Пользователь: Что означает ошибка TypeError\ndemorg: TypeError — неправильный тип данных. Проверь тип: print(type(переменная)).",
    "Пользователь: Что означает ошибка NameError\ndemorg: NameError — переменная не объявлена. Проверь правильность написания имени.",
    "Пользователь: Что означает ZeroDivisionError\ndemorg: ZeroDivisionError — деление на ноль. Добавь проверку: if divisor != 0.",
    "Пользователь: Что означает ModuleNotFoundError\ndemorg: ModuleNotFoundError — модуль не найден. Установи: pip install название.",
    "Пользователь: Как открыть терминал в VS Code\ndemorg: Нажми Ctrl+` или выбери Terminal → New Terminal в верхнем меню.",
    "Пользователь: Как найти файл в VS Code\ndemorg: Нажми Ctrl+P и начни вводить название файла.",
    "Пользователь: Как найти текст в VS Code\ndemorg: Ctrl+F — поиск в файле. Ctrl+Shift+F — поиск по всему проекту.",
    "Пользователь: Как удалить строку в VS Code\ndemorg: Ctrl+Shift+K удаляет текущую строку.",
    "Пользователь: Как форматировать код в VS Code\ndemorg: Shift+Alt+F форматирует весь файл.",
    "Пользователь: Как посчитать сумму в Excel\ndemorg: Используй =СУММ(A1:A10) для суммы диапазона.",
    "Пользователь: Как найти среднее в Excel\ndemorg: Используй =СРЗНАЧ(A1:A10) для среднего значения.",
    "Пользователь: Как сделать условие в Excel\ndemorg: Используй =ЕСЛИ(условие, значение_если_истина, значение_если_ложь).",
    "Пользователь: Что такое Git\ndemorg: Git — система контроля версий. Сохраняет историю изменений и позволяет команде работать вместе.",
    "Пользователь: Что такое API\ndemorg: API — интерфейс для взаимодействия программ. Один сервис предоставляет его, другие вызывают для получения данных.",
    "Пользователь: Что такое JSON\ndemorg: JSON — формат данных. Выглядит как словарь: {'имя': 'Иван', 'возраст': 25}.",
    "Пользователь: Что такое ООП\ndemorg: ООП — объектно-ориентированное программирование. Код организуется в классы с методами и свойствами.",
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

# ── ИНИЦИАЛИЗАЦИЯ ─────────────────────────────────
CHECKPOINT = "minigpt_v3.pt"
tokenizer  = Tokenizer()

# Загружаем существующий чекпоинт если есть
if __import__('os').path.exists(CHECKPOINT):
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

dataset = TextDataset(raw_data * 6, tokenizer)
loader  = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

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

# ── CTRL+C — сохраняет и выходит ──────────────────
def handle_exit(sig, frame):
    print("\n\n⛔ Остановка по Ctrl+C...")
    save(current_epoch, best_loss, "прерывание")
    print("✅ Прогресс сохранён. Запусти снова чтобы продолжить.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# ── БЕСКОНЕЧНОЕ ОБУЧЕНИЕ ──────────────────────────
current_epoch = start_epoch
print(f"\n🚀 Бесконечное обучение запущено. Ctrl+C — остановить и сохранить.\n")

while True:
    current_epoch += 1
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        loss = criterion(model(x).view(-1,tokenizer.vocab_size), y.view(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); total += loss.item()
    scheduler.step()

    avg = total / len(loader)
    if avg < best_loss:
        best_loss = avg

    # Показываем прогресс каждые 50 эпох
    if current_epoch % 50 == 0:
        print(f"Эпоха {current_epoch:5d} | Loss: {avg:.4f} | Best: {best_loss:.4f}")

    # Сохраняем каждые 200 эпох автоматически
    if current_epoch % 200 == 0:
        save(current_epoch, best_loss, "авто")

