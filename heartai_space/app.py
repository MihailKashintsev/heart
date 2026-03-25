
import torch, torch.nn as nn, math, secrets, os, warnings, re
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download, snapshot_download

warnings.filterwarnings("ignore")

# ── Пробуем загрузить ruGPT-3 ─────────────────────
USE_RUGPT = False
model     = None
tokenizer = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Пробую загрузить ruGPT-3...")
    rugpt_path = snapshot_download(
        repo_id="jfenviejijeijef/heartai-demorg",
        allow_patterns=["rugpt3/*"],
    )
    rugpt_dir = os.path.join(rugpt_path, "rugpt3")
    if os.path.exists(os.path.join(rugpt_dir, "config.json")):
        tokenizer = AutoTokenizer.from_pretrained(rugpt_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(rugpt_dir)
        model.eval()
        USE_RUGPT = True
        print(f"✅ ruGPT-3 загружена! Параметров: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    else:
        print("ruGPT-3 ещё не готова — использую базовую модель")
except Exception as e:
    print(f"ruGPT-3 недоступна ({e}) — использую базовую модель")

# ── Базовая модель (fallback) ──────────────────────
class CharTokenizer:
    def __init__(self):
        chars = ("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                 "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                 "abcdefghijklmnopqrstuvwxyz"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "0123456789 \n\t.,!?;:\'\"-()[]{}=+*/\\<>_@#$%^&|~`")
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
        if mask is not None: s=s.masked_fill(mask==0,float("-inf"))
        return self.o(torch.matmul(torch.softmax(s,dim=-1),V).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self,e,h):
        super().__init__()
        self.a=SelfAttention(e,h)
        self.f=nn.Sequential(nn.Linear(e,e*4),nn.GELU(),nn.Linear(e*4,e))
        self.n1=nn.LayerNorm(e); self.n2=nn.LayerNorm(e)
    def forward(self,x,mask=None):
        x=x+self.a(self.n1(x),mask); return x+self.f(self.n2(x))

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,embed_dim=256,num_heads=8,num_layers=6,max_len=512):
        super().__init__(); self.max_len=max_len
        self.te=nn.Embedding(vocab_size,embed_dim); self.pe=nn.Embedding(max_len,embed_dim)
        self.blocks=nn.ModuleList([Block(embed_dim,num_heads) for _ in range(num_layers)])
        self.norm=nn.LayerNorm(embed_dim); self.head=nn.Linear(embed_dim,vocab_size,bias=False)
    def forward(self,x):
        B,T=x.shape; pos=torch.arange(T,device=x.device).unsqueeze(0)
        out=self.te(x)+self.pe(pos)
        mask=torch.tril(torch.ones(T,T,device=x.device)).unsqueeze(0).unsqueeze(0)
        for b in self.blocks: out=b(out,mask)
        return self.head(self.norm(out))
    def count_params(self): return sum(p.numel() for p in self.parameters())

device = torch.device("cpu")

if not USE_RUGPT:
    print("Загружаю базовую модель...")
    char_tok = CharTokenizer()
    try:
        path = hf_hub_download(
            repo_id="jfenviejijeijef/heartai-demorg",
            filename="minigpt_v3.pt", force_download=True)
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt["model_state"]
        embed_dim  = sd["te.weight"].shape[1]
        num_layers = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))+1
        base_model = MiniGPT(char_tok.vocab_size,embed_dim,8,num_layers,512).to(device)
        base_model.load_state_dict(sd)
        base_model.eval()
        print(f"✅ Базовая модель: {base_model.count_params():,} параметров")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        base_model = None

# ── ГЕНЕРАЦИЯ ─────────────────────────────────────
def chat(text):
    if USE_RUGPT:
        return chat_rugpt(text)
    else:
        return chat_base(text)

def chat_rugpt(text):
    prompt = f"Пользователь: {text}\ndemorg:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens     = 100,
            do_sample          = True,
            temperature        = 0.7,
            top_p              = 0.9,
            repetition_penalty = 1.3,
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    if "demorg:" in full:
        answer = full.split("demorg:")[-1].strip()
        if "Пользователь:" in answer:
            answer = answer.split("Пользователь:")[0].strip()
    else:
        answer = full[len(prompt):].strip()
    return answer[:300]

def chat_base(text):
    if base_model is None:
        return "Модель загружается, попробуй позже."
    prompt = f"Пользователь: {text}\ndemorg:"
    ids = char_tok.encode(prompt)
    if len(ids) > 400: ids = ids[-400:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(150):
            nl = base_model(x)[0,-1,:].clone()
            nl[char_tok.bos_id]=float("-inf")
            nl[char_tok.pad_id]=float("-inf")
            nl[char_tok.unk_id]=float("-inf")
            nid = torch.argmax(nl).item()
            if nid == char_tok.eos_id: break
            x = torch.cat([x, torch.tensor([[nid]])], dim=1)
    answer = char_tok.decode(x[0].tolist()[len(ids):])
    for stop in ["Пользователь:", "demorg:"]:
        if stop in answer: answer = answer.split(stop)[0]
    is_code = any(kw in answer for kw in ["def ","return ","SELECT","print(","    "])
    if not is_code:
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        answer = " ".join(parts[:2])
    return answer.strip()

# ── API ───────────────────────────────────────────
import os as _os
API_KEYS = {
    _os.environ.get("MAIN_API_KEY","sk-my-main-key-001"): {"name":"main","active":True}
}

def check_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Формат: Bearer sk-...")
    key = authorization.split(" ",1)[1]
    if key not in API_KEYS or not API_KEYS[key]["active"]:
        raise HTTPException(status_code=403, detail="Неверный ключ")
    return API_KEYS[key]

app = FastAPI(title="HeartAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str
    use_search: bool = False

class KeyRequest(BaseModel):
    name: str

def total_params():
    if USE_RUGPT: return sum(p.numel() for p in model.parameters())
    if base_model: return base_model.count_params()
    return 0

@app.get("/")
def root():
    return {"status":"ok","model":"ruGPT-3" if USE_RUGPT else "demorg-base","params":total_params()}

@app.get("/v1/health")
def health():
    m = "ruGPT-3 fine-tuned" if USE_RUGPT else "demorg base"
    return {"status":"online","parameters":f"{total_params():,}","model":m,"device":str(device)}

@app.post("/v1/ask")
def ask(req: AskRequest, authorization: str = Header(...)):
    check_key(authorization)
    return {"question":req.question,"answer":chat(req.question)}

@app.post("/v1/keys/new")
def new_key(req: KeyRequest, authorization: str = Header(...)):
    check_key(authorization)
    key = "sk-" + secrets.token_hex(12)
    API_KEYS[key] = {"name":req.name,"active":True}
    return {"key":key,"name":req.name}
