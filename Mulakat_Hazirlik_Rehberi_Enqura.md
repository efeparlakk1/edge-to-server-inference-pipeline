# Edge-to-Server Inference Pipeline — Sıfırdan Mülakat Hazırlık Rehberi

Bu rehber, `STUDY_TR.md` içindeki tüm kavramları **kanıta dayalı, sayısal örneklerle** ve **sıfır bilgi varsayımıyla** açıklar. Amaç ezber değil, her kavramın *neden* öyle çalıştığını anlamak — çünkü mülakatta sorulan ikinci soru genelde "neden?" olur.

**Nasıl çalışmalısın:** Her bölümü oku → örneği kendi kelimelerinle tekrar anlat (retrieval practice — hatırlama pratiği, bilginin kalıcılığını artırdığı deneysel olarak gösterilmiş bir yöntemdir) → bölüm sonundaki soruyu kağıda cevap yazmadan, sadece zihninde cevapla.

---

## BÖLÜM 0 — Büyük Resim

Proje tek bir soruyu cevaplıyor: **Aynı eğitilmiş model, hem küçük/güçsüz cihazlarda (edge) hem de yüksek trafikli sunucularda (server) nasıl verimli çalışır?**

```
CIFAR-10 görüntü
  → ResNet-18 (embedding üretici)
  → 512 boyutlu vektör (embedding)
  → ONNX'e aktarılır (taşınabilir format)
      ├─ INT8'e sıkıştırılır (edge için küçük)
      ├─ TFLite'a çevrilir (mobil için)
      └─ Triton sunucusuna yüklenir (yüksek throughput için)
```

Tek cümlelik özet: **Bir "temsil" (representation) öğreniliyor, sonra bu temsil iki farklı hedefe göre iki farklı biçimde paketleniyor.**

---

## BÖLÜM 1 — Python Sözdizimi (Bu Projede Geçen Her Yapı)

### 1.1 `import` çeşitleri

```python
import torch                          # tüm modülü, tam isimle getirir → torch.tensor(...)
import torch.nn as nn                 # kısaltma (alias) → nn.Linear(...)
from src.models.resnet_arcface import build_model   # sadece bir ismi doğrudan getirir
```

`src.models.resnet_arcface` ifadesi, disk üzerindeki `src/models/resnet_arcface.py` dosyasına **birebir** karşılık gelir. Nokta (`.`), klasör ayracının (`/`) Python'daki karşılığıdır.

**Neden önemli:** Mülakatta "bu import nereden geliyor?" diye sorulursa, dosya yolunu anında çıkarabilmelisin.

### 1.2 Fonksiyon imzası: type hint + default değer

```python
def build_model(embedding_dim: int = 512, pretrained: bool = True) -> nn.Module:
    ...
```

- `embedding_dim: int` → *tip ipucu* (type hint). Python bunu **çalışma zamanında zorlamaz**; sadece IDE/okuyucu için belge niteliğindedir. Bu kritik bir ayrım: `build_model(embedding_dim="512")` yazsan bile Python hata vermez, sadece mantıksal hata oluşabilir.
- `= 512` → argümanı isteğe bağlı (optional) yapar.
- `-> nn.Module` → dönüş tipini belgeler.

**Tuzak (mülakat sorusu):** Mutable bir nesneyi (örn. `[]`, `{}`) default değer olarak kullanmak tehlikelidir, çünkü o nesne fonksiyon tanımlandığında **bir kez** oluşturulur ve tüm çağrılar arasında paylaşılır.

```python
def add_item(x, items=[]):   # YANLIŞ desen
    items.append(x)
    return items

add_item(1)   # [1]
add_item(2)   # [1, 2]  ← beklenmedik! Aynı liste paylaşıldı.
```

Doğrusu: `items=None` verip fonksiyon içinde `if items is None: items = []`.

### 1.3 `argparse` — komut satırı argümanları

```python
p.add_argument("--batch-size", type=int, default=256)
cfg = p.parse_args()
print(cfg.batch_size)   # 128 (eğer --batch-size 128 verildiyse)
```

Kural: argüman adındaki `-` karakteri, attribute adında `_` olur (`--batch-size` → `cfg.batch_size`).

**Projeye özgü hata (gerçek bir mülakat sorusu):** `--pretrained` hem `action="store_true"` hem `default=True` ile tanımlanmış. Sonuç: CLI'dan kapatılamaz (`--pretrained` vermesen de zaten `True`). Doğru çözüm: `action=argparse.BooleanOptionalAction` kullanmak, böylece hem `--pretrained` hem `--no-pretrained` mümkün olur. *(Düzeltildi: Kod içerisinde `argparse.BooleanOptionalAction` kullanılarak her iki durum desteklendi.)*

### 1.4 Comprehension ve unpacking

```python
train_loader, val_loader = get_dataloaders()          # tuple unpacking
results = [None] * N_THREADS                          # N_THREADS elemanlı, hepsi None
threads = [threading.Thread(target=worker, args=(results, i)) for i in range(N_THREADS)]
all_latencies = [lat for worker_result in results for lat in worker_result]  # iç içe listeyi düzleştirir
```

Son satır kritik: `results` bir **liste listesi** (her thread'in kendi latency listesi). Nested comprehension bunu tek boyutlu listeye indirger. Sayısal örnek:

```python
results = [[1, 2], [3, 4]]
flat = [x for row in results for x in row]   # [1, 2, 3, 4]
```

### 1.5 f-string

```python
print(f"Val acc: {val_acc * 100:.2f}%")   # 87.35%  → .2f: virgülden sonra 2 basamak
print(f"Requests: {total_reqs:,}")        # 16,000  → binlik ayraç
```

### 1.6 Context manager (`with`)

```python
with torch.no_grad():
    ...
with open(path, "w") as file:
    file.write(text)
```

`with`, bir kaynağı açar ve **hata olsa bile** kapatılmasını garanti eder. `torch.no_grad()` gradyan kaydını (autograd graph) kapatır → bellek ve zaman tasarrufu. `open()` ise dosya tanıtıcısını (file handle) güvenli şekilde kapatır.

### 1.7 Exception handling — burada bir zayıflık var

```python
try:
    output = subprocess.check_output([...])
except Exception:
    return {"total_mb": 4096, "free_mb": 4096}
```

`except Exception` çok geniş bir yakalama; `nvidia-smi` bulunamaması (`FileNotFoundError`) ile beklenmeyen bir parsing hatasını **ayırt etmeden** aynı şekilde ele alır. Bu, gerçek bir bug'ı sessizce gizleyebilir. Production kodda beklenen hata tiplerini ayrı ayrı yakalamak gerekir. *(Düzeltildi: `scripts/generate_triton_config.py` içerisinde `(FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError)` spesifik olarak yakalandı.)*

### 1.8 Script giriş noktası

```python
if __name__ == "__main__":
    main()
```

Python, bir dosya **doğrudan** çalıştırıldığında `__name__` değişkenine `"__main__"` atar; dosya **import edildiğinde** ise modül adını atar. Bu sayede `import train` yaptığında training otomatik başlamaz.

> 🎯 **Olası soru:** "Bu kontrol olmasaydı ne olurdu?" → Dosya başka bir yerden import edildiğinde `main()` istemeden çalışırdı (örneğin test dosyasında import ettiğinde training tetiklenirdi).

---

## BÖLÜM 2 — Derin Öğrenme Temelleri

### 2.1 Tensor ve shape

Tensor, çok boyutlu bir sayı dizisidir. `(B, 3, 32, 32)` şu anlama gelir:
- `B` = batch size (aynı anda işlenen görüntü sayısı)
- `3` = kanal sayısı (RGB)
- `32, 32` = yükseklik × genişlik (piksel)

Bu sıralama **channel-first (NCHW)** olarak adlandırılır ve PyTorch'un varsayılanıdır (TensorFlow genelde NHWC — channel-last kullanır). Bu bir fark, hata değil; ama iki framework arası veri taşırken kanal sırasını çevirmezsen model yanlış öğrenir.

### 2.2 ResNet-18 ne yapar?

ResNet, "residual connection" (kalan bağlantı) fikrine dayanır: her katman bloğu, girdiyi doğrudan çıktıya ekleyen bir "kestirme yol" (skip connection) içerir: `output = F(x) + x`. Bu, çok derin ağlarda gradyanın kaybolması (vanishing gradient) problemini azaltır — gradyan bu kestirme yoldan doğrudan geriye akabilir.

### 2.3 Classification head vs Embedding head (en temel fark)

Standart ImageNet ResNet-18'in son katmanı `nn.Linear(512, 1000)` şeklindedir — 1000 sınıf olasılığı üretir. Bu proje son katmanı değiştiriyor:

```python
model.fc = nn.Sequential(
    nn.Linear(512, embedding_dim, bias=False),
    nn.BatchNorm1d(embedding_dim),
)
```

Sonuç artık bir olasılık dağılımı **değil**, 512 sayıdan oluşan bir **koordinat** (embedding). Bunun anlamı: model "bu resim kedi mi köpek mi?" demiyor, "bu resmi 512 boyutlu uzayda nereye yerleştirmeliyim?" diyor. Bu ayrım Bölüm 3'ün temelidir.

**Neden `bias=False` sonra BatchNorm var?** BatchNorm zaten öğrenilen bir kaydırma (shift, β parametresi) içerir. Linear katmana ayrıca bias eklemek, aynı işlemi iki kez yapmak (redundant) olur — az miktarda gereksiz parametre.

### 2.4 Transfer learning — avantaj ve gerçek bir uyumsuzluk

`ResNet18_Weights.DEFAULT`, ImageNet üzerinde (1.2M görüntü, 224×224) öğrenilmiş ağırlıklarla başlar. Ancak burada iki gerçek uyumsuzluk var:

1. Standart ResNet'in ilk katmanı 224×224 gibi büyük görüntüler için 7×7 stride-2 konvolüsyon + max-pooling kullanır. 32×32 CIFAR görüntüsünde bu **çok agresif** downsample yapar (bilgi kaybı).
2. ImageNet normalizasyon istatistikleri ile CIFAR-10 istatistikleri farklıdır; bu proje CIFAR normalizasyonu kullanıyor.

**Bu bir hata değil ama mülakatta göstermen gereken bir engineering gözlemi:** "Model çalışır, ama CIFAR'a özgü bir stem (3×3, stride 1, pooling yok) daha verimli olurdu; bunu ölçmeden iddia etmem."

### 2.5 Veri artırma (augmentation) sırası neden önemli?

```
RandomCrop(32, padding=4) → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize
```

Mantık: **Görsel/uzamsal işlemler önce, sayısal normalizasyon en son.** `ToTensor()` piksel değerlerini `[0,255]` tam sayıdan `[0,1]` float'a çevirir; `Normalize()` bu float tensor'u bekler ve her kanal için `(değer - ortalama) / std` uygular. Sırayı bozarsan (örn. Normalize'den önce ColorJitter uygularsan) renk bozulmaları normalizasyonu geçersiz kılabilir.

**Validation'da neden augmentation yok?** Çünkü değerlendirme hedefi **sabit** kalmalı — rastgelelik ekleseydin, aynı modelin aynı veri üzerindeki skoru her çalıştırmada değişirdi; bu da karşılaştırmayı anlamsız kılar.

> 🎯 **Olası soru:** "`drop_last=True` neden var?" → Yorumlar BatchNorm ile ilişkilendiriyor ama daha güçlü gerekçe **batch tutarlılığı**dır: son eksik batch, ölçüm ve gradyan davranışını farklılaştırabilir.

---

## BÖLÜM 3 — Metric Learning ve ArcFace (En Kritik Bölüm)

### 3.1 Zihniyet farkı

| | Klasik Classification | Metric Learning |
|---|---|---|
| Soru | "Bu hangi sabit sınıf?" | "Bu öğe, benzer öğelere yakın, farklı öğelere uzak olacak şekilde uzayda nereye oturmalı?" |
| Yeni sınıf eklemek | Son katmanı yeniden eğitmen gerekir | Sadece yeni bir embedding hesaplarsın, model sabit kalır |
| Çıktı | Olasılık dağılımı | Koordinat vektörü |

### 3.2 Cosine similarity — somut sayısal örnek

```
cosine_similarity(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Üç örnek:
- `a = [1, 0]`, `b = [1, 0]` → aynı yön → benzerlik = **1**
- `a = [1, 0]`, `b = [0, 1]` → dik (90°) → benzerlik = **0**
- `a = [1, 0]`, `b = [-1, 0]` → zıt yön (180°) → benzerlik = **-1**

Yani benzerlik, **yön** ölçer, **büyüklük** değil. İki embedding'in "aynı kişi/sınıf" olup olmadığını anlamak için vektörlerin uzunluğu değil, aralarındaki açı önemlidir.

### 3.3 Angular margin — somut sayısal örnek

Diyelim embedding'in class-A proxy'sine açısı **30°**, class-B proxy'sine açısı **40°**. `cos(30°) ≈ 0.866`, `cos(40°) ≈ 0.766`. `0.866 > 0.766` olduğundan model doğru şekilde A'yı seçer.

ArcFace, eğitim sırasında doğru sınıfın açısına ekstra bir "ceza" (margin, `m`) ekler. Yani model, A sınıfını seçebilmek için artık `cos(30° + m)` değerini `cos(40°)`'tan büyük tutmak zorundadır. Eğer `m = 10°` ise, gerçek eşik `cos(40°) = cos(40°)` olur — yani sınır durumu. Bu, modeli, gerçek açıyı **30°'nin altına** indirmeye (aynı sınıf örneklerini daha da sıkılaştırmaya) zorlar.

**Sonuç:** Margin ne kadar büyükse, model aynı sınıf içi kümeleri o kadar sıkı, sınıflar arası boşluğu o kadar geniş yapmaya zorlanır — ama optimizasyon da o kadar zorlaşır.

### 3.4 `margin` ve `scale`

- **Margin** büyüdükçe → sınıf ayrımı güçlenir, ama eğitim (convergence) zorlaşır.
- **Scale**, sınırlı `[-1, 1]` aralığındaki cosine değerlerini, cross-entropy'nin işleyebileceği büyüklükte logit'lere çevirir (örn. `cos_değeri × 64`).
- Literatürde standart ayar: margin `0.5` **radyan**, scale `64`.

### 3.5 Projedeki gerçek hata — derece/radyan karışıklığı (EN ÖNEMLİ DETAY)

`pytorch-metric-learning` kütüphanesindeki `ArcFaceLoss`, `margin` parametresini **derece** cinsinden bekler (varsayılan: `28.6°`, ki bu `≈ 0.5 radyan`'a eşittir). Ancak proje kodda `margin=0.5` geçiriyor.

Hesap:
```
0.5 radyan × (180 / π) ≈ 28.6 derece   ← niyet edilen değer
0.5 (olduğu gibi derece kabul edilirse) = 0.5 derece   ← gerçekte uygulanan değer
```

**Sonuç:** Uygulanan margin, niyet edilenin **yaklaşık 1/57'si** kadar küçük. Bu, ArcFace'in temel amacı olan "güçlü açısal ayrım"ı neredeyse tamamen etkisiz kılar. *(Düzeltildi: `src/losses/arcface.py` ve `train.py` dosyalarında `margin=28.6` derece olarak güncellendi.)*

> 🎯 **Kesin sorulacak soru:** "Bu hatayı checkpoint'i değiştirmeden düzeltir misin?" → **Hayır.** Margin, loss fonksiyonunun bir parçasıdır; değiştirmek eğitim hedefini (objective) değiştirir. Mevcut checkpoint, yanlış (küçük) margin ile eğitildi. Doğru margin (`≈28.6`) ile **yeniden eğitim** ve **yeniden değerlendirme** gerekir.

### 3.6 `criterion.parameters()` neden optimizer'a ekleniyor?

```python
optimizer = optim.SGD(
    list(model.parameters()) + list(criterion.parameters()), ...
)
```

ArcFace, her sınıf için öğrenilebilir bir "proxy vektörü" (o sınıfın uzaydaki temsilcisi) tutar. Bu vektörler de gradyan inişiyle güncellenmesi gereken parametrelerdir. Eğer `criterion.parameters()` eklenmezse, proxy'ler rastgele başlangıç değerlerinde donmuş kalır ve loss anlamlı şekilde öğrenemez.

### 3.7 Eğitim sınırı ile inference sınırı arasındaki fark

- **Validation sırasında:** `criterion.get_logits(embeddings)` çağrılır → ArcFace proxy'leri kullanılarak sınıf tahmini yapılır → "proxy-classification accuracy" raporlanır.
- **Export/deployment sırasında:** Sadece `model` (backbone) export edilir, `criterion` **dahil değildir**.
- **Sonuç:** Triton'a sorduğunda sana bir sınıf ismi değil, 512 sayılık bir embedding döner. Kullanışlı bir ürün için bu embedding'i sen (veya downstream sistem) cosine similarity / k-NN / eşik (threshold) ile yorumlamalısın.

**Ek detay:** Export edilen graph, embedding'i açıkça L2-normalize etmez. Cosine retrieval yapmadan önce embedding'i normalize etmek (`v / ‖v‖`) gerekir, aksi halde büyüklük (magnitude) farkları benzerlik hesabını bozar.

---

## BÖLÜM 4 — Training Loop

### 4.1 Kesin sıra (ezber değil, mantık)

```python
optimizer.zero_grad(set_to_none=True)   # 1. eski gradyanları temizle
embeddings = model(images)              # 2. ileri geçiş (forward)
loss = criterion(embeddings, labels)    # 3. skaler kayıp hesapla
loss.backward()                         # 4. geri yayılım (gradyan hesapla)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 5. gradyan kırpma
optimizer.step()                        # 6. parametreleri güncelle
```

**Neden `zero_grad` en başta?** PyTorch, `backward()` çağrıldığında gradyanları **biriktirir** (accumulate), üzerine yazmaz. Eğer temizlemezsen, her iterasyonda önceki adımın gradyanı da eklenir → yanlış (şişirilmiş) güncelleme.

**`set_to_none=True` neden daha verimli?** Gradyan tensor'larını sıfırlarla doldurmak yerine tamamen serbest bırakır (`None` yapar); bir sonraki `backward()` gerekli belleği zaten yeniden ayırır. Bellek/performans optimizasyonu, davranış değişikliği değil.

### 4.2 Gradient clipping — ve projedeki eksik

`clip_grad_norm_`, belirtilen parametrelerin **toplam (global) gradyan normunu** hesaplar; eğer `max_norm`'u aşıyorsa hepsini orantılı şekilde küçültür. Amaç: "patlayan gradyan" (exploding gradient) durumunda tek bir kötü adımın modeli bozmasını engellemek.

**Eksik:** Sadece `model.parameters()` kırpılıyor; `criterion.parameters()` (ArcFace proxy'leri) kırpılmıyor. Eğer proxy gradyanları patlarsa, bu korumasız kalır. Tutarlı bir versiyon, birleşik parametre kümesini kırpmalıdır. *(Düzeltildi: `train.py` içinde `list(model.parameters()) + list(criterion.parameters())` birleşik parametre kümesi kırpıldı.)*

### 4.3 Optimizer bileşenleri

| Bileşen | Değer | Ne yapar |
|---|---|---|
| Learning rate | 0.1 | Her adımda ne kadar büyük hareket edileceği |
| Momentum | 0.9 | Önceki gradyan yönlerini hatırlayıp güncellemeyi yumuşatır/hızlandırır |
| Weight decay | 5e-4 | L2 regularizasyon — büyük ağırlıkları cezalandırır, overfitting'i azaltır |
| Nesterov | açık | Momentumla "ileri bakılan" noktada gradyan hesaplanır — standart momentumdan daha isabetli düzeltme sağlar |

**SGD vs AdamW:** Tek "en iyi" optimizer yoktur. SGD+momentum, görüntü sınıflandırmada güçlü ve klasik bir seçimdir ama learning rate ayarına (tuning) daha duyarlıdır. AdamW genelde daha az başlangıç ayarı gerektirir çünkü adaptif öğrenme oranı kullanır.

### 4.4 Scheduler — `CosineAnnealingLR`

Learning rate'i, 30 epoch boyunca başlangıç değerinden `1e-5`'e doğru **kosinüs eğrisi** şeklinde azaltır (başta yavaş azalma, ortada hızlı, sonda tekrar yavaş). Mantık: eğitimin başında büyük adımlarla hızlı keşif, sonunda küçük adımlarla ince ayar (fine-tuning).

### 4.5 `model.train()` vs `model.eval()` vs `torch.no_grad()` — üçü farklı problemleri çözer

- `model.train()`: BatchNorm'un **o anki batch'in** ortalama/varyansını kullanmasını ve bunları biriken (running) istatistiklere eklemesini sağlar.
- `model.eval()`: BatchNorm'a, o anki batch yerine **kayıtlı (sabit) running istatistikleri** kullanmasını söyler → determinizm sağlar (aynı görüntü her zaman aynı sonucu verir, batch kompozisyonundan bağımsız).
- `torch.no_grad()`: Gradyan hesaplama grafiğini (autograd graph) hiç kurmaz → bellek ve hız kazancı. Bu bir **davranış** değişikliği değil, bir **hesaplama optimizasyonu**dur.

> 🎯 **Klasik tuzak sorusu:** "`eval()` çağırmayı unutursan ama `no_grad()` kullanırsan ne olur?" → Gradyan hesaplanmaz (bellek tasarrufu olur) ama BatchNorm hâlâ batch istatistiklerini kullanır → sonuçlar **deterministik olmaz**, farklı batch'lerde farklı çıktı üretebilir. İkisi ayrı problemleri çözer, biri diğerinin yerine geçmez.

### 4.6 Checkpoint içeriği ve sınırı

```python
{"epoch": epoch, "model_state": model.state_dict(),
 "arcface_state": criterion.state_dict(), "val_acc": val_acc}
```

Bu, **inference için yeterli** ama **eğitimi kaldığı yerden aynen devam ettirmek için yetersiz** — çünkü optimizer state (momentum buffer'ları), scheduler state, RNG (random number generator) durumu ve tam konfigürasyon eksik. *(Düzeltildi: `torch.save` içeriğine `optimizer_state`, `scheduler_state` ve `rng_state` eklendi.)*

### 4.7 Reproducibility eksikliği

Proje hiçbir seed (Python, NumPy, PyTorch, CUDA) sabitlemiyordu. Bilimsel/production bir deneyde şunlar loglanmalı: kaynak commit, kesin bağımlılık versiyonları, CLI konfigürasyonu, seed/determinism ayarları, donanım/sürücü versiyonu, veri versiyonu, model/optimizer/scheduler durumları. *(Düzeltildi: `train.py` içerisine `set_seed(42)` eklenerek Python, NumPy, PyTorch CPU/CUDA ve cuDNN determinizmi sağlandı.)*

---

## BÖLÜM 5 — Validation ve Metrik-Öğrenme Değerlendirmesi

Mevcut validation, sadece ortalama loss ve proxy-classification accuracy ölçüyor:

```python
total_loss += loss.item() * labels.size(0)     # batch'i örnek sayısıyla ağırlıklandır
mean_loss = total_loss / total_samples
```

**Neden `labels.size(0)` ile çarpılıyor?** Son batch'in boyutu diğerlerinden farklı olabilir (örn. 256 yerine 100 örnek). Ağırlıklandırma yapılmazsa, küçük batch'lerin ortalaması büyük batch'lerle eşit sayılır → yanlış genel ortalama.

**Bu neden yetersiz?** Bu bir *metrik öğrenme* (metric learning) sistemi; asıl kullanım senaryosu retrieval/verification'dır, sınıflandırma değil. Eksik metrikler: Recall@K, precision@K / mAP, verification ROC-AUC, sabit false-accept-rate'te true-accept-rate, ayrı bir gallery ile nearest-neighbor accuracy.

---

## BÖLÜM 6 — ONNX Export

### 6.1 ONNX'in rolü

PyTorch: model **tanımlar ve eğitir**. ONNX: bu modelin **hesaplama grafiğini** (operatörler + tensor'lar), framework'ten bağımsız, taşınabilir bir dosyaya **serileştirir**. ONNX, bir **format**tır; onu çalıştıran motor (ONNX Runtime, TensorRT, Triton backend) ayrı bir şeydir. Analoji değil, doğrudan tanım: ONNX = "ne hesaplanacağının" tarifi; ONNX Runtime = "bu tarifi çalıştıran" yorumlayıcı.

### 6.2 `eval()` ve dummy input neden gerekli?

```python
model.eval()
dummy = torch.randn(1, 3, 32, 32)
```

Exporter, modeli bu örnek girdiyle **çalıştırarak** (trace ederek) hangi operasyonların hangi sırayla yürüdüğünü kaydeder. Burada **değerler önemsiz** (rastgele olabilir), önemli olan **shape ve dtype**'ın doğru olması. `eval()` şart, çünkü aksi halde BatchNorm training davranışıyla (batch istatistiği) export edilir — bu, deployment'ta yanlış/deterministik-olmayan sonuçlara yol açar.

### 6.3 Dynamic batch axis

```python
dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}}
```

Sadece **0. boyut** (batch) dinamik; kanal sayısı (3) ve uzamsal boyutlar (32×32) **sabit** kalır. Bu, Triton'un farklı sayıda isteği tek bir batch'te birleştirebilmesi için zorunludur (Bölüm 9.4'e bakınız).

### 6.4 Opset

Opset, ONNX operatör sözleşmesinin (contract) versiyonudur — "hangi operatörler hangi anlamda kullanılabilir" sorusunun cevabı. Daha yeni opset, daha yeni operatör semantiklerini destekler ama **hedef runtime** (ONNX Runtime/Triton versiyonu) bu opset'i desteklemek zorundadır. Uyumsuzluk, export başarılı olsa bile çalışma zamanında hata verebilir.

### 6.5 Export doğrulama adımları (mülakatta "sen ne eklerdin?" sorusuna cevap)

1. `onnx.checker.check_model` ile graph'ın yapısal geçerliliğini doğrula.
2. Aynı girdiyi hem PyTorch hem ONNX Runtime'da çalıştır, çıktıları karşılaştır.
3. Shape ve sayısal hata (numerical error) değerlerini uygun tolerans ile kıyasla.
4. Batch size 1, tercih edilen bir değer, ve yapılandırılmış maksimum değeri test et.
5. Rastgele tensor değil, **gerçek/temsili veri** kullan.

*(Düzeltildi: `scripts/convert_to_onnx.py` içerisine `onnx.checker` yapısal kontrolü ve PyTorch ↔ ONNX Runtime sayısal eşdeğerlik `assert_allclose` testi eklendi.)*

> 🎯 **Kritik ayrım:** "Export'un hatasız tamamlanması ne kanıtlar, ne kanıtlamaz?" → Sadece **serileştirmenin başarılı olduğunu** kanıtlar; PyTorch ve ONNX çıktılarının **sayısal olarak eşdeğer** olduğunu kanıtlamaz. Bu ikisi farklı iddialardır.

---

## BÖLÜM 7 — Quantization ve LiteRT (Edge Optimizasyonu)

### 7.1 FP32 vs INT8 — somut sayısal örnek

- FP32: her sayı için 32 bit (4 byte).
- INT8: her sayı için 8 bit (1 byte).
- Teorik oran: **4×** küçülme (sadece ağırlık depolama için; metadata ve quantize edilmeyen operatörler bu oranı biraz azaltır).

Proje raporu: FP32 = 45.8 MB, INT8 = 11.5 MB → oran ≈ `45.8 / 11.5 ≈ 3.98×` — teorik beklentiyle (4×) tutarlı.

### 7.2 Quantization formülü

```
gerçek_değer ≈ scale × (tam_sayı_değeri - zero_point)
```

Somut örnek: `scale = 0.1`, `zero_point = 0`, `tam_sayı_değeri = 45` → `gerçek_değer ≈ 0.1 × 45 = 4.5`.

Yani sürekli (continuous) FP32 değerleri, sabit bir aralıkta (`scale`) örneklenmiş tam sayılara **yaklaşık olarak** eşleniyor. "Yaklaşık" kelimesi kilit — bu bir bilgi kaybıdır (lossy), bu yüzden accuracy her zaman yeniden ölçülmelidir.

### 7.3 Bu projedeki dynamic quantization — hız her zaman artmaz!

```python
quantize_dynamic(..., weight_type=QuantType.QInt8)
```

Ağırlıklar **önceden** (offline) quantize edilir; ama aktivasyon (activation) quantization parametreleri **çalışma zamanında** (runtime) hesaplanır. Bu, kalibrasyon verisi gerektirmez ama runtime'da ek dönüştürme (dequantize/requantize) maliyeti getirir.

**Projenin raporladığı gerçek sayılar:** FP32 = 0.49 ms, INT8 = 1.61 ms → **INT8 bu ortamda daha yavaş!** Bunun nedeni: CPU'da verimli INT8 kernel'i olmayan operatörler için runtime dönüştürme overhead'i, boyut kazancını gölgeleyebilir.

> 🎯 **En sık yapılan hata (ve mülakatta yakalanan nokta):** "INT8 her zaman daha hızlıdır" demek **yanlıştır**. Doğru ifade: "Bu ortamda daha küçük ama daha yavaş." Hız; operatör desteği, execution provider, donanım (CPU/GPU) instruction seti, batch size ve threading'e bağlıdır.

### 7.4 Dynamic vs Static Quantization

| | Dynamic | Static |
|---|---|---|
| Kalibrasyon verisi | Gerekmez | Gerekir (representative dataset) |
| Kurulum | Basit, hızlı | Daha fazla hazırlık |
| CNN'lerde verimlilik | Genelde daha düşük | Genelde daha yüksek (konvolüsyon ağırlıklı ağlarda) |

**Quantization-Aware Training (QAT):** Eğitim sırasında quantization'ı simüle eder; post-training quantization çok fazla accuracy kaybettirdiğinde, kaybı geri kazanmak için kullanılır.

### 7.5 LiteRT/TFLite

Mobil/gömülü (embedded) runtime'lar için tasarlanmıştır. Conversion script: önce ONNX'i doğrular → `onnx2tf` çağırır → çıktıyı bulur/taşır. **Kritik nokta:** Conversion'ın başarıyla tamamlanması, sayısal eşdeğerliğin (numerical parity) kanıtı **değildir** — gerçek cihazda test edilmedi.

---

## BÖLÜM 8 — Benchmarking Metodolojisi

### 8.1 Temel tanımlar

- **Latency:** Tek bir isteğin tamamlanma süresi.
- **Throughput:** Birim zamanda tamamlanan istek sayısı.
- **Warm-up:** Kernel/cache/bellek başlatma maliyetlerini ölçüm dışı bırakmak için yapılan, zamanlanmayan ön çalıştırma.

### 8.2 P50/P95/P99 — somut sayısal örnek

100 isteklik bir gecikme (latency) listesi düşün, küçükten büyüğe sıralı:
- **P50 (medyan):** Listenin tam ortasındaki değer — isteklerin yarısı bundan hızlı, yarısı yavaş.
- **P95:** İsteklerin %95'i bu değerden hızlı; sadece en yavaş %5'i bunu aşıyor.
- **P99:** En yavaş %1'lik "kuyruk" (tail) davranışını gösterir.

Projenin raporu: **P50 = 7.02 ms, P95 = 287 ms, P99 = 798 ms.** P50 ile P99 arasındaki fark **çok büyük** (≈114×) — bu, "uzun kuyruklu" (long-tail) bir dağılım olduğunu gösterir. Yani çoğu istek hızlı ama azınlık istekler ciddi şekilde yavaş; **ortalama** bunu gizler, bu yüzden P95/P99 raporlanmadan "sistem hızlı" demek yanıltıcıdır.

### 8.3 Bu projedeki benchmark'ın zayıflıkları (kritik — mülakatta "eksik ne?" sorusuna hazır cevap)

- Tek warm-up yeterli olmayabilir.
- Sadece ortalama (mean) raporlanıyor, varyans/percentile yok.
- Normalize edilmiş gerçek veri yerine `[0,1)` rastgele veri kullanılıyor.
- Hız karşılaştırmasından önce **çıktı doğruluğu** kontrol edilmiyor (yani "hızlı ama yanlış" olabilir, bunu ölçmüyoruz).
- Sadece batch size 1 test ediliyor.
- Sonuç mesajı "INT8, X× daha hızlı" diyor — ama INT8 daha yavaşsa oran 1'in altına düşer ve ifade yanıltıcı olur.

*(Düzeltildi: `scripts/benchmark.py` script'ine 10 adımlık warm-up, P50/P95/P99 persentil ölçümleri ve ortama göre dinamik hız/yavaşlık metni eklendi.)*

---

## BÖLÜM 9 — Triton Inference Server

### 9.1 Model Repository yapısı

```
triton_repo/
└── arcface_model/          ← model adı
    ├── config.pbtxt        ← davranış tanımı
    └── 1/                  ← versiyon numarası (klasör adı!)
        └── model.onnx
```

Versiyonlama bir dosya adı değil, bir **klasör adıdır** — Triton bu sayede aynı anda birden fazla model versiyonunu barındırıp yönetebilir.

### 9.2 `config.pbtxt` — kritik alan: `max_batch_size` vs `dims`

```protobuf
name: "arcface_model"
backend: "onnxruntime"
max_batch_size: 128
```

`max_batch_size` **pozitif** olduğunda, Triton'a "girdilerin önünde örtük (implicit) bir batch boyutu var" der. Bu yüzden config içindeki `dims: [3, 32, 32]` ve `dims: [512]` **batch boyutunu içermez** — Triton onu otomatik ekler. Bu, ONNX export'taki `dynamic_axes` ile **doğrudan bağlantılıdır**: ONNX graph batch boyutunu dinamik kabul etmezse, Triton bunu birleştiremez.

### 9.3 Dynamic Batching — mantık

Triton, aynı anda/örtüşen (overlapping) zamanlarda gelen birden fazla isteği, kısa bir süre bekleterek **tek bir GPU çalıştırmasında** birleştirir. Trade-off matematiksel olarak nettir:

```
biraz bekle  → daha büyük batch  → GPU başına daha verimli kullanım (yüksek throughput)
çok bekle    → daha yüksek istek gecikmesi → kötü tail latency (P95/P99 kötüleşir)
```

**Kritik gereksinim — mülakatın en sık sorduğu tuzak:** Dynamic batching yalnızca istekler **zaman içinde örtüştüğünde** işe yarar. Eğer client **sıralı (sequential)** çalışıyorsa — yani bir önceki yanıtı almadan sonrakini göndermiyorsa — hiçbir zaman kuyruk oluşmaz, dolayısıyla birleştirilecek bir şey olmaz. Bu proje, tam olarak bunu göstermek için 32 thread'li **eşzamanlı (concurrent)** bir client kullanıyor.

Config'deki `dynamic_batching { max_queue_delay_microseconds: 2000 }` → Triton, uyumlu isteklerin gelmesini en fazla **2 milisaniye** bekler.

### 9.4 `instance_group` (birden fazla kopya)

`count: 4` → aynı modelin GPU 0 üzerinde **4 bağımsız kopyası** (instance) yüklenir; bunlar paralel çalışabilir ve kuyruklanmayı azaltabilir. Bedeli: 4× GPU belleği, ve compute/bellek bant genişliği/CPU thread için rekabet (contention). "4" evrensel bir optimum değil, bu projede seçilmiş bir **buluşsal (heuristic)** değerdir — gerçek optimum, ölçüm gerektirir.

### 9.5 Auto-config heuristic'inin sınırlamaları (dikkatli mülakat sorusu)

Generator şunları varsayar: Triton için 800 MB ayır, instance başına 650 MB varsay, kalan VRAM'in %70'ini kullan, instance sayısını 4 ile sınırla.

**Sorun:** `nvidia-smi`'den **boş (free)** VRAM'i sorguluyor ama hesaplamada **toplam (total)** VRAM kullanıyor — yani GPU üzerinde başka bir process zaten bellek kullanıyorsa, bu heuristic bunu **görmezden gelir** ve gerçekte kullanılabilir olandan fazla bellek varsayabilir. Bu, gerçek yükleme sırasında OOM (out-of-memory) hatasına yol açabilir. *(Düzeltildi: `scripts/generate_triton_config.py` içinde kullanılabilir VRAM hesabı `gpu["free_mb"]` baz alınarak güncellendi.)*

---

## BÖLÜM 10 — gRPC Client ve Concurrency

### 10.1 gRPC neden REST/JSON'dan hızlı?

gRPC, verileri **Protocol Buffers (protobuf)** ile **ikili (binary)** formatta serileştirir; JSON gibi metin tabanlı değildir. Sonuç: daha küçük payload, daha hızlı parse. Ayrıca HTTP/2 üzerinde çalışır (multiplexing — tek bağlantı üzerinden birden fazla eşzamanlı istek).

### 10.2 Request oluşturma

```python
image = np.random.rand(1, 3, 32, 32).astype(np.float32)
inp = grpcclient.InferInput("input", image.shape, "FP32")
inp.set_data_from_numpy(image)
out = grpcclient.InferRequestedOutput("embedding")
result = client.infer(model_name="arcface_model", inputs=[inp], outputs=[out])
embedding = result.as_numpy("embedding")
```

`"input"` ve `"embedding"` isimleri **ONNX graph'taki isimlerle birebir eşleşmek zorundadır** (Bölüm 6.3, 9.2). İsim, dtype, rank ve shape'ten herhangi biri uyuşmazsa çalışma zamanı hatası alırsın.

### 10.3 Kritik preprocessing sorunu

Client, `[0, 1)` aralığında **rastgele** veri gönderiyor, CIFAR-10 normalizasyonu **uygulamıyor**. Bu, bağlantı/isim/shape/dtype doğrulaması için (smoke test) yeterlidir ama **model kalitesi** testi için geçersizdir — çünkü girdi dağılımı, eğitimde görülen dağılımla uyuşmuyor.

### 10.4 Sequential vs Concurrent — neden 32 thread?

Sequential (sıralı) client, önceki yanıtı beklemeden sonraki isteği göndermez → istekler asla örtüşmez → dynamic batching test edilemez (Bölüm 9.3). Bu yüzden concurrent script **32 thread** açıyor, her biri **kendi client'ıyla** 500 istek gönderiyor (toplam 16.000).

**Neden her thread kendi client'ını kullanıyor?** Client state paylaşımından kaynaklanabilecek belirsizlikleri (race condition benzeri sorunları) ortadan kaldırmak için.

**Neden threading (multiprocessing değil) yeterli?** Çünkü iş, **I/O-bound**'dur (ağ isteği gönderip yanıt beklemek) — zamanın büyük kısmı CPU hesaplaması değil, ağdan yanıt bekleyerek geçer. Python'un GIL'i (Global Interpreter Lock), bekleme sırasında serbest bırakılır, dolayısıyla thread'ler CPU-bound işler kadar GIL tarafından kısıtlanmaz.

### 10.5 Little's Law — somut sayısal örnek

```
sistemdeki istek sayısı ≈ throughput × ortalama sistemde kalma süresi
L = λ × W
```

Projenin sayıları: throughput `λ ≈ 622 istek/s`. Ortalama gecikmeyi (P50 değil, **ortalama**) `W ≈ 50 ms = 0.05 s` varsayarsak:

```
L = 622 × 0.05 ≈ 31.1
```

Yani ortalama olarak sistemde her an ≈31 istek işleniyor/bekliyor olmalı. **Kritik uyarı:** Bu formülde **P50 değil ortalama (mean) latency** kullanılmalıdır — P50 medyandır, ortalama değildir; uzun kuyruklu dağılımlarda ikisi çok farklı olabilir.

---

## BÖLÜM 11 — Raporlanan Sonuçları Yorumlama

| Aşama | Sonuç | Yorum |
|---|---|---|
| Training | %87.35 val accuracy, ~4.5 s/epoch (RTX 4070 Ti SUPER) | Bu **proxy-classification** accuracy'dir, retrieval/verification metriği değil |
| FP32 ONNX (CPU) | 45.8 MB, 0.49 ms | Bu ortamda daha hızlı |
| INT8 ONNX (CPU) | 11.5 MB, 1.61 ms | ~4× küçük ama bu ortamda daha yavaş |
| Triton (32 thread, 16.000 istek) | ~622 req/s, P50 7.02 ms, P95 287 ms, P99 798 ms | Uzun kuyruklu dağılım — araştırılmalı |

**Kritik uyarı:** Local CPU benchmark ile Triton GPU sonucunu **aynı deney gibi karşılaştırma** — farklı runtime, donanım yolu, scheduling, concurrency ve ölçüm sınırları kullanıyorlar.

---

## BÖLÜM 12 — Production-Readiness: Öncelik Sıralı İyileştirme Listesi

1. **ArcFace margin birimini düzelt** (derece/radyan) → yeniden eğit.
2. Deterministic seed ekle, tam deney konfigürasyonunu logla.
3. Config hesaplama, transform ve tensor contract'ları için unit test yaz.
4. PyTorch ↔ ONNX ↔ quantized çıktı arasında sayısal eşdeğerlik (parity) testleri ekle.
5. Sadece proxy-accuracy değil, retrieval/verification metrikleri ölç.
6. Her gerçek client'ta validation preprocessing'i birebir tekrar kullan.
7. Deployed embedding'leri L2-normalize et (veya bunun consumer sorumluluğunda olduğunu belgele).
8. Timeout, structured error, health check, request correlation ekle.
9. Queue time, compute time, batch size, GPU utilization, failure/version metriklerini export et (gözlemlenebilirlik).
10. Triton'u temsili trafik ve açık latency/throughput hedefiyle tune et.
11. Model/veri lineage, artifact bütünlük kontrolü, kademeli rollout (canary) ve rollback ekle.
12. Kontrollü varış hızıyla (arrival rate) load test yap; güven aralığı ve dağılım raporla.

**Security/Operations başlıkları (mülakatçının sorabileceği ek liste):** gRPC için TLS+auth, request boyutu/input validation, rate limiting/backpressure, container/artifact tarama, drift monitoring, embedding'lerin gizlilik (privacy) riski, canary+rollback, SLI'lar (availability, error rate, P95/P99, saturation).

---

## BÖLÜM 13 — Mülakat Anlatım Şablonu (2 Dakika)

Bir seçim/tasarım kararı sorulduğunda **bu sırayı** izle (kanıta dayalı, yapılandırılmış cevap — rastgele konuşmaktan çok daha ikna edicidir):

1. **Gereksinimi belirt** ("Hem edge hem server'da çalışması gerekiyordu")
2. **Seçilen mekanizmayı açıkla** ("Bu yüzden tek backbone'u ONNX'e export edip iki yola ayırdım")
3. **Ölçülen sonucu söyle** (somut sayı ver: "45.8 MB → 11.5 MB")
4. **Trade-off'u adlandır** ("ama bu CPU ortamında INT8 daha yavaştı")
5. **Nasıl doğrulayacağını/geliştireceğini anlat** ("bir sonraki adım static quantization ile karşılaştırmak olurdu")

Bu 5 adım, projenin **her** kararı için (ArcFace seçimi, ONNX kullanımı, Triton batching, thread sayısı) tekrar kullanılabilir bir şablondur.

---

## BÖLÜM 14 — Hızlı Tekrar Kartı (Sınav Öncesi Son Bakış)

- **Dataset:** CIFAR-10, 50.000 train / 10.000 test, 10 sınıf, 32×32 RGB.
- **Backbone:** Pretrained ResNet-18 + 512-boyutlu Linear+BatchNorm embedding head.
- **Training:** 30 epoch, batch 256, SGD (LR 0.1, momentum 0.9, weight decay 5e-4, Nesterov).
- **Sonuç:** %87.35 val accuracy (proxy-classification), ~4.5 s/epoch.
- **Scheduler:** Cosine annealing → 1e-5.
- **ArcFace:** scale 64; margin **0.5 derece olarak yorumlanıyor** (niyet: 0.5 radyan ≈ 28.6°) — **bilinen hata**.
- **Export:** ONNX opset 17, input adı `input`, output adı `embedding`, sadece batch ekseni dinamik.
- **Quantization:** Dynamic INT8, 45.8 MB → 11.5 MB (~4× küçük, bu CPU'da daha yavaş: 0.49 ms → 1.61 ms).
- **Triton:** ONNX Runtime backend, max_batch 128, preferred [32,64,128], queue delay 2000 µs, 4 GPU instance.
- **Load test:** 32 thread × 500 istek = 16.000 istek → ~622 req/s, P50 7.02 ms, P95 287 ms, P99 798 ms.
- **En kritik iki mesaj:** (1) Dynamic batching sadece **örtüşen** istekler varsa işe yarar. (2) Her "daha küçük/hızlı" iddiası **ayrı ayrı ölçülmeli**, birbirine genellenmemeli.

---

## BÖLÜM 15 — Aktif Hatırlama Testi (Retrieval Practice)

Bilimsel literatür, bir bilgiyi tekrar okumak yerine **hatırlamaya çalışmanın** (active recall) kalıcı öğrenmeyi anlamlı şekilde artırdığını gösteriyor. Aşağıdaki soruları **cevaba bakmadan önce zihninde/sesli cevapla**, sonra kontrol et.

1. Model neden 10 sınıf olasılığı değil 512 sayı üretiyor?
2. Cosine similarity ne ölçer, büyüklük mü yön mü?
3. Projedeki ArcFace margin hatası tam olarak nedir ve neden checkpoint'i etkiler?
4. `criterion.parameters()` optimizer'a eklenmezse ne olur?
5. `model.eval()` ile `torch.no_grad()` arasındaki fark nedir — biri diğerinin yerini tutar mı?
6. ONNX'te sadece batch ekseni neden dinamik, diğerleri neden değil?
7. INT8 model her zaman daha mı hızlıdır? Bu projede ne gözlemlendi?
8. Dynamic batching'in çalışması için hangi koşul (istekler arasında) gereklidir?
9. `max_batch_size` ile config'teki `dims` arasındaki ilişki nedir?
10. P50 ile P99 arasındaki büyük fark neyi gösterir?
11. Concurrent load test'te neden 32 ayrı client kullanılıyor, tek client değil?
12. Little's Law formülünde P50 mi ortalama mı kullanılmalı, neden?
13. Auto-generated Triton config'in en büyük metodolojik zayıflığı nedir?
14. Export'un hatasız tamamlanması tam olarak neyi kanıtlar, neyi kanıtlamaz?

**Kendini kontrol etmek için:** Yukarıdaki her soru, STUDY_TR.md Bölüm 19'daki (Q1-Q20) sorularla ve bu rehberin ilgili bölümleriyle birebir eşleşir. Cevap veremediğin bir soru varsa, o bölümü tekrar oku — ama bu sefer **kendi cümlelerinle özetlemeyi dene**, metni tekrar okumakla yetinme.

---

## BÖLÜM 16 — Ekstra Zorluk Soruları (Belgede Olmayan, Muhtemel Ek Sorular)

**S: Neden `nn.BatchNorm1d` (2D değil) embedding head'de kullanılıyor?**
C: Embedding head'in çıktısı `(B, 512)` şeklinde — 2 boyutlu bir tensor (batch + özellik). `BatchNorm1d`, tam olarak bu şekli normalize etmek için tasarlanmıştır. `BatchNorm2d`, `(B, C, H, W)` gibi uzamsal boyutu olan konvolüsyon çıktıları içindir.

**S: ONNX Runtime'da "Execution Provider" nedir?**
C: Bir operatörün fiilen hangi donanım/kütüphane ile çalıştırılacağını belirleyen backend'dir (örn. `CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorRTExecutionProvider`). Aynı ONNX graph'ı, farklı execution provider'larla çalıştırınca farklı hız (ve bazen farklı sayısal hassasiyet) elde edilebilir.

**S: `dynamic_batching` preferred size listesi (`[32, 64, 128]`) garanti mi?**
C: Hayır, bunlar **hedef/ipucu**dur (hint). `max_queue_delay_microseconds` süresi dolduğunda Triton, tercih edilen boyuta ulaşmamış olsa bile eldeki isteklerle çalıştırabilir.

**S: Neden `subprocess` ile `nvidia-smi` çağrılıyor, PyTorch API'si yerine?**
C: Config generator'ın amacı toplam sistem VRAM'ini (PyTorch process'i başlamadan/bağımsız olarak) sorgulamaktır; bu bilgi genelde harici komut satırı aracıyla (nvidia-smi) elde edilir, çünkü PyTorch henüz bir CUDA context açmamış olabilir.

**S: Weight decay (L2 regularization) matematiksel olarak ne yapar?**
C: Loss fonksiyonuna `λ × Σ(w²)` terimi ekler (kavramsal olarak). Bu, büyük ağırlıkları cezalandırarak modeli daha "basit" (küçük ağırlıklı) çözümlere yönlendirir — overfitting riskini azaltır.

**S: Nesterov momentum, standart momentumdan matematiksel olarak nasıl farklı?**
C: Standart momentum, gradyanı **şu anki** konumda hesaplar. Nesterov, önce momentum yönünde "ileri bakılan" (look-ahead) bir konum tahmin eder, gradyanı **o noktada** hesaplar — bu, güncellemeyi daha isabetli hale getirir çünkü momentumun "aşırı gitme" (overshoot) riskini kısmen öngörür.

**S: `pin_memory=True` ne işe yarar, bu projede gerçekten fayda sağlıyor mu?**
C: CPU belleğini "sayfalanamaz" (page-locked) hale getirerek CPU→GPU veri transferini hızlandırır. Sadece GPU eğitiminde anlamlı fayda sağlar; CPU-only eğitimde etkisi ihmal edilebilir düzeydedir — projede bu doğru şekilde not edilmiş.

**S: gRPC neden HTTP/REST'ten daha uygun burada?**
C: Yüksek frekanslı, düşük gecikmeli, tekrarlayan-şemalı (repetitive schema) isteklerde: (1) binary protobuf serileştirme JSON'dan daha küçük/hızlıdır, (2) HTTP/2 multiplexing aynı bağlantı üzerinden çok sayıda eşzamanlı isteğe izin verir, (3) şema (schema) taraflar arasında sabit olduğundan tip güvenliği (type safety) sağlar.

---

### Kapanış Notu

Bu rehberi bir kez okumak yeterli değildir — **Bölüm 15'i** birkaç gün arayla (spaced repetition — bilimsel olarak kalıcılığı artırdığı gösterilmiş bir teknik) tekrar tekrar kendine sormak, mülakat performansını gerçek anlamda artıracak yöntemdir. Bir sonraki adım olarak: (1) her bölümün sonundaki soruyu yazılı olarak cevapla, (2) STUDY_TR.md'deki 20 soruyu hiç bakmadan cevaplamayı dene, (3) yalnızca yanlış/eksik cevapladığın bölümleri tekrar çalış.
