# SynapseFly / FlyBrain

> **$FLY** — Bir *Drosophila melanogaster* erkek merkezi sinir sistemi konektomu (MaleCNS v1.0 şekilli) üzerinde
> koşan **spiking (LIF) sinir ağı**; duyuları bir token piyasasına bağlı; bedeni bir **Windows 95 Paint** penceresinde
> resim çizen bir sinek; ruh hali değişince kendi beynini anlatan tweet'ler atan bir ajan.
> Meta-meme ile DeSci'nin kesişimi: ekranda gördüğünüz her şey gerçek nöron gruplarının (LB3b şeker GRN'leri, MN9
> hortum motor nöronu, DNp01 dev lif, LC4/LPLC2 looming dedektörleri, DNa02 dümen, PAM dopamin…) ateşleme
> oranlarından türetilir.

**Bugün, sıfır hesapla, çevrimdışı çalışır**: sentetik konektom + simüle piyasa + kuru-çalışma (dry-run) LLM ve X.
Gerçek MaleCNS verisi, DexScreener, Claude ve X yolları birer ortam değişkeni uzaklıktadır (aşağıya bakın).

---

## İçindekiler

1. [Önemli uyarı: varsayılan beyin sentetiktir](#1-önemli-uyarı-varsayılan-beyin-sentetiktir)
2. [5 dakikada başlangıç](#2-5-dakikada-başlangıç)
3. [Mimari](#3-mimari)
4. [Depo yerleşimi](#4-depo-yerleşimi)
5. [Ortam değişkenleri](#5-ortam-değişkenleri)
6. [Çalıştırma reçeteleri](#6-çalıştırma-reçeteleri)
7. [Veri kaynakları ve lisanslar](#7-veri-kaynakları-ve-lisanslar)
8. [Kuklacılık listesi (puppeteering)](#8-kuklacılık-listesi-puppeteering)
9. [Test, smoke ve selftest](#9-test-smoke-ve-selftest)
10. [LLM ve X modları, maliyet](#10-llm-ve-x-modları-maliyet)
11. [Arayüz ve klavye kısayolları](#11-arayüz-ve-klavye-kısayolları)
12. [Yol haritası](#12-yol-haritası)
13. [Katkıda bulunma](#13-katkıda-bulunma)
14. [Bilimsel referanslar](#14-bilimsel-referanslar)

---

## 1. Önemli uyarı: varsayılan beyin sentetiktir

`FLY_CONNECTOME_SOURCE=synthetic` (varsayılan) ile çalışan beyin **gerçek konektom verisi değildir**. Bu, MaleCNS v1.0
şeklinde üretilmiş bir **"sentetik yapısal vekil"**dir (İngilizce arayüzde: *synthetic structured stand-in,
MaleCNS-shaped*):

* 8 bölgeli taksonomi, gerçek hücre tipi adları ve doğrulanmış sayımlar (RESEARCH §4), 112 projeksiyon kuralı
  (tasarım hedefi; gönderilen üreteç 114 üretir — fazlalık P113/P114 GF dinlenme frenleridir, bkz. §8) (RESEARCH
  §5'teki doğrulanmış yolak ağırlıkları) ve rastgele bir arka plan grafı ile üretilir.
* Sinaps sayıları gerçek veriden **kopyalanmaz**; `calibrated` modda kararlı-durum formülüyle hesaplanır,
  `literature` modda literatür ortalamaları kullanılır.
* Arayüz (`Help ▸ About`), `hello.connectome.note` alanı, LLM'e giden özet ve bu README bunu her yerde açıkça belirtir:
  `"synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data"`.

Gerçek veriye geçmek için §6'daki *Gerçek MaleCNS* veya *neuPrint* reçetelerini izleyin; o zaman `source` alanı `csv` /
`neuprint`, lisans alanı `CC-BY 4.0` ve atıf alanı Berg ve ark. (2026) olur.

---

## 2. 5 dakikada başlangıç

Gereksinimler (bu makinede doğrulandı): Windows 11, Python 3.14 (`py -3`), Node 24 / npm 11, `frontend/node_modules`
kurulu. Hesap, ağ ve GPU **gerekmez**. Tüm komutlar PowerShell içindir ve depo kökünden (`C:\Users\USER\fly`) çalışır.

```powershell
cd C:\Users\USER\fly
Copy-Item .env.example .env                                   # bir kez
Copy-Item frontend\.env.local.example frontend\.env.local     # bir kez
$env:PYTHONUTF8 = "1"

py -3 scripts\selftest.py        # 20k sentetik beyni kurar (~2 s), 5 kapıyı + RTF + hareket testini + bir dry-run tweet'i yazdırır
py -3 backend\run.py             # terminal 1 -> http://127.0.0.1:4000   (ilk açılışta konektom önbelleğe alınır)
```

İkinci terminalde:

```powershell
cd C:\Users\USER\fly\frontend ; npm run dev                  # -> http://localhost:3000
```

Ya da ikisini birden: `powershell -ExecutionPolicy Bypass -File scripts\dev.ps1`

Kontrol listesi (toplam ≈ 5 dk):

1. `curl http://127.0.0.1:4000/api/health` → `"ok":true`, `rtf > 1`.
2. Tarayıcıda Paint penceresi 2 s içinde yürüyen bir sinek ve siyah/yeşil/kırmızı bir iz gösterir; *Oscilloscope*
   penceresinde 8 şerit akar; *Fly Status* `CRUISING` der; ticker `FLY` / `SIM` gösterir.
3. `S` (şeker): sinek durur, hortum uzar, mürekkep turuncu-noktalı olur, `feed_mn` yükselir, başlıkta `FEEDING`.
   `L` (satış duvarı = looming): optik lob şeridinde kırmızı bir rampa, `DNp01` yıldız satırı ateşler, sinek zıplar
   (zikzak iz), `ESCAPE → PANIC/ANXIOUS`. `1..6` piyasa rejimini zorlar, `T` Notepad penceresine dry-run tweet düşürür.
4. `py -3 scripts\demo_market.py` (terminal 3): `PUMP → RUG → CALM → DEAD → CHOP` sırasıyla `EUPHORIA`, `PANIC`,
   `CRUISING`, `SLEEP` ve `out\tweets\` altında iki dry-run tweet üretir.
5. İsteğe bağlı: `py -3 -m pytest backend\tests -q -m "not slow"` (< 25 s) ve `cd frontend; npm run typecheck; npm run lint`.

---

## 3. Mimari

```
                    ┌──────────────────────── backend (Python, :4000) ────────────────────────┐
  DexScreener ──►   │  MarketFeed (thread)  ──► StateBus ◄──  SimulationLoop (thread, 20 Hz)   │
  veya SimulatedMarket                          │            ├─ FeatureExtractor  (piyasa → duyu özellikleri)   │
                    │                           │            ├─ SensoryEncoder    (özellik → Drive listesi)      │
                    │                           │            ├─ LIFEngine         (CSR + exact 2-D LIF, numpy/torch)
                    │                           │            ├─ SpikeMonitor / RateEstimator (raster, Hz)        │
                    │                           │            ├─ MotorDecoder + FlyBody (DN oranları → kinematik, mürekkep)
                    │                           │            └─ MoodMachine (8 ruh hali, zamanlayıcılar)         │
                    │  TweetAgent (1 işçi)  ◄────┘  (ruh hali onaylanınca: özet JSON → Claude/şablon → X/dry-run) │
                    │  FastAPI + uvicorn: /ws (JSON metin çerçeveleri), /api/*                                     │
                    └──────────────────────────────────┬───────────────────────────────────────┘
                                                       │ WebSocket: hello, tick (20 Hz), mood_change, market, tweet, event
                    ┌──────────────────────────────────▼──────────────── frontend (Next 16, :3000) ───┐
                    │ lib/ws.ts → lib/store.ts (değişebilir halka; React state yok)                     │
                    │  Paint penceresi (FlyCanvas: iz + sinek, rAF)  ·  Oscilloscope (SpikeRaster)      │
                    │  Fly Status (MoodPanel + MarketTicker, 4 Hz)   ·  tweets.txt - Notepad            │
                    └────────────────────────────────────────────────────────────────────────────────┘
```

Tasarım kararlarının özeti (tam tablo: `docs/SPEC.md` §0):

| Konu | Karar |
|---|---|
| Nöron modeli | Shiu 2024 LIF; **tam 2-D matris-üstel güncelleme**; Brian2 "unless refractory" semantiği (refrakter süre boyunca `v` ve `g` donar, gelen girdi birikir); reset `v = v_reset, g = 0`. Sabitler RESEARCH §7'de sayısal olarak doğrulandı. |
| Zaman adımı | `FLY_DT_MS=1.0` (refrakter 2 adım, gecikme 2 adım); 0.5 / 0.2 / 0.1 sadakat koşuları için desteklenir. |
| Duyusal sürüş | `Drive` (grup, Hz, recruit, side, episode); Poisson zorlama `g += 68.75 mV` veya `current` modu. |
| Dinlenme aktivitesi | **1–5 Hz zorunlu**: adım başına `g += N(mu·dt, sigma·sqrt(dt))`, `mu 0.5, sigma 3.5` → ≈ 2.3 Hz; "sessiz beyin" kapısı yok. |
| Sentetik graf | MaleCNS şekilli yapısal üreteç; gerçek tip adları + doğrulanmış sayımlar; 8 bölge; 112 projeksiyon (tasarım; gönderilen kod 114, bkz. §8); ağırlıklar `calibrated` (varsayılan) veya `literature`. |
| Doğrulanmış yolak düzeltmeleri | PFL3→DNa02/03/DNb01 **kontralateral**; LC9/LC31a→DNp09; histaminerjik fotoreseptörler **inhibitör**; gerçek SEZ beslenme zinciri (GNG215/232/132/089 → GNG108/120/117/234/DNge062/080 → MN9, GNG042→GNG015 disinhibisyon); DNg02 → uçuş MN'leri. |
| Gap junction | Yama tablosu (DNp01→TTMn, DNp01→PSI, DNp01↔DNp01, LC4/LPLC2→DNp01 dendro-dendritik vekil), `meta.patches_applied` içinde kayıtlı. |
| Garantili hareket | Keşif taban çizgisi (DNg100), OU gezinme, `wander_floor`, EPG proprioseptif geri besleme, duvar çarpması → DNa02. Hepsi `encoder.py`/`decoder.py` içinde yalıtılmış, **kuklacılık** olarak belgelenmiş, sıfıra çekilebilir (§8). |
| Zamanlayıcı | Duvar saati 20 Hz tik, **asla atlanmaz**; tik başına beyin süresi `50 ms × speed`, `speed ∈ [0.25, 1]` hesaba uyum sağlar; `FLY_REALTIME=0` → sabit 50 adım/tik, uyku yok (test/replay). |
| Seyrek düzen | Tek CSR (`indptr int64 + int32 gölge`, `indices int32`, `data float32`); `.npz` önbellek `mmap_mode='r'`; `numpy` (`np.add.at`) ve `torch` (`index_add_`) yayıcılar, `auto` → E > 5M ise torch. |
| Homeostaz | Aktif oran 10 adım boyunca > %5 → `gain *= 0.9` (taban 0.3), `homeostasis` olayı; `calibrate_gain` ikiye bölme (önbellekli). |
| Piyasa | `MarketSource` protokolü; `SimulatedMarket` (5 rejimli Markov GBM + kümelenmiş işlemler + balinalar); `DexScreenerSource` (`/tokens/v1`, 60 s); `MarketFeed` 3 başarısız sorgudan sonra otomatik sim'e döner (`market_source` olayı). |
| Looming | Her satış: 300 ms genişleyen disk rampası `220·amp·(t/300 ms)²` Hz, LC4/LPLC2'nin rastgele %60'ında, tek taraf; tonik bileşen; balina satışı = 300 ms tam alan flaş; yavaş sürekli looming → LC9/LC31a (donma). |
| Ruh hali | 8 durum, öncelik ESCAPE > PANIC > COURTSHIP > EUPHORIA > FEEDING > ANXIOUS > SLEEP > CRUISING; 1.5 s minimum kalış; tweet için 3 s onay; ESCAPE asla tweet atmaz. |
| Ajan | Anthropic çağrısı (`claude-opus-5`, `max_tokens=512`), `output_config` yapılandırılmış çıktı, dry-run şablonları, `validate_tweet` (URL yok, ≤ 280, ≤ 2 hashtag, nöron sözcüğü zorunlu), `data/agent_state.json`, `data/tweets.jsonl`; tweepy `create_tweet` + medya yükleme + salt-metin yedek. |
| Frontend | Mevcut Next 16.3.4 / React 19 / Tailwind 4 iskeleti, **yeni bağımlılık yok**; dört sürüklenebilir Win95 penceresi, prosedürel sinek, mürekkep stilleri, Notepad tweet günlüğü, klavye kısayolları, `document.title = ruh hali`. |
| Tuval | 800 × 500 mantıksal px, y aşağı; `FLY_WALLS=bounce` (varsayılan) veya `wrap`. |
| Köken bilgisi | `hello.connectome.source ∈ {synthetic, csv, neuprint}` + `license` + `citation`; her yerde "sentetik yapısal vekil" etiketi. |
| Tekrar (replay) | `SessionLog` JSONL (her tikin girdileri) ve `scripts/replay.py` → bit-bit aynı spike sayıları. |

**Rastgelelik**: tek `np.random.SeedSequence(FLY_SEED)` sabit sırayla dallanır (`[0] konektom, [1] motor, [2] piyasa
sim, [3] gezinme, [4] raster örnekleme, [5] ajan şablonları, [6] encoder episodları`). `FLY_MARKET=sim` + numpy arka
ucu ile koşu **deterministiktir**.

**İş parçacıkları**: `SimulationLoop` (motor/encoder/decoder/ruh hali/ajan tetikleyici), `MarketFeed` (HTTP), uvicorn
asyncio döngüsü (WebSocket), `TweetAgent` (1 işçili havuz). Çapraz geçiş yalnızca `StateBus` ve iş-parçacığı-güvenli
kuyruklarla; **sim iş parçacığı asla G/Ç'de bloklanmaz**; hiçbir dış hata simülasyonu durdurmaz (`event` çerçevesi +
log satırı üretir).

---

## 4. Depo yerleşimi

```
fly\
  README.md                 bu dosya (Türkçe)          docs\SPEC.md      normatif sözleşme (İngilizce)
  docs\RESEARCH.md          doğrulanmış sayılar        docs\NOTICE.md    veri kökeni, atıflar, lisans metinleri
  .env.example              tüm FLY_* değişkenleri     data\README.md    data\ dizin haritası (data\ gitignore'da)
  backend\run.py            py -3 backend\run.py       backend\flybrain\ config, connectome\, snn\, market\, encoder,
                                                       decoder, mood, agent\, server\
  backend\tests\            pytest (çevrimdışı)        scripts\          dev.ps1 smoke selftest demo_market replay
                                                                          bench prepare_malecns fetch_neuprint export_synthetic
  frontend\                 Next 16 (app\, components\, lib\)
  out\tweets\               dry-run tweet artefaktları (<id>.txt/.json/.png; gitignore'da)
```

---

## 5. Ortam değişkenleri

`load_settings()` önce `<repo>/.env` dosyasını, sonra `os.environ`'ı okur (environ kazanır). Boolean değerler
`1/0/true/false/yes/no` kabul eder. Tam liste ve doğrulama kuralları `docs/SPEC.md` §b'dedir.

| Değişken | Varsayılan | Anlamı |
|---|---|---|
| `FLY_CONNECTOME_SOURCE` | `synthetic` | `synthetic` (üreteç) · `csv` (`FLY_CONNECTOME_DIR/{neurons,connections}.csv(.gz)`, neuPrint veya Codex şeması otomatik) · `neuprint` (`data/connectome/neuprint`, `scripts/fetch_neuprint.py` ile indirilir; çalışma anında asla ağ yok) |
| `FLY_CONNECTOME_DIR` | `data/connectome/malecns` | `csv` kaynağı için dizin |
| `FLY_CONNECTOME_NAME` | `""` | önbellek / görüntü adı |
| `FLY_N_NEURONS` | `20000` | sentetik boyut (4000..200000); gerçek veride `core` alt kümesi tavanı |
| `FLY_MEAN_OUTDEG` | `25` | sentetik arka plan çıkış derecesi |
| `FLY_SYNTH_WEIGHTS` | `calibrated` | `calibrated` · `literature` |
| `FLY_SUBSET` | `core` | gerçek veri: `core` (tüm işlevsel gruplar + rastgele dolgu) · `all` |
| `FLY_MIN_WEIGHT` | `3` | gerçek veri: bu sinaps sayısının altındaki kenarları at |
| `FLY_DT_MS` | `1.0` | LIF adımı: `1.0 · 0.5 · 0.2 · 0.1` |
| `FLY_SEED` | `1337` | kök tohum |
| `FLY_BACKEND` | `numpy` | `numpy · torch · auto` (`auto`: E > 5M ve torch varsa torch) |
| `FLY_GAIN` | `auto` | sinaptik kazanç; `auto` = kalibre sentetikte 1.0, aksi halde `data/cache/<key>.calib.json` veya ilk açılışta `calibrate_gain` |
| `FLY_NOISE_MU` / `FLY_NOISE_SIGMA` | `0.5` / `3.5` | arka plan gürültüsü (mV/ms, mV/√ms) → ≈ 2.3 Hz dinlenme |
| `FLY_DRIVE_MODE` | `poisson` | `poisson · current` |
| `FLY_TICK_HZ` | `20` | duvar tik hızı (5..50) |
| `FLY_REALTIME` | `1` | 1 = faz kilitli tikler + uyarlanır `speed`; 0 = sabit adım, uyku yok |
| `FLY_MARKET` | `sim` | `sim · dexscreener` (`dexscreener` için `FLY_TOKEN_ADDRESS` gerekir; 3 hata sonrası sim'e döner) |
| `FLY_TOKEN_ADDRESS` / `FLY_CHAIN` | `""` / `solana` | token adresi, DexScreener `chainId` |
| `FLY_DEX_POLL_S` | `60` | DexScreener sorgu aralığı (≥ 15; uç önbellek ≈ 60 s) |
| `FLY_SIM_REGIME_S` | `90` | simüle piyasanın ortalama rejim süresi |
| `FLY_LLM` | `dryrun` | `dryrun · anthropic` |
| `ANTHROPIC_API_KEY` | — | `FLY_LLM=anthropic` iken SDK okur |
| `FLY_LLM_MODEL` | `claude-opus-5` | model kimliği (tarih soneki eklemeyin) |
| `FLY_LLM_JSON` | `1` | `output_config` yapılandırılmış çıktı; hata olursa düz metin |
| `FLY_LLM_FALLBACKS` | `0` | sunucu tarafı reddetme yedekleri (anthropic ≥ 1.x); aksi halde yok sayılır |
| `FLY_TWEET_LANG` | `en` | `en · tr` |
| `FLY_X` | `dryrun` | `dryrun · post` |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | — | OAuth1 kullanıcı bağlamı (Read+Write uygulama) |
| `FLY_TWEET_COOLDOWN_S` / `FLY_TWEET_REASON_COOLDOWN_S` | `900` / `2700` | genel / sebep başına bekleme |
| `FLY_TWEETS_PER_DAY` | `12` | günlük sert tavan (UTC) |
| `FLY_PORT` / `FLY_HOST` | `4000` / `127.0.0.1` | HTTP/WS |
| `FLY_CORS_ORIGINS` | `http://localhost:3000` | virgülle ayrılmış izin listesi |
| `FLY_CANVAS_W` / `FLY_CANVAS_H` | `800` / `500` | tuval sınırları |
| `FLY_WALLS` | `bounce` | `bounce · wrap` |
| `FLY_RASTER_PER_REGION` / `FLY_RASTER_CAP` | `48` / `2000` | bölge başına raster satırı / tik başına en fazla olay |
| `FLY_MOOD_FEEDBACK` | `1` | ruh hali → beyin geri besleme sürüşleri (**kuklacılık**, 0 kapatır) |
| `FLY_EXPLORE_BASELINE` | `0.25` | keşif sürüşü tabanı (0 kapatır; `wander_floor` da kapanır) |
| `FLY_WANDER_SIGMA` | `0.6` | OU gezinme gücü (0 kapatır) |
| `FLY_EASTER_EGGS` | `1` | 69/420 / 04:20 kur yapma tetikleyicileri |
| `FLY_SESSION_LOG` | `1` | `data/sessions/<run_id>.jsonl` yaz |
| `FLY_REPLAY` | `""` | canlı girdi yerine bir oturum günlüğünü tekrar oynat |
| `FLY_DATA_DIR` / `FLY_OUT_DIR` | `<repo>/data` / `<repo>/out` | durum / artefaktlar |
| `FLY_LOG_LEVEL` | `INFO` | günlük seviyesi |
| `NEUPRINT_APPLICATION_CREDENTIALS` | — | yalnızca `scripts/fetch_neuprint.py` (boşsa anonim HTTP) |
| Frontend `NEXT_PUBLIC_WS_URL` | `ws://localhost:4000/ws` | WebSocket adresi (`frontend/.env.local`) |
| Frontend `NEXT_PUBLIC_API_URL` | `http://localhost:4000` | REST tabanı |

Türetilenler: `tick_ms = 1000/tick_hz`, `steps_per_tick = round(tick_ms/dt_ms)`, `run_id = sha1(kaynak|n|tohum|dt|ağırlık|piyasa)[:8]`.

---

## 6. Çalıştırma reçeteleri

### Sıfır hesap (varsayılan)

```powershell
cd C:\Users\USER\fly
$env:PYTHONUTF8 = "1"
py -3 backend\run.py                              # -> http://127.0.0.1:4000   (ilk açılış konektomu kurar + önbelleğe alır, ~2 s)
cd frontend ; npm run dev                         # -> http://localhost:3000  (ikinci terminal)
```

CLI geçersiz kılmaları env'i ezer: `py -3 backend\run.py --n 8000 --seed 7 --port 4001 --no-realtime`.
Zayıf bir dizüstünde `FLY_N_NEURONS=8000` yeterlidir.

### Gerçek MaleCNS v1.0 (CC-BY 4.0; GCS üzerinden hesapsız; ≈ 570 MB indirme, 2–5 dk hazırlık)

```powershell
py -3 scripts\prepare_malecns.py --out data\connectome\malecns --min-weight 1    # pyarrow gerekir (kurulu)
$env:FLY_CONNECTOME_SOURCE = "csv" ; $env:FLY_CONNECTOME_DIR = "data\connectome\malecns"
$env:FLY_SUBSET = "core" ; $env:FLY_N_NEURONS = "20000"
py -3 scripts\selftest.py --source csv --dir data\connectome\malecns    # ilk açılışta calibrate_gain (önbelleğe alınır), kapıları yazdırır
py -3 backend\run.py
```

Tam graf (166.7k nöron / ≈ 25.6M kenar, memmap, torch otomatik): `$env:FLY_SUBSET="all"; $env:FLY_BACKEND="auto"`
(bu CPU'da RTF 0.3–0.9x; arayüz `brain 0.5x` gibi gösterir).

İndirilen dosyalar: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` altındaki
`body-annotations-…feather` (14.5 MB), `body-neurotransmitters-…feather` (43.3 MB) ve
`connectome-weights-…-traced-only.feather` (508 MB). Betik bunları `neurons.csv` (neuPrint şeması) ve
`connections.csv` (`bodyId_pre,bodyId_post,weight`) olarak yazar. **Bu depo hiçbir MaleCNS verisi dağıtmaz**; veriyi
kendiniz indirirsiniz ve `data/` gitignore'dadır.

### neuPrint yolu (belirteç isteğe bağlı; anonim okuma bugün çalışıyor)

```powershell
$env:NEUPRINT_APPLICATION_CREDENTIALS = "<token>"     # https://neuprint.janelia.org/account — isteğe bağlı
py -3 scripts\fetch_neuprint.py --out data\connectome\neuprint --n-random 25000 --min-weight 3
$env:FLY_CONNECTOME_SOURCE = "neuprint" ; py -3 backend\run.py
```

Belirteç yoksa betik `Authorization` başlığı **göndermeden** stdlib `urllib` ile `POST /api/custom/custom` kullanır
(boş/sahte bir belirteç 401 döndürür); belirteç varsa `neuprint-python`. Sorgular ≤ 2000 gövde/parça, ≤ 3 eşzamanlı
istek, ≤ 8 tip çifti/sorgu ile parçalanır ve parça bazında sürdürülebilir.

### Codex / FlyWire (FAFB, dişi; CC-BY-NC 4.0 — yalnızca araştırma)

`classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`, `connections.csv.gz` dosyalarını bir dizine
koyup `FLY_CONNECTOME_SOURCE=csv`, `FLY_CONNECTOME_DIR=<dizin>` verin; yükleyici şemayı otomatik tanır, nöropil
başına satırları toplar ve `license` alanını `CC-BY-NC 4.0` yapar. Ticari/token bağlamında **kullanmayın**.

### Gerçek piyasa (DexScreener, anahtarsız; uç önbellek 60 s; 3 hatadan sonra sim'e döner ve arayüzde söyler)

```powershell
$env:FLY_MARKET = "dexscreener" ; $env:FLY_CHAIN = "solana" ; $env:FLY_TOKEN_ADDRESS = "<mint veya 0x adres>" ; py -3 backend\run.py
```

Sorgular arasında işlem akışı, `buys_m5`/`sells_m5` deltalarından **hız-eşlemeli vekil işlemler** ile doldurulur
(ticker'da `~` ön eki).

### Gerçek Claude (ücretli; claude-opus-5 fiyatlarında tweet başına ≈ $0.01)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..." ; $env:FLY_LLM = "anthropic" ; py -3 backend\run.py
curl -X POST http://127.0.0.1:4000/api/tweet/test          # üretir; FLY_X=dryrun olduğu sürece asla göndermez
```

### Gerçek X (kullandıkça öde; uygulama Read+Write olmalı, yazma açıldıktan sonra belirteçler yenilenmeli)

```powershell
$env:X_API_KEY="..." ; $env:X_API_SECRET="..." ; $env:X_ACCESS_TOKEN="..." ; $env:X_ACCESS_TOKEN_SECRET="..." ; $env:FLY_X = "post"
```

Fiyatlandırmayı önce developer.x.com'da doğrulayın; günlük tavan `FLY_TWEETS_PER_DAY=12`, bekleme süreleri §5'te.

### Diğer

* Oturumu bit-bit tekrar oynat: `py -3 scripts\replay.py data\sessions\<run_id>.jsonl --assert`
* Sentetik grafı CSV olarak dışa aktar (yükleyici eşdeğerlik kanıtı): `py -3 scripts\export_synthetic.py --out data\connectome\synthetic_export`
* Kıyaslama: `py -3 scripts\bench.py --n 20000,40000,80000 --dt 1.0,0.5`
* Sahne demosu: `py -3 scripts\demo_market.py [--loop]`

Portlar: backend 4000 (yalnızca 127.0.0.1), frontend 3000. `Ctrl+C` uvicorn'u durdurur; kapanışta sim/piyasa iş
parçacıkları bir tik içinde durur, `data/sessions/<run_id>.jsonl` ve `data/agent_state.json` yazılır. Günlükler:
konsol (ASCII) ve `/api/health.log_tail`.

---

## 7. Veri kaynakları ve lisanslar

| Kaynak | Ne için | Lisans | Erişim |
|---|---|---|---|
| **Janelia FlyEM MaleCNS v1.0** (`male-cns:v1.0`; 165,122 Traced nöron) | gerçek konektom (`csv` / `neuprint`), sentetik üretecin tip adları ve sayımları | **CC-BY 4.0** | GCS `gs://flyem-male-cns` (anonim HTTPS) · neuPrint API · https://male-cns.janelia.org/ |
| **FlyWire / Codex (FAFB, dişi)** | yükleyici şema eşdeğerliği, araştırma | **CC-BY-NC 4.0** (ticari değil) | Codex CSV indirmeleri |
| **MANC** (Takemura 2024) | VNC tip adları (`IN13A001`, `AN08B020`…) | CC-BY 4.0 | neuPrint `manc:v1.2.3` |
| Shiu ve ark. 2024 LIF parametreleri | nöron modeli | literatür | Nature 634:210 |
| DexScreener `/tokens/v1` | gerçek piyasa | API kullanım şartları | anahtarsız |

Atıf dizesi (`hello.connectome.citation`, `docs/NOTICE.md`):

> Berg et al. (2026) Sexual dimorphism in the complete Drosophila male central nervous system connectome. *Cell*.
> doi:10.1016/j.cell.2026.08.015 (data: FlyEM MaleCNS v1.0, CC-BY 4.0)

Sentetik modda `license = "synthetic (no data)"`, `citation = null`. Tam metinler ve DOI listesi: [`docs/NOTICE.md`](docs/NOTICE.md).

---

## 8. Kuklacılık listesi (puppeteering)

Sinek **her zaman** hareket etsin ve demo "canlı" görünsün diye eklenen, biyolojik olarak gerekçelendirilmemiş
mekanizmalar aşağıdadır. Kodda her sayı `[E]` (engineered) etiketi taşır; `[V]` doğrulanmış, `[L]` literatür demektir.
Her satırın kapatma anahtarı verilmiştir; hepsi kapalıyken kalan hareket saf konektom-güdümlüdür.

| # | Mekanizma | Nerede | Kapatma |
|---|---|---|---|
| 1 | **Keşif taban çizgisi** — `dng100` grubuna `30·explore` Hz sürüş (`explore = FLY_EXPLORE_BASELINE + 0.25·activity`, ≥ 7.5 Hz) | `encoder.py` satır 17 | `FLY_EXPLORE_BASELINE=0` |
| 2 | **OU gezinme** — yürüme/uçuşta yöne eklenen Ornstein–Uhlenbeck gürültüsü (`σ`, τ 1 s) | `decoder.py` `Wander` | `FLY_WANDER_SIGMA=0` |
| 3 | **Wander floor** — 3 s boyunca hız < 5 px/s ise `v_target = 40 px/s` (`wander_floor` olayı, `selftest.py` sayar) | `decoder.py` | `FLY_EXPLORE_BASELINE=0` ile birlikte kapanır |
| 4 | **Ruh hali geri beslemesi** — EUPHORIA → PAM 60 Hz + `flight_dn`; PANIC → PPL1 60 Hz + `dn_freeze` 20 Hz; ANXIOUS → PPL1 20 Hz; kur yapma → P1 80 Hz / 6 s; SLEEP → ruh hali geri besleme satırları **dışındaki** tüm satırlar × 0.3 (EUPHORIA/PANIC satırları SLEEP ile birlikte var olamaz; `p1` kur satırı da × 0.3 ölçeklenir); uyanınca 2 s boyunca tüm satırlar × 0.5 | `encoder.py` satır 19 | `FLY_MOOD_FEEDBACK=0` |
| 5 | **EPG proprioseptif geri besleme** — mevcut yöne göre 16 kama; kama k'ya 40 Hz, komşulara 15 Hz | `encoder.py` satır 18 | anahtar yok (`encoder.py` içinde tek satır) |
| 6 | **Duvar çarpması** — `wall_bump` tikinde `steer_a02_<taraf>` 40 Hz / 100 ms darbe | `server/loop.py` adım 6 | `FLY_WALLS=wrap` (çarpma olmaz) |
| 7 | Diğer `[E]` encoder satırları: su (2), loom tonik/aux/flaş (5–7), ışık flicker (9), koku (10), ödül/ceza (11–12), toz (13) — piyasa → duyu eşlemesinin kendisi bir tasarım kararıdır | `encoder.py` §f.2 | piyasa girdisi olmadan sıfır |
| 8 | GF zıplama refrakteri 1.5 s; DNp02/04/11 ileri-kalkış eğilimi | `decoder.py` | — |
| 9 | Gap-junction yama tablosu (DNp01→TTMn 300, DNp01→PSI 200, DNp01↔DNp01 80, LC4/LPLC2→DNp01 6) | `connectome/patches.py` | `meta.patches_applied` ile denetlenir |
| 10 | Tonik akım tablosu (lamina / motion_in 7.3 mV …) | `server/loop.py` başlangıç | — |
| 11 | **GF dinlenme freni** (P113 + P114) — P113: 159 hücreden DNp01'e ≈ 32 mV kararlı-durum inhibisyonu (derinlik, gürültü tabanıyla ölçeklenir); P114: tonik tablonun eşiğe yakın tuttuğu inhibitör hücrelerden (`mal`, Mi4/Mi9) ≈ 8 mV (süreklilik — soğuk başlangıçta ilk 10 ms içinde ateşleyen tek popülasyon onlardır) | `connectome/synthetic.py` `REST_BRAKES` | `FLY_NOISE_SIGMA=0` (gürültü yokken P113 de sıfırdır); `meta['rest_brakes']` ile denetlenir |

**Sentetik üretecin 21 `E` projeksiyon satırı** — gönderilen üreteç 114 projeksiyon kuralı üretir: 79'u `V`
(doğrulanmış), 14'ü `D` (türetilmiş), 21'i `E` (mühendislik). Denetim tek satırla yapılabilir:

```powershell
py -3 -c "import sys; sys.path.insert(0,'backend'); from flybrain.connectome.synthetic import PROJECTIONS; print(len(PROJECTIONS), [p.pid for p in PROJECTIONS if p.provenance=='E'])"
```

`E` satırları: kalkış DN → TTMn (P026), iniş (P028), yürüme premotor havuzları (P040–P042), beslenme → durma (P045),
valans → dümen/yürüme (P046–P047), uçuş dümen bağlamı (P056), sakkad vekili (P058), GNG117/234 kapanışı (P066), ikinci
disinhibisyon dalı (P071), acı bastırma / acı → PPL1 (P075–P076), şeker → PAM (P077), su (P078), ExR uyarılma kapısı
(P091), feromon/işitsel röle → pC1 (P100), tımar bacakları (P112) ve **GF dinlenme freni (P113 + P114)**: halka nöronları
(ER4d/ER4m/ER2_a), PVLP020/AOTU019/PS049/PS059/LAL083/LAL126/VES051/VES052, VNC inhibitör ara nöronları
(IN13A022/IN21A026/IN08A002), vPR9_a–c, GNG458/DNge129 ve dFB (FB6A/FB6H/FB7A/FB7B) → DNp01 (P113) ve
mAL_m8/mAL_m1 + Mi4/Mi9 → DNp01 (P114, tonik tabloyla sürülen süreklilik freni). Bu fren olmadan zorunlu
gürültü altında dev lif (DNp01) uyaransız 10–20 Hz ateşler (her GF spike'ı bir zıplamadır; §h.3 dinlenmede **0** DNp01
spike'ı ister); freni yapan hücrelerin tabloda girdisi ve encoder sürüşü yoktur, bu yüzden fren gürültü tabanıyla
ölçeklenir. Gerçek veriye geçince bu satırların hiçbiri kullanılmaz (yalnızca gap-junction yaması uygulanır).

> Not: `docs/SPEC.md` §g.3 bu tabloyu 112 satır / 19 `E` olarak sayar; üreteç bugün 114 satır / 21 `E` içeriyor
> (fazlalık P113 ve P114'tür). Bu README gönderilen kodu anlatır — yukarıdaki tek satırlık denetim her zaman kodu okur.

---

## 9. Test, smoke ve selftest

```powershell
py -3 -m pytest backend\tests -q                 # tümü, < 90 s;  -m "not slow" < 25 s
py -3 scripts\smoke.py                           # 2000 adım: rest 1-5 Hz -> sugar->MN9 -> loom->GF <= 20 ms -> recover; RTF >= 1.5
py -3 scripts\selftest.py                        # 5 kapı + RTF + 20 s başsız hareket testi + dry-run tweet; çıkış kodu 0/1
cd frontend ; npm run typecheck ; npm run lint   # strict TS (tsc --noEmit) + ESLint; ikisi de uyarısız geçer
cd frontend ; npm run build                      # üretim derlemesi (Turbopack); CI'da son adım
```

Test dosyaları (`backend/tests/`): `test_csr test_engine test_engine_backends test_synthetic test_groups test_loaders
test_encoder test_decoder test_mood test_market test_protocol test_agent test_server test_calibrate` (+ `test_cache`,
`test_monitor`, `test_patches`). Hepsi çevrimdışı ve deterministik; `torch`/`anthropic`/`tweepy` yalnızca `optional`
işaretli testlerde ve paket kuruluysa içe aktarılır. `test_protocol::test_types_ts_mirror`, `frontend/lib/types.ts`
alanlarını pydantic modelleriyle karşılaştırır — iki taraf sessizce ayrışamaz.

Sentetik graf kapıları (`snn/calibrate.run_gates`, §g.6): dinlenme 1–5 Hz · şeker 100 Hz → MN9 ≥ 20 Hz · looming 150 Hz →
DNp01 ilk spike ≤ 20 ms · kaçak yok (aktif oran ≤ %5) · PFL3 → DNa02 %100 kontralateral.

---

## 10. LLM ve X modları, maliyet

| Mod | `FLY_LLM` | `FLY_X` | Davranış |
|---|---|---|---|
| Kuru çalışma (varsayılan) | `dryrun` | `dryrun` | Tweet metni tohumlu şablonlardan (`model: "template"`), `out/tweets/<id>.{txt,json,png}` yazılır, `tweet` çerçevesi `dry_run: true` ile yayınlanır; ağ yok |
| Claude, gönderme yok | `anthropic` | `dryrun` | Metin Claude'dan (`claude-opus-5`, `max_tokens 512`, sistem istemi §d.6), yine gönderilmez |
| Tam canlı | `anthropic` | `post` | tweepy `create_tweet` + tuval PNG'si (v2 medya yükleme, v1.1 yedek, salt-metin yedek) |

LLM'in gördüğü **tek şey** ≤ 1.5 KB'lık beyin özeti JSON'u ve sistem istemidir (`docs/SPEC.md` §d.6). Sentetik kaynakta
istem "asla gerçek konektom olduğunu iddia etme" der. `validate_tweet` URL'leri siler, 280 karaktere keser, ≤ 2 hashtag
bırakır, `$FLY` dışındaki cashtag'leri atar ve nöron sözcük dağarcığından en az bir terim ister.

Tetikleyici: ruh hali geçişi 3 s boyunca korununca (`MoodMachine.confirmed`), `ESCAPE` hariç; `ESCAPE_BURST` (60 s'de
≥ 3 zıplama) ayrıca tetikler. Bekleme süreleri: 900 s genel, 2700 s sebep başına, günde 12 (UTC); `manual` (`T`
tuşu / `POST /api/tweet/test`) beklemeleri atlar, günlük tavanı atlamaz. Durum `data/agent_state.json`'da yeniden
başlatmalara dayanır.

**Maliyet notları**: Claude ≈ **$0.01 / tweet** (claude-opus-5 fiyatlarıyla, ≈ 1.5 KB girdi + 512 token çıktı; güncel
fiyatı doğrulayın). X API **kullandıkça öde**; yazma erişimi olan bir uygulama gerekir — fiyatı developer.x.com'da
doğrulayın. Varsayılan modda **hiçbir ücret** oluşmaz.

---

## 11. Arayüz ve klavye kısayolları

Dört Win95 penceresi: `untitled - Paint ($FLY)` (sinek + iz), `Oscilloscope - spike raster` (8 şerit, `REGIONS`
sırası: optic_lobe, antennal_lobe, mushroom_body, central_complex, sez, central_other, descending_motor, vnc; yıldız
satırları etiketli: `DNp01 R`, `MN9`, `PFL3 L`…), `Fly Status` (ruh hali paneli + piyasa ticker'ı) ve `tweets.txt -
Notepad`. Pencereler sürüklenebilir, konumları `localStorage`'da; 1280 px altında alt alta dizilir.

Pencerelerin içeriği:

* **Oscilloscope** — kaydırmalı raster (8 şerit × bölge başına 48 satır, 60 px/s, 8 s pencere), satır üzerine gelince
  etiket/taraf/nöron balonu, şerit başına anlık `Hz`, başlıkta `spikes/s` ve `active %`; `View ▸ Freeze raster` (`F`)
  ekran görüntüsü için kaydırmayı durdurur, `View ▸ Raster labels` etiketleri açar/kapatır.
* **Fly Status / ruh hali paneli** — 32 px piksel yüz, `MOOD_COLORS` rengiyle ruh hali adı ve `since`/`dwell`
  sayaçları, eşik çizgili (0.40/0.70 euphoria, 0.35/0.45/0.70 anxiety) segmentli çubuklar, dokuz sürüş göstergesi
  (looming'de `◄`/`►` taraf oku), 5 dakikalık ruh hali zaman çizelgesi ve 11 popülasyonun 8 s sparkline'lı `Hz`
  tablosu; altbilgide `run_id`, konektom adı/kaynağı, `n/e`, kazanç, `dt`, arka uç, `rtf`, `speed`.
* **Fly Status / piyasa ticker'ı** — sembol, kaynak rozeti (`SIM` / `DEX` / `SIM(fallback)`), rejim çipi, abonetik
  fiyat (`0.0₅1234` gösterimi), `m5/h1/h24` değişimleri, alış/satış çubuğu, `vol/liq/mcap`, son işlem (vekil
  işlemlerde `~`), `market` çerçevelerinden kurulan 5 dakikalık **çizgi** veya **15 s mum** grafiği (`line`/`candle`
  düğmeleri) ve son 12 işlemin bandı; altta `Options ▸ Market regime` ile aynı rejim çipleri.
* **tweets.txt - Notepad** — açılışta `GET /api/tweets?limit=20`, canlı `tweet` çerçeveleri başa eklenir (en çok 50);
  her satır `[SS:DD:ss] <ruh hali> (dry-run|posted) <metin>`, dry-run'da `DRY RUN` kaşesi, nöron çipleri, `url` varsa
  `open on X` bağlantısı, hata kırmızı; `Generate test tweet` düğmesi (`T`) 5 s kilitlenir.

| Tuş | Etki | Tuş | Etki |
|---|---|---|---|
| `S` | şeker dürtüsü | `B` | acı |
| `L` | looming (satış duvarı) | `W` | su |
| `D` | toz (tımar) | `C` | feromon (kur yapma) |
| `Z` | uyku | `R` / `P` | ödül / ceza |
| `N` | yeni tuval (izi temizle) | `F` | raster'ı dondur |
| `T` | test tweet'i | `1..6` | rejim CALM/PUMP/DUMP/CHOP/RUG/DEAD |
| `0` | `sim` kaynağına dön | `?` | Hakkında (köken bilgisi) |

Menüler: **File** (New, Save As = birleşik PNG), **View** (raster etiketleri, dondur, 1.5× yakınlaştır, FPS),
**Options ▸ Poke / Market regime / Test tweet**, **Help ▸ About** (`hello.connectome` adı/kaynağı/n/e/kazanç/lisans/
atıf/**not**, `run_id`, arka uç, dt, RTF). Tuvale tıklamak sineğin soluna/sağına göre taraflı bir şeker dürtüsü gönderir.

Mürekkep: renk mum yönüne göre (`#00a800` yukarı, `#a80000` aşağı) ve ruh haline göre ezilir (FEEDING turuncu,
COURTSHIP pembe, PANIC kırmızı, SLEEP gri, EUPHORIA gökkuşağı); kalınlık `1 + round(3·sat(|chg_m5|/3))` px; stil
`rainbow / zigzag / dotted / hearts / solid`; damgalar `blob / heart / zzz / dash / bump`.

---

## 12. Yol haritası

| Faz | Kapsam | Durum |
|---|---|---|
| **Faz 1 — Bugün, sıfır hesap** | Sentetik 20k beyin, simüle piyasa, dry-run LLM/X, dört Win95 penceresi, replay, smoke/selftest, çevrimdışı test paketi | bu depo |
| **Faz 2 — Gerçek girdiler** | `prepare_malecns.py` / `fetch_neuprint.py` ile MaleCNS `core` alt kümesi (calibrate_gain), DexScreener canlı piyasa, Claude ile tweet üretimi, X'e gönderim; About/README/LLM özetinde lisans ve atıf | env değişkenleriyle hazır; kullanıcı verisini indirir |
| **Faz 3 — Sadakat** | Tam graf (166.7k / 25.6M kenar, memmap + torch), `dt 0.1`, `literature` ağırlıkları ve literatür kapılarıyla karşılaştırma, Shiu 2024 MN9 eğrilerine göre `w_syn`/gain yeniden ayarı, kuklacılık satırlarının teker teker sıfırlanması ve etkisinin ölçülmesi | planlandı |
| **Faz 4 — DeSci / topluluk** | Oturum günlüklerinin paylaşımı ve bit-bit tekrarı, çoklu token / çoklu sinek, plastisite (PAM/PPL1 dopamin → KC→MBON ağırlık güncellemesi), dişi FlyWire beyniyle karşılaştırmalı koşular, neuPrint tip güncellemelerini izleyen otomatik yeniden üretim | fikir aşamasında |

---

## 13. Katkıda bulunma

* **Sözleşme önce**: `docs/SPEC.md` normatiftir. Bir imzayı, çerçeve alanını ya da formülü değiştirmek istiyorsanız önce
  SPEC'i değiştirin; `docs/RESEARCH.md` yalnızca doğrulanmış sayılar içindir (her satırda `[V]/[D]/[L]/[?]` güven etiketi).
* **Köken etiketleri**: biyolojik her sayının docstring'inde `[V]` / `[L]` / `[E]` bulunmalı; `[E]` satırları §8'deki
  listeye eklenmelidir.
* **Kurallar**: kod ve yorumlar İngilizce, README Türkçe; JSON anahtarları `snake_case` ve `frontend/lib/types.ts` ile
  1:1; konsol çıktısı ASCII (Windows konsolu cp1254); modül yükleme anında isteğe bağlı paket (`torch`, `anthropic`,
  `tweepy`, `pyarrow`, `pandas`, `neuprint`) içe aktarılmaz; `scipy` kullanılmaz; yollar CWD'ye değil `Settings.data_dir`
  / `__file__`'a göre çözülür; sim iş parçacığı asla G/Ç'de bloklanmaz; hiçbir dış hata simülasyonu durdurmaz.
* **Frontend**: mevcut Next 16 iskeleti; **yeni npm bağımlılığı yok**; `components/` altındaki her dosya `"use client"`;
  tik başına React state güncellemesi yok (tuvaller `requestAnimationFrame`, paneller 4 Hz `useSyncExternalStore`).
* **Testler**: §h'deki test adları normatiftir (ekleyebilirsiniz, yeniden adlandıramazsınız); PR'dan önce
  `py -3 -m pytest backend\tests -q`, `py -3 scripts\smoke.py`, `cd frontend; npm run typecheck; npm run lint`.
* Veri dosyalarını (`data/`, `out/`) asla commit etmeyin; MaleCNS verisi bu depodan dağıtılmaz (`docs/NOTICE.md`).

---

## 14. Bilimsel referanslar

Nöron modeli ve konektom (RESEARCH §1, §7):

* Berg S., Beckett I.R., Costa M., Schlegel P., Januszewski M., Marin E.C., Nern A., ve ark. (2026). *Sexual dimorphism in
  the complete Drosophila male central nervous system connectome.* **Cell**. doi:10.1016/j.cell.2026.08.015 — MaleCNS v1.0
  (ön baskı: bioRxiv 10.1101/2025.10.09.680999).
* Shiu P.K., Sterne G.R., Spiller N., ve ark. (2024). *A Drosophila computational brain model reveals sensorimotor
  processing.* **Nature** 634:210–219. doi:10.1038/s41586-024-07763-9 — LIF parametreleri (`v_rest −52`, `v_th −45`,
  `τ_m 20 ms`, `τ_s 5 ms`, `t_ref 2.2 ms`, gecikme 1.8 ms, `w_syn 0.275 mV`).
* Dorkenwald S. ve ark. (2024) **Nature** doi:10.1038/s41586-024-07558-y; Schlegel P. ve ark. (2024) **Nature**
  doi:10.1038/s41586-024-07686-5 — FlyWire (FAFB).
* Takemura S. ve ark. (2024) **eLife** doi:10.7554/eLife.97769 — MANC; Cheong H.S.J. ve ark. **eLife** doi:10.7554/eLife.96084 — DN → motor devreleri.
* Lappalainen J.K. ve ark. (2024) **Nature** doi:10.1038/s41586-024-07939-3 — konektom-kısıtlı görme modeli, histamin işareti.
* Eckstein N. ve ark. (2024) **Cell** doi:10.1016/j.cell.2024.03.016 — nörotransmitter tahmini.

Davranış literatürü (decoder, RESEARCH §13; DOI'ler Crossref'te doğrulandı):

* von Reyn C.R. ve ark. (2014) Nat Neurosci doi:10.1038/nn.3741 — GF spike zamanlaması → zıplama.
* Ache J.M. ve ark. (2019) Curr Biol doi:10.1016/j.cub.2019.01.079 — LC4/LPLC2 → GF looming; Klapoetke N.C. ve ark. (2017) Nature doi:10.1038/nature24626 — LPLC2.
* Zacarias R. ve ark. (2018) Nat Commun doi:10.1038/s41467-018-05875-1 — DNp09 donma.
* Bidaye S.S. ve ark. (2014) Science doi:10.1126/science.1249964 — MDN geri yürüme; Bidaye S.S. ve ark. (2020) Neuron doi:10.1016/j.neuron.2020.07.032 — ileri yürüme.
* Namiki S. ve ark. (2018) eLife doi:10.7554/eLife.34272 — DN atlası; Namiki S. ve ark. (2022) Curr Biol doi:10.1016/j.cub.2022.01.008 — DNg02 uçuş gücü.
* Yang H.H. ve ark. (2024) Cell doi:10.1016/j.cell.2024.08.033 — DNa02/DNa01/DNg13 dümen.
* Westeinde E.A. ve ark. (2024) Nature doi:10.1038/s41586-024-07039-2; Mussells Pires P. ve ark. (2024) Nature doi:10.1038/s41586-023-07006-3 — PFL3 → DNa02 hedef dümeni.
* Seelig J.D. & Jayaraman V. (2015) Nature doi:10.1038/nature14446; Green J. ve ark. (2017) Nature doi:10.1038/nature22343 — EPG/PEN pusula.
* Aso Y. ve ark. (2014) eLife doi:10.7554/eLife.04577 — MBON valans.
* Sterne G.R. ve ark. (2021) eLife doi:10.7554/eLife.71679; Shiu P.K. ve ark. (2022) eLife doi:10.7554/eLife.79887; Engert S. ve ark. (2022) eLife doi:10.7554/eLife.78110 — tat devreleri (LB3b/c şeker, LB1 acı).
* von Philipsborn A.C. ve ark. (2011) Neuron doi:10.1016/j.neuron.2011.01.011 — pIP10/dPR1/vPR6 şarkı.
* Hampel S. ve ark. (2015) eLife doi:10.7554/eLife.08758 — aDN tımar.
* Donlea J.M. ve ark. (2014) Neuron doi:10.1016/j.neuron.2013.12.013 — dFB uyku.

Veri lisansları ve tam köken bildirimi: [`docs/NOTICE.md`](docs/NOTICE.md). Bu proje bir sanat/meme deneyidir; **yatırım
tavsiyesi değildir** ve tweet'ler kazanç vaadi içermez (sistem istemi bunu yasaklar).
