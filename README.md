# MARS audit kamerasi

Kamera MARS ning rasmiy audit qoidalarini kuzatadi va nomzod hodisalarni
ko'rsatadi. **Jarima qo'ymaydi** — qarorni auditor qabul qiladi.

## Asosiy tamoyil

Qoidalar bu kodda yozilmagan. Yagona manba — Mars API:

```
GET /api/v2/reports/audit-rules      → 32 ta faol qoida
GET /api/v1/groups?branch_id=N       → dars jadvali (xona, vaqt, mentor)
```

Metodist adminda qoida qo'shsa yoki o'chirsa, kamera ham o'sha zahoti
yangi ro'yxat bilan ishlaydi. Kodga tegish shart emas.

## Ishga tushirish

```bash
./venv/bin/python app.py                      # brauzer: localhost:5001
```

Standart holda Yunusobod va Chilonzor birga ishga tushadi, sahifa tepasidan
tanlanadi. Boshqacha kerak bo'lsa:

```bash
BRANCHES="Chilonzor" ./venv/bin/python app.py
```

Har filialning o'z NVR si, o'z tezlik budjeti va o'z qulfi bor — biri
qulflansa ikkinchisi ishlayveradi. Oybek (`oybek.marsits.uz:8080`) ofis
tarmog'idan ochilmaydi.

## Qismlari

| Fayl | Vazifasi |
|---|---|
| `rules.py` | Qoidalar (Mars API) + qaysi qoidani kamera aniqlay oladi |
| `schedule.py` | Dars jadvali — xona, vaqt, mentor |
| `detectors.py` | Qoida detektorlari + zona xaritasi |
| `faces.py` | Kim ekanini tanish (YuNet + SFace) |
| `pose.py` | Tana holati — o'tirgan, bosh pastda |
| `nvr.py` | Hikvision kadr olish |
| `app.py` | Dashboard |

Har bir modul mustaqil ishlaydi:

```bash
./venv/bin/python rules.py           # qoidalar + kamera qamrovi
./venv/bin/python schedule.py        # bugungi jadval + kamera mosligi
./venv/bin/python detectors.py       # qaysi detektor yozilgan
./venv/bin/python faces.py rasm.jpg  # rasmda kim bor
./venv/bin/python pose.py rasm.jpg   # kim o'tirgan, kimning boshi pastda
./venv/bin/python test_detectors.py  # detektor mantiqi sinovi
```

## Qamrov

32 ta qoidadan:

| | Soni | Holati |
|---|---|---|
| 1-daraja (yuz + jadval + zona) | 7 | Yozilgan |
| 2-daraja (pose/obyekt) | 10 | 4 tasi yozilgan |
| Kamera tegmaydi | 14 | Sabab `rules.OUT_OF_SCOPE` da |

Hali yozilmaganlari: `badge` (1.1, 2.9), `phone_in_hand` (3.2),
`food_drink` (3.4), `chairs_left_untidy` (1.3), `dress_code` (2.3).

## Nega ba'zi qoidalarga tegilmaydi

`rules.OUT_OF_SCOPE` da har biri uchun sabab yozilgan. Qisqasi: ovoz kerak
(3.6, 3.12, 4.4), ekran mazmuni kerak (1.2, 3.9), yoki hodisa emas baho
(2.6, 3.7). 4.1 (janjal) va 4.3 (o'quvchini urish) texnik jihatdan mumkin,
lekin **ataylab qilinmagan** — noto'g'ri ayblov narxi yo'q signaldan qimmat.

## Yolg'on signalga qarshi uch qoida

1. **Bir kadr yetarli emas.** Har bir detektor holatni vaqt bo'yicha kuzatadi
   (`Streak`) va faqat chegaradan oshsa signal beradi.
2. **Ishonchsiz kadr — signal emas.** Pose nuqtalari zaif yoki yuz tanilmasa
   detektor "bilmayman" deydi.
3. **Sovish vaqti.** Bitta hodisa 15 daqiqada bir martadan ko'p signal bermaydi.

## Chegaralar qayerdan

Qoida matnida raqam bo'lsa — o'sha ishlatiladi (`schedule.EARLY_MIN = 5`,
`OVERRUN_TOLERANCE = 5` — ikkalasi ham qoida matnida yozilgan). Matnda raqam
bo'lmasa ehtiyotkor qiymat olingan va `detectors.py` da izohlangan.

## Ma'lum cheklovlar

- **Akkaunt qamrovi tor.** Metodist roli bilan `/api/v1/groups` atigi 9 ta
  guruh qaytaradi va `teacher_id` ko'pincha bo'sh. To'liq jadval uchun kengroq
  huquqli akkaunt kerak.
- **Bejik ishonchsiz.** Ko'k devor tufayli HSV usuli yolg'on signal beradi —
  shuning uchun eski `has_badge()` olib tashlandi, qayta yozilishi kerak.
- **Ovoz yo'q.** Ataylab: ovozga tayanadigan qoidalar hozircha qamrovdan tashqarida.
- **Zona xaritasi qo'lda.** `detectors.ZONES` — qaysi kamera nimani ko'radi.
  Kamera qo'shilsa shu yerga yozish kerak. Chilonzor zonalari kamera
  nomlaridan taxmin qilingan, joyida tekshirilmagan.
- **NVR qulflanadi.** Hikvision ko'p parallel digest so'rovni hujum deb biladi
  va akkauntni ~26 daqiqaga bloklaydi. Kod buni sezadi va qulf ochilishini
  kutadi, lekin qo'lda ko'p so'rov yubormaslik kerak.

## Bazadagi muammolar (kamera aybi emas)

`./venv/bin/python -c "import rules; print(rules.audit())"` ko'rsatadi:

- **1.3 ikki marta** — biri (id=1) butunlay bo'sh, jarima qo'yib bo'lmaydi
- **2.1 va 2.2** matni so'zma-so'z bir xil
- **2.4** va **1.3** bir xil qoida, lekin biri −2 ball, biri 0 ball
- 32 tadan **19 tasi "Boshqa"** kategoriyasida
