"""Qoida detektorlari — kadrdan qoida buzilishini topadi.

Har bir detektor bitta savolga javob beradi va rules.COVERAGE dagi nom bilan
chaqiriladi. Qoida matni bu yerda yozilmaydi — u API'dan keladi (rules.py),
bu yerda faqat "qanday aniqlanadi" mantiqi.

Uch tamoyil:

1) **Hodisa darhol jarima emas.** Bir kadrda mentor ko'rinmasligi — u egilgan
   yoki kamera burchagida bo'lishi mumkin. Shuning uchun har bir detektor
   holatni VAQT bo'yicha kuzatadi va faqat chegaradan oshsa signal beradi.

2) **Ishonchsiz kadr — signal emas.** Yuz tanilmasa yoki pose nuqtalari zaif
   bo'lsa detektor "bilmayman" deydi, "buzilgan" demaydi. Noto'g'ri jarima
   yo'q signaldan qimmat.

3) **Detektor jarima qo'ymaydi.** U faqat nomzod hodisa qaytaradi — odam
   ko'rib tasdiqlaydi. Avtomatik jarima qo'yish uchun rasm + odam qarori kerak.
"""
import time
import datetime
import threading

import rules
import schedule

# ── Zona turlari ─────────────────────────────────────────────────────
# Qaysi kamera nimani ko'radi. Ba'zi qoidalar faqat ma'lum zonada ishlaydi:
# 3.10 dars xonasida uxlash, 3.11 coworkingda uxlash — matni boshqa, jarimasi bir xil.
CLASSROOM = "classroom"
COWORKING = "coworking"
ADMIN = "admin"
KITCHEN = "kitchen"
ENTRANCE = "entrance"

ZONES = {
    "Yunusobod": {
        "101": ADMIN,      # Admin-kassa
        "201": COWORKING,  # Coworking 1
        "301": CLASSROOM,  # A2
        "401": CLASSROOM,  # A4
        "601": ADMIN,      # Admin
        "901": CLASSROOM,  # A5
        "1001": CLASSROOM,  # B2
        "1101": CLASSROOM,  # B4
        "1201": COWORKING,  # Coworking
        "1301": KITCHEN,   # Oshxona
        "1401": ADMIN,     # Admin 2
        "1501": CLASSROOM,  # B1
        "1601": CLASSROOM,  # B3
        "1701": CLASSROOM,  # A2 (2)
        "1801": CLASSROOM,  # A1
    },
    "Oybek": {
        "101": ENTRANCE, "201": COWORKING, "401": COWORKING, "501": CLASSROOM,
    },
    "Chilonzor": {
        "101": CLASSROOM, "201": CLASSROOM, "301": CLASSROOM, "401": CLASSROOM,
        "501": CLASSROOM, "601": KITCHEN, "701": CLASSROOM, "801": CLASSROOM,
        "901": ADMIN, "1001": CLASSROOM, "1101": COWORKING, "1201": COWORKING,
        "1301": COWORKING, "1401": CLASSROOM, "1501": COWORKING, "1601": ADMIN,
    },
}

# ── Chegaralar ───────────────────────────────────────────────────────
# Qoida matnida raqam bo'lsa — o'sha (schedule.py da). Bo'lmasa ehtiyotkor
# qiymat: yolg'on signal chinidan qimmatroq.
ADMIN_LOITER_MIN = 5      # 3.8: administratsiya zonasida shuncha turish
GATHERING_MIN = 5         # 2.5: coworkingda uzoq turib gaplashish
GATHERING_PEOPLE = 3      # shuncha odam yig'ilsa
SLEEP_MIN = 1.0           # 3.10/3.11: bosh shuncha vaqt pastda tursa
SEATED_SHARE = 0.7        # 2.10: mentor dars vaqtining shuncha qismini o'tirsa
COOLDOWN_SEC = 900        # bitta qoida bitta odamga shuncha vaqtda bir marta


class Streak:
    """Holat qancha vaqt uzluksiz davom etayotganini sanaydi.

    Kamera fonda sekin (8 sek) so'raladi, shuning uchun "uzluksiz" ni kadr
    soni bilan emas, vaqt bilan o'lchaymiz. gap_tol — shu vaqtgacha uzilish
    holatni bekor qilmaydi (bitta kadr o'tkazib yuborilishi normal).
    """

    def __init__(self, gap_tol=20.0):
        self.gap_tol = gap_tol
        self._since = {}
        self._last = {}
        self._lock = threading.Lock()

    def update(self, key, active, now=None):
        """Qaytaradi: holat necha sekunddan beri davom etayotgani (yoki 0)."""
        now = now or time.time()
        with self._lock:
            if not active:
                if now - self._last.get(key, 0) > self.gap_tol:
                    self._since.pop(key, None)
                return 0.0
            self._last[key] = now
            self._since.setdefault(key, now)
            return now - self._since[key]

    def reset(self, key):
        with self._lock:
            self._since.pop(key, None)
            self._last.pop(key, None)


class Detector:
    """Barcha detektorlar uchun umumiy asos.

    check() kadr tahlilini oladi va nomzod hodisalar ro'yxatini qaytaradi.
    Hodisa: {rule, who, reason, seconds} — qoida, kim, nega, qancha vaqt.
    """

    name = "base"
    gap_tol = 20.0

    def __init__(self):
        self.streak = Streak(self.gap_tol)
        self._last_fired = {}

    def rule_for(self, number):
        return rules.by_number(number)

    def _cooldown_ok(self, key, now=None):
        """Bitta hodisa qayta-qayta signal bermasin."""
        now = now or time.time()
        if now - self._last_fired.get(key, 0) < COOLDOWN_SEC:
            return False
        self._last_fired[key] = now
        return True

    def event(self, number, who, reason, seconds=0.0, now=None):
        rule = self.rule_for(number)
        if rule is None:      # qoida bazadan o'chirilgan — signal bermaymiz
            return None
        key = f"{number}:{who}"
        # Sovish vaqti ham kadr vaqtini ishlatadi — streak bilan bir xil soat.
        # Aks holda arxiv kadrlarni qayta tahlil qilganda hodisalar tushib qoladi.
        if not self._cooldown_ok(key, now):
            return None
        return {
            "rule_number": number,
            "rule_id": rule["id"],
            "rule_text": (rule.get("description") or "").strip(),
            "rule_type": rule["type"],
            "score": rule["score"],
            "who": who,
            "reason": reason,
            "seconds": round(seconds, 1),
            "at": datetime.datetime.fromtimestamp(
                now or time.time()).isoformat(timespec="seconds"),
            "detector": self.name,
        }

    def check(self, ctx):
        raise NotImplementedError


class Context:
    """Bitta kadr haqidagi hamma narsa — detektorlar shundan o'qiydi."""

    def __init__(self, branch, channel, camera_name, faces=None, persons=None,
                 now=None):
        self.branch = branch
        self.channel = channel
        self.camera_name = camera_name
        self.faces = faces or []        # faces.identify() natijasi
        self.persons = persons or []    # pose.people() natijasi
        self.now = now or datetime.datetime.now()

    @property
    def zone(self):
        return ZONES.get(self.branch, {}).get(self.channel)

    @property
    def room(self):
        """Kamera nomi = xona nomi (jadval bilan bog'lanish kaliti)."""
        return self.camera_name

    @property
    def lesson(self):
        return schedule.current_lesson(self.room, at=self.now, branch=self.branch)

    @property
    def named(self):
        """Tanilgan odamlar ismlari."""
        return [f["name"] for f in self.faces if f["name"]]

    @property
    def head_count(self):
        """Kadrdagi odamlar soni — pose ishonchliroq, bo'lmasa yuz."""
        reliable = [p for p in self.persons if p["reliable"]]
        return len(reliable) if self.persons else len(self.faces)


# ─────────────────────────────────────────────────── 1-daraja (yuz+jadval)
class LessonStart(Detector):
    """3.1 — darsni o'z vaqtida boshlash.

    Dars boshlangandan keyin mentor xonada ko'rinmasa signal. Mentorni
    ism bo'yicha emas, "kattalardan kimdir bormi" bo'yicha tekshiramiz:
    jadvaldagi teacher_id ko'pincha bo'sh, ism esa yuz bazasida bor.
    """

    name = "lesson_start"

    def check(self, ctx):
        lesson = ctx.lesson
        if lesson is None:
            return []
        late_by = (ctx.now - datetime.datetime.combine(
            ctx.now.date(), lesson["start"])).total_seconds() / 60
        if late_by < schedule.LATE_START_MIN:
            return []
        # Xonada hech kim yo'q = dars boshlanmagan
        empty = ctx.head_count == 0
        seconds = self.streak.update(f"{ctx.channel}:empty", empty, ctx.now.timestamp())
        if empty and seconds >= schedule.LATE_START_MIN * 60:
            ev = self.event("3.1", lesson.get("group_name") or ctx.room,
                            f"{ctx.room}: dars {lesson['start']} da boshlanishi kerak edi, "
                            f"{late_by:.0f} daqiqadan beri xona bo'sh", seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


class LeftRoom(Detector):
    """3.3 — dars vaqtida dars xonasini tark etmaslik."""

    name = "left_room"

    def check(self, ctx):
        lesson = ctx.lesson
        if lesson is None or ctx.zone != CLASSROOM:
            return []
        # O'quvchilar bor, lekin kattalardan hech kim tanilmadi
        students_present = ctx.head_count > 0
        mentor_present = bool(ctx.named)
        away = students_present and not mentor_present
        seconds = self.streak.update(f"{ctx.channel}:away", away, ctx.now.timestamp())
        if away and seconds >= schedule.AWAY_MIN * 60:
            ev = self.event("3.3", lesson.get("group_name") or ctx.room,
                            f"{ctx.room}: dars ketyapti, xonada {ctx.head_count} kishi bor, "
                            f"lekin mentor {seconds/60:.0f} daqiqadan beri tanilmadi", seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


class AloneWithStudent(Detector):
    """3.5 — mentor o'quvchi bilan xonada yolg'iz qolmasin.

    Qoida matni erkak mentor + qiz o'quvchi deydi. Kamera jinsni aniqlamaydi,
    shuning uchun kengroq holatni ko'rsatamiz: xonada atigi ikki kishi va
    biri tanilgan xodim. Qolganini odam hal qiladi.
    """

    name = "alone_with_student"
    gap_tol = 30.0

    def check(self, ctx):
        if ctx.zone != CLASSROOM:
            return []
        alone = ctx.head_count == 2 and len(ctx.named) == 1
        seconds = self.streak.update(f"{ctx.channel}:alone", alone, ctx.now.timestamp())
        if alone and seconds >= 120:
            who = ctx.named[0]
            ev = self.event("3.5", who,
                            f"{ctx.room}: {who} xonada bitta o'quvchi bilan "
                            f"{seconds/60:.0f} daqiqadan beri yolg'iz", seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


class AdminZoneLoitering(Detector):
    """3.8 — administratsiya hududida bemaqsad turish."""

    name = "admin_zone_loitering"

    def check(self, ctx):
        if ctx.zone != ADMIN:
            return []
        out = []
        for name in ctx.named:
            seconds = self.streak.update(f"{ctx.channel}:{name}", True,
                                         ctx.now.timestamp())
            if seconds >= ADMIN_LOITER_MIN * 60:
                ev = self.event("3.8", name,
                                f"{ctx.camera_name}: {name} administratsiya zonasida "
                                f"{seconds/60:.0f} daqiqadan beri", seconds,
                            now=ctx.now.timestamp())
                if ev:
                    out.append(ev)
        # Ketganlarni tozalash
        for name in list(self.streak._since):
            if name.startswith(f"{ctx.channel}:") and \
                    name.split(":", 1)[1] not in ctx.named:
                self.streak.update(name, False, ctx.now.timestamp())
        return out


class LateArrival(Detector):
    """2.1 / 2.2 — dars vaqtidan 5 daqiqa oldin ish joyida bo'lish.

    Ikkala qoida matni bazada bir xil (dublikat) — bitta signal beramiz,
    2.1 raqami bilan. Ikkinchisiga alohida jarima qo'yish adolatsiz.
    """

    name = "late_arrival"

    def check(self, ctx):
        if ctx.zone != CLASSROOM:
            return []
        lesson = schedule.upcoming_lesson(ctx.room, at=ctx.now, branch=ctx.branch,
                                          within_min=schedule.EARLY_MIN)
        if lesson is None or ctx.named:
            return []
        minutes_left = (datetime.datetime.combine(ctx.now.date(), lesson["start"])
                        - ctx.now).total_seconds() / 60
        ev = self.event("2.1", lesson.get("group_name") or ctx.room,
                        f"{ctx.room}: darsga {minutes_left:.0f} daqiqa qoldi, "
                        f"xonada mentor yo'q", now=ctx.now.timestamp())
        return [ev] if ev else []


class LessonOverrun(Detector):
    """2.8 — darsni o'z vaqtida tamomlash (5 daqiqagacha ruxsat)."""

    name = "lesson_overrun"

    def check(self, ctx):
        if ctx.zone != CLASSROOM:
            return []
        for lesson in schedule.lessons_today(branch=ctx.branch, day=ctx.now.date()):
            if lesson["room"] != ctx.room:
                continue
            end = datetime.datetime.combine(ctx.now.date(), lesson["end"])
            over_min = (ctx.now - end).total_seconds() / 60
            if over_min < schedule.OVERRUN_TOLERANCE or over_min > 60:
                continue
            still_going = ctx.head_count >= 2 and bool(ctx.named)
            seconds = self.streak.update(f"{ctx.channel}:over", still_going,
                                         ctx.now.timestamp())
            if still_going and seconds >= 60:
                ev = self.event("2.8", ctx.named[0] if ctx.named else ctx.room,
                                f"{ctx.room}: dars {lesson['end']} da tugashi kerak edi, "
                                f"{over_min:.0f} daqiqa oshdi", seconds,
                            now=ctx.now.timestamp())
                return [ev] if ev else []
        return []


# ─────────────────────────────────────────────────── 2-daraja (pose/obyekt)
class Sleeping(Detector):
    """3.10 (dars xonasida) / 3.11 (coworkingda) — uxlash, bosh partada."""

    name = "sleeping"

    def check(self, ctx):
        number = {CLASSROOM: "3.10", COWORKING: "3.11"}.get(ctx.zone)
        if number is None:
            return []
        # Dars xonasidagi qoida faqat dars vaqtida amal qiladi
        if number == "3.10" and ctx.lesson is None:
            return []
        sleeping = [p for p in ctx.persons if p["reliable"] and p["head_down"]]
        active = bool(sleeping)
        seconds = self.streak.update(f"{ctx.channel}:sleep", active,
                                     ctx.now.timestamp())
        if active and seconds >= SLEEP_MIN * 60:
            who = ctx.named[0] if ctx.named else f"{ctx.camera_name} ({len(sleeping)} kishi)"
            ev = self.event(number, who,
                            f"{ctx.camera_name}: bosh {seconds/60:.0f} daqiqadan beri "
                            f"pastda — uxlayotgan bo'lishi mumkin", seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


class MentorSeated(Detector):
    """2.10 — mentor darsni o'tirib o'tmasin."""

    name = "mentor_seated"

    def check(self, ctx):
        if ctx.zone != CLASSROOM or ctx.lesson is None or not ctx.named:
            return []
        seated = [p for p in ctx.persons if p["reliable"] and p["seated"]]
        # Hamma o'tirgan bo'lsa mentorni ajratib bo'lmaydi — o'quvchilar ham o'tiradi.
        # Faqat "hech kim turmagan" holatni belgilaymiz.
        all_seated = bool(seated) and len(seated) == len(
            [p for p in ctx.persons if p["reliable"]])
        seconds = self.streak.update(f"{ctx.channel}:seated", all_seated,
                                     ctx.now.timestamp())
        lesson_len = (datetime.datetime.combine(ctx.now.date(), ctx.lesson["end"])
                      - datetime.datetime.combine(ctx.now.date(),
                                                  ctx.lesson["start"])).total_seconds()
        if all_seated and seconds >= lesson_len * SEATED_SHARE:
            ev = self.event("2.10", ctx.named[0],
                            f"{ctx.room}: dars davomida hech kim turmadi "
                            f"({seconds/60:.0f} daqiqa) — mentor o'tirib o'tgan bo'lishi mumkin",
                            seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


class CoworkingGathering(Detector):
    """2.5 — coworkingda yig'ilib gaplashib turish."""

    name = "coworking_gathering"

    def check(self, ctx):
        if ctx.zone != COWORKING:
            return []
        # Faqat tanilgan xodimlar — o'quvchilar coworkingda o'tirishi normal
        staff = len(ctx.named)
        active = staff >= GATHERING_PEOPLE
        seconds = self.streak.update(f"{ctx.channel}:gather", active,
                                     ctx.now.timestamp())
        if active and seconds >= GATHERING_MIN * 60:
            ev = self.event("2.5", ", ".join(ctx.named[:4]),
                            f"{ctx.camera_name}: {staff} xodim {seconds/60:.0f} "
                            f"daqiqadan beri yig'ilib turibdi", seconds,
                            now=ctx.now.timestamp())
            return [ev] if ev else []
        return []


# ─────────────────────────────────────────────────────────────── ro'yxat
# Faqat yozilgan detektorlar. rules.COVERAGE da bor, lekin bu yerda yo'qlari
# hali yozilmagan — status() ularni ko'rsatadi.
DETECTORS = [
    LessonStart(), LeftRoom(), AloneWithStudent(), AdminZoneLoitering(),
    LateArrival(), LessonOverrun(),
    Sleeping(), MentorSeated(), CoworkingGathering(),
]


def run(ctx):
    """Barcha detektorlarni bitta kadrga qo'llaydi."""
    events = []
    for det in DETECTORS:
        try:
            events.extend(e for e in det.check(ctx) if e)
        except Exception as e:
            print(f"[detectors] {det.name} xato berdi: {e}")
    return events


def status():
    """Qaysi qoida qoplangan, qaysi biri hali yozilmagan."""
    written = {d.name for d in DETECTORS}
    done, todo = [], []
    for rule, cov in rules.camera_rules():
        (done if cov["detector"] in written else todo).append(
            (rule["title"], cov["detector"], cov["level"]))
    return sorted(done), sorted(todo)


if __name__ == "__main__":
    done, todo = status()
    print(f"── Yozilgan detektorlar: {len(done)} qoida")
    for number, det, lvl in done:
        print(f"   {number:<5} {lvl}-daraja  {det}")
    print()
    print(f"── Hali yozilmagan: {len(todo)} qoida")
    for number, det, lvl in todo:
        print(f"   {number:<5} {lvl}-daraja  {det}")
