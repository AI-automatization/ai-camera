from PIL import Image, ImageDraw, ImageFont

IMG = "/Users/madalievsardor33gmail.com/.claude/image-cache/92dd64d0-6443-4538-a94e-bde6bb74b8b8/17.jpeg"
LOGO = "/Users/madalievsardor33gmail.com/Desktop/Tezcode/tezcode-landing/public/tezcode-logo-white.png"
OUT = "/Users/madalievsardor33gmail.com/Desktop/camera-ai/robot_tezcode.png"

base = Image.open(IMG).convert("RGBA")
W, H = base.size

# logo — shaffof chetlarini kesib olamiz
logo = Image.open(LOGO).convert("RGBA")
bbox = logo.split()[3].getbbox()
logo = logo.crop(bbox)

# monogramma balandligi
mh = 66
mw = int(logo.width * mh / logo.height)
logo = logo.resize((mw, mh), Image.LANCZOS)

margin = 46
overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
overlay.alpha_composite(logo, (margin, margin))

# "tezCode" yozuvi (oq, monogramma yonida)
draw = ImageDraw.Draw(overlay)
try:
    font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 44, index=8)  # bold
except Exception:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 44)

tx = margin + mw + 18
tb = draw.textbbox((0, 0), "tezCode", font=font)
ty = margin + (mh - (tb[3] - tb[1])) // 2 - tb[1]
draw.text((tx, ty), "tezCode", font=font, fill=(255, 255, 255, 255))

out = Image.alpha_composite(base, overlay).convert("RGB")
out.save(OUT, quality=95)
print("saqlandi:", OUT, out.size)
