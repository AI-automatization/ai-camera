#!/bin/bash
# AI-kuzatuvchi (Claude) uchun Telegram yuborish: tools_tg.sh "matn" [rasm.jpg]
set -a; source ~/Desktop/camera-ai/.env; set +a
if [ -n "$2" ] && [ -f "$2" ]; then
  curl -s -F chat_id="$TELEGRAM_CHAT_ID" -F caption="$1" -F photo=@"$2" "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendPhoto" | head -c 80
else
  curl -s -d chat_id="$TELEGRAM_CHAT_ID" --data-urlencode text="$1" "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" | head -c 80
fi
echo
