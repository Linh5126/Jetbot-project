#!/bin/bash
# Run ON the JetBot, from the folder that holds agent.py (e.g. a USB stick):
#   bash install.sh          copy to ~/jetbot_agent, check deps
#   bash install.sh --service  also start the agent automatically at boot
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/jetbot_agent"
mkdir -p "$DST"
cp "$SRC/agent.py" "$SRC/jetbot-agent.service" "$DST/"
echo "Copied to $DST"

python3 -c "import cv2, numpy; print('cv2', cv2.__version__, '| gstreamer:', 'YES' if 'GStreamer:                   YES' in cv2.getBuildInformation() else 'check')"
if python3 -c "import jetbot" 2>/dev/null; then
  echo "motor: jetbot package found"
elif python3 -c "import Adafruit_MotorHAT" 2>/dev/null; then
  echo "motor: Adafruit_MotorHAT found"
else
  echo "motor: no driver library on host python3 -> installing Adafruit-MotorHAT (needs internet)"
  pip3 install --user Adafruit-MotorHAT
fi
groups | grep -qw i2c || echo "WARNING: user $USER is not in group i2c: sudo usermod -aG i2c $USER (then reboot)"

if [ "$1" = "--service" ]; then
  sed "s#/home/jetbot#$HOME#g; s#User=jetbot#User=$USER#" "$DST/jetbot-agent.service" | sudo tee /etc/systemd/system/jetbot-agent.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now jetbot-agent
  sleep 2
  systemctl --no-pager status jetbot-agent | head -5
  echo "Logs: journalctl -u jetbot-agent -f"
else
  echo "Test by hand:  python3 $DST/agent.py     (Ctrl+C to stop)"
fi
