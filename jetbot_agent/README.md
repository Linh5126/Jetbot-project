# JetBot agent (chạy TRÊN JetBot)

Agent HTTP cổng 8100 để `demo_pc.py --driver jetbot` trên PC điều khiển xe. Chỉ cần Python 3.6+, `cv2`, `numpy` (có sẵn trong JetPack) và thư viện motor (`jetbot` hoặc `Adafruit-MotorHAT`).

```bash
bash install.sh                 # chép vào ~/jetbot_agent, kiểm tra thư viện
python3 ~/jetbot_agent/agent.py # chạy tay, Ctrl+C để dừng
bash install.sh --service       # tự chạy khi bật máy
journalctl -u jetbot-agent -f   # xem log
```

Kiểm tra từ PC: `python scripts/jetbot_tool.py check`. Bánh xe được kê lên khi thử motor.

Tùy chọn: `--max-wheel 0.6` (trần tốc độ bánh), `--watchdog-s 1.0` (dừng nếu không nhận /drive), `--camera "<pipeline GStreamer>"` nếu camera CSI mặc định không mở được, `--motor fake` để chạy thử không động cơ.
