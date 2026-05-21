# CIC 吊运考试手号识别 Demo

这是一个本地摄像头实时识别 Demo，使用 OpenCV 和 MediaPipe Pose 识别第一版手势：`STOP / 停止`。

## 识别逻辑

程序读取人体右肩、右肘、右手腕三个关键点，并判断：

- 右手臂是否向身体侧边水平伸出
- 右肩、右肘、右手腕是否大致在同一水平线上
- 右手腕是否明显在右肩外侧
- 姿态是否持续 0.5 秒以上

满足条件后，左上角会显示：

```text
Gesture: STOP
```

否则显示：

```text
Gesture: Not Detected
```

界面还会显示 `Match Score` 和 0.5 秒保持进度条。

## 环境要求

- Python 3.10 或 3.11 推荐
- 本地摄像头
- Windows / macOS / Linux 均可运行

## 安装

进入项目目录：

```powershell
cd "C:\Users\asgpm\Desktop\CIC demo"
```

如果之前装过环境，建议先删除旧虚拟环境：

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
```

用 Python 3.11 创建虚拟环境：

```powershell
py -3.11 -m venv .venv
```

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

```powershell
.\.venv\Scripts\python.exe main.py
```

运行后会打开摄像头窗口：

- 左上角显示 `Gesture: STOP / Not Detected`
- 显示 `Match Score`
- 画面上绘制 MediaPipe 人体骨架
- 按 `q` 退出

## 摄像头问题

如果窗口打开后提示摄像头读帧失败：

- 关闭微信、腾讯会议、浏览器等正在占用摄像头的软件
- 把 `main.py` 顶部的 `CAMERA_INDEX = 0` 改成 `1` 或 `2`
- 程序会自动尝试 `DirectShow`、`Media Foundation` 和默认 OpenCV 后端

终端里的 TensorFlow Lite、protobuf warning 通常不是致命错误，可以先忽略。

## 调参

如果现场摄像头角度、距离或左右镜像导致识别不稳定，可以优先调整 `main.py` 顶部这些参数：

```python
HOLD_SECONDS = 0.5
SHOULDER_ELBOW_WRIST_MAX_Y_SPREAD = 0.08
WRIST_OUTSIDE_SHOULDER_MIN_DISTANCE = 0.12
MIN_LANDMARK_VISIBILITY = 0.5
```

如果摄像头不是默认设备，修改：

```python
CAMERA_INDEX = 0
```
