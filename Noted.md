```
Take Noted and Command Explain.
```
There are 2 options for processing:
1. On CPU:
   + Libraries needed: 
  - pip  install torch torchvision
2. On GPU(CUDA):
   + Libraries need to install:
  - pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121.
  - python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" : used for verify after installed.
3. To switch from CPU to GPU:
   - always check config/setting.py after installed.
 + DEVICE = "cpu"   # force CPU mode for this test.
 + DEVICE = "cuda" if torch.cuda.is_available() else "cpu".
  ```
  "We tested the system on both CPU and GPU (NVIDIA RTX 3050, CUDA 12.5). 
  CPU performance averaged ~12 FPS, while GPU performance averaged ~35 FPS — approximately a 2.9x speed improvement. 
  This confirms that GPU acceleration significantly improves real-time performance, especially valuable if the system needs to process multiple camera feeds or higher-resolution video in production."
  ```

  ## Step-by-Step Comparison

| Step                        | CPU Install                           | GPU Install                                                                        |
| --------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------- |
| **Command**                 | `pip install torch torchvision`       | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **Where it downloads from** | Default PyPI storage                  | Special PyTorch CUDA storage                                                       |
| **File size**               | Smaller                               | Bigger (includes CUDA support code)                                                |
| **Works on**                | Any computer                          | Only computers with a matching NVIDIA GPU and driver                               |
| **Result**                  | `torch.cuda.is_available()` → `False` | `torch.cuda.is_available()` → `True` (if a supported NVIDIA GPU is available)      |


## video source

```
# Video source (0 for webcam, or path to video file or RTSP stream)
# VIDEO_SOURCE = 0 #for webcam
# VIDEO_SOURCE = "videos/test_video.mp4"  # for video file
```


## config Line crossing setting 
```
# --- Line crossing settings ---
# Settings for defining the line across which people are counted.
# Auto-line detection:
# LINE_TYPE = "vertical"
# LINE_TYPE = "horizontal"  # Horizontal line for counting people.
# LINE_POSITION = 180  # X for vertical, or Y for horizontal.
# LINE_INSIDE_POSITIVE_SIDE = False  # Right-to-left direction.
# LINE_INSIDE_POSITIVE_SIDE = True  # Left-to-right direction.
#
# Draw the line on the frame:
# LINE_POINT_A = (80, 302)  # First point of the line.
# LINE_POINT_B = (470, 162)  # Second point of the line.
# LINE_INSIDE_POSITIVE_SIDE = True  # Positive side is the inside side.
```

## RSTP Camera Source

```
# VIDEO_SOURCE = "rtsp://admin:Hik123456@192.168.88.117:554/ch1/main/av_stream"  # for RTSP stream
VIDEO_SOURCE = "rtsp://admin:Hik123456@192.168.88.125:554/ch1/main/av_stream"  # for RTSP stream
```