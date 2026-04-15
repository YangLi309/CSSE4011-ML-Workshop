# CSSE4011 — ML & Edge AI Workshop

Hands-on notebook for CSSE4011: train and compress a small vision model, then export it for deployment.

## What’s in the workshop

- Fine-tune pretrained **AlexNet** on **MNIST** with PyTorch  
- **Prune** the model, then compare accuracy and inference speed  
- **Export** to ONNX and confirm ONNX Runtime matches PyTorch  
- **Quantize** ONNX models to FP16 and convert them to **TensorRT** engines  
- Compile and test **image inference** and **video inference** pipelines on the edge device  



## Task 1: Finetune and Export Your Model in Colab

Notebook: `CSSE4011_ML.ipynb`

**[Open the notebook in Google Colab](https://colab.research.google.com/github/YangLi309/CSSE4011-ML-Workshop/blob/main/CSSE4011_ML.ipynb)**

Use a **GPU** runtime (e.g., T4 GPU). Run the cells in order, and read the instructions and comments carefully to understand what each part is doing.

The exported ONNX models can also be downloaded from **[Google Drive](https://drive.google.com/drive/folders/1HAUrQRe-iRyVTkpbbwrqvOe5XdypbjsJ?usp=sharing)**

## Task 2: Deploy Your Model on Jetson

### 1. Quantize and Convert to TensorRT Engine

Quantize the exported ONNX models to **FP16** precision and convert them to TensorRT engines using `trtexec`:

```bash
trtexec --onnx=alexnet_mnist.onnx --fp16 --saveEngine=alex_mnist.engine --workspace=4068
```

### 2. Compile the Image Inference Pipeline

Compile the `mnist_inference.cpp` script to enable inference on a folder of images:

```bash
g++ -std=c++11 -o mnist_inference mnist_inference.cpp \
-I/usr/include/opencv4/opencv \
-I/usr/include/opencv4 \
-I/usr/local/cuda-10.2/targets/aarch64-linux/include \
-L/usr/local/cuda-10.2/targets/aarch64-linux/lib \
-lnvinfer -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
```

### 3. Test Image Inference

Run the compiled image inference pipeline on the test image folder:

```bash
./mnist_inference alex_mnist.engine ./mnist_jpg_test/
```

### 4. Compile the Video Inference Pipeline

Compile the `mnist_video_inference.cpp` script to enable inference on a video stream:

```bash
g++ -std=c++11 -o mnist_video_inference mnist_video_inference.cpp \
-I/usr/include/opencv4/opencv \
-I/usr/include/opencv4 \
-I/usr/local/cuda-10.2/targets/aarch64-linux/include \
-L/usr/local/cuda-10.2/targets/aarch64-linux/lib \
-lnvinfer -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
```

### 5. Identify Camera Path

Check the camera device path on your development board:

```bash
v4l2-ctl --list-devices
```

Example output:

```bash
USB2.0 UVC PC Camera (usb-3610000.xhci-2.1):
    /dev/video0
```

### 6. Test Video Inference

Run the video inference pipeline using the camera device path:

```bash
./mnist_video_inference alex_mnist.engine /dev/video0
```
