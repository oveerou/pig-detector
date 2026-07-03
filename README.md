# Pig Detector

> 猪只目标检测与养殖场景计数分析系统

## 项目简介

本项目面向养猪行业算法岗位，基于猪舍可见光图像实现猪只目标检测、数量统计、密度分析和拥挤风险提示。项目使用 YOLOv8n 作为检测模型，结合检测框面积占比和平均置信度设计规则化风险等级。

## 岗位匹配

匹配算法工程师岗位中以下能力点：

- 养猪行业相关算法模型开发
- 生产过程关键问题的数据分析和建模
- 深度学习框架使用（PyTorch / Ultralytics）
- 算法性能评估与优化
- 算法集成与系统展示
- Python 编码能力和工程化能力

## 背景来源

项目技术方向参考综述论文《生猪智能检测技术研究进展与未来展望》（DOI: 10.12133/j.smartag.SA202507048）中“可见光图像猪只检测 / 盘点 / 异常识别特征提取”分支。该论文为行业背景综述，**本项目不是对该论文的复现**。

## 数据说明

本地数据集路径：`D:/下载/pig`

```text
D:/下载/pig
├── train_img   # 700 张训练图片
├── train_json  # 700 个 JSON 检测框标注
└── test        # 220 张测试图片
```

JSON 标注格式示例：

```json
{
  "shape": [
    {
      "label": "猪",
      "boxes": [x1, y1, x2, y2],
      "points": null
    }
  ]
}
```

> **注意：** 原始数据不随本仓库发布。

## 技术栈

- Python 3.10+
- YOLOv8 (Ultralytics)
- OpenCV
- Pillow
- NumPy / pandas / matplotlib / seaborn
- Gradio
- pytest

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据审计

```bash
python scripts/audit_dataset.py --config configs/default.yaml
```

### 3. 数据格式转换

```bash
python scripts/convert_json_to_yolo.py --config configs/default.yaml
```

### 4. 训练模型

完整训练（50 轮）：

```bash
python scripts/train_yolo.py --config configs/default.yaml
```

冒烟训练（5 轮）：

```bash
python scripts/train_yolo.py --config configs/default.yaml --epochs 5
```

### 5. 模型评估

```bash
python scripts/evaluate_yolo.py --weights outputs/pig_yolov8n/weights/best.pt
```

### 6. 单图推理

```bash
python scripts/predict_image.py --weights outputs/pig_yolov8n/weights/best.pt --image D:/下载/pig/test/某张图片.jpg
```

### 7. 批量分析

```bash
python scripts/analyze_batch.py --weights outputs/pig_yolov8n/weights/best.pt --image-dir D:/下载/pig/test
```

### 8. Gradio 演示

```bash
python app.py
```

打开 `http://127.0.0.1:7860`，上传猪舍图片即可查看检测框、猪只数量、平均置信度和拥挤风险提示。

## 结果

待本地训练完成后补充真实指标。当前未真实跑出的指标不写入 README。

## 项目边界

- 本项目使用本地样例数据，**非牧原真实生产数据**。
- 提供本地 Gradio 演示，**非生产级部署系统**。
- 第一版不包含视频跟踪、体重估计、红外体温、声音识别、多模态融合。

## 目录结构

```text
pig-detector/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── app.py
├── scripts/
│   ├── audit_dataset.py
│   ├── convert_json_to_yolo.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── predict_image.py
│   └── analyze_batch.py
├── src/
│   ├── dataset_audit.py
│   ├── dataset_convert.py
│   ├── inference.py
│   ├── analytics.py
│   ├── visualization.py
│   └── utils.py
├── tests/
│   ├── test_dataset_convert.py
│   ├── test_analytics.py
│   └── test_inference_contract.py
└── docs/
    └── screenshots/
```

## 来源说明

本项目基于本地猪舍图像数据与课程项目工程化经验整理，模型结构、训练流程、风险规则均为项目自研实现。
