# ExpStar流程说明

该部分对ExpStar的整体流程进行说明，涵盖了数据处理、模型训练、推理与评估的完整流程。以下为各模块功能及使用说明。

## 目录结构

```
repo/
├── code/                # 主要代码目录
│   ├── 1video_commentary_pair_construction/  # 视频-讲解对构建
│   ├── 2dataset_construction/                # 数据集构建
│   ├── 3retrevial/                           # 检索相关代码（支持多种检索器）
│   ├── 4train/                               # 训练脚本
│   ├── 5infer/                               # 推理脚本
│   └── 6eval/                                # 评估脚本
├── Demo/                # 数据与结果示例
│   ├── 1data/           # 原始数据（视频、ASR、步骤等）
│   ├── 2pair-data/      # 视频-讲解对
│   ├── 3baseline_dataset/    # baseline数据集
│   ├── 4Expstar_dataset/     # ExpStar数据集
│   ├── 5Expstar_rl_dataset/  # ExpStar_RL数据集
│   ├── 6Expstar_result/      # 推理结果示例
│   └── 7eval/                # 评估数据示例
└── Readme.md            # 项目说明文档
```

## 数据处理流程

1. **原始数据清洗**  
   - 位于 `Demo/1data/`，包括原始视频、ASR转写、实验步骤（部分通过GPT-4o自动补充安全与原理信息）。
2. **视频-讲解对构建**  
   - 使用 `code/1video_commentary_pair_construction/` 处理，生成示例详见 `Demo/2pair-data/`。
3. **数据集构建**  
   - 通过 `code/2dataset_construction/`，生成 baseline、ExpStar、ExpStar_RL 数据集，生成示例详见 `Demo/3baseline_dataset/`、`Demo/4Expstar_dataset/`、`Demo/5Expstar_rl_dataset/`。
4. **检索增强（RAG）**  
   - 参考 `code/3retrevial/`，支持多种检索器（如 CLIP、EVA_CLIP、ViCLIP）和检索方式。

## 模型训练

- 推荐使用 4 卡 A100 进行训练。
- 包含 SFT（监督微调）与 DPO（偏好优化）两阶段训练。
- 训练脚本：
  - SFT: `code/4train/expstar_train.sh`
  - DPO: `code/4train/rl_dpo.sh`

## 推理流程

- baseline 推理采用单轮对话，脚本：`code/5infer/baseline_data_infer.sh`
- ExpStar 推理采用多轮对话，客户端-服务器模式：
  - 服务器端：`code/5infer/deploy_multi_port.sh`
  - 客户端：`code/5infer/expstar_data_infer.py`
- 推理结果示例见 `Demo/6Expstar_result/`

## 评估

- 批量评估脚本：`code/6eval/batch_evaluate.py`
- 评估所用数据格式示例见 `Demo/7eval/`

## 其他说明

- 各阶段数据、结果均有 Demo 示例，便于复现和理解流程。
- 检索增强部分代码参考了 self-rag 项目。



