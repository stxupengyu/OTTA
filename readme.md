# RAG for Multi-Label Classification

基于 RAG (Retrieval-Augmented Generation) 方法的大语言模型多标签分类项目。本项目通过检索相似样本作为示例来增强 LLM 的分类能力，从而提高多标签分类的准确性。

## 核心思想

与基础的 LLM 分类方法不同，本项目实现了 **测试时自适应** 的 RAG 方法：

1. **样本缓存机制**：在预测过程中维护一个高质量的样本缓存
2. **相似度检索**：对于每个待分类样本，从缓存中检索 top-k 个最相似的样本作为示例
3. **置信度筛选**：根据预测置信度选择高质量样本加入缓存，动态更新缓存内容
4. **渐进式增强**：在 warmup 阶段使用基础预测，之后逐步启用 RAG 检索

## 功能特性

- **RAG 增强预测**：使用相似样本作为示例，提升分类准确性
- **智能缓存管理**：基于置信度的样本筛选，维护高质量示例库
- **支持多种 LLM**：GPT-3.5、GPT-4o、Qwen2.5 等
- **完整的评估指标**：Micro-F1、Macro-F1、Example-F1
- **灵活的配置**：支持 base 模式和 rag 模式切换

## 环境要求

- Python 3.7+
- 安装所需依赖：

```bash
pip install openai sentence-transformers numpy
```

## 数据格式

### 测试文件格式 (`test.txt`)

每两个行表示一个样本，格式如下：

```
Text: "这里是待分类的文本内容"
Labels: 标签1, 标签2, 标签3

Text: "下一个样本的文本"
Labels: 标签1, 标签2

...
```

**注意**：每个样本之间需要有空行分隔。

### 标签描述文件格式 (`tag_description.csv`)

CSV 格式，包含两列：

- `tags`：标签名称
- `outputs`：标签的详细描述

示例：

```csv
tags,outputs
Action,"**Description**: Action films are a genre known for their high-energy..."
Comedy,"**Description**: This tag is assigned to films, TV shows..."
Drama,"**Description**: Drama is a genre of storytelling that focuses on..."
```

## 使用方法

### 基本用法

```bash
python3 main.py \
    --mode <模式> \
    --model-type <模型类型> \
    --dataset <数据集名称>
```

数据集名称可选值：`movie`, `aapd`, `rcv`, `se`（分别对应 MOVIE、AAPD、RCV、StackExchange 数据集）

### RAG 模式示例

```bash
python3 main.py \
    --mode rag \
    --model-type gpt3.5 \
    --dataset movie \
    --use-label-desc
```

### 基础模式示例（对比用）

```bash
python3 main.py \
    --mode base \
    --model-type gpt3.5 \
    --dataset movie \
    --use-label-desc
```

### 使用 GPT-4o

```bash
python3 main.py \
    --mode rag \
    --model-type gpt4o \
    --dataset aapd \
    --use-label-desc
```

### 使用 Qwen2.5

```bash
python3 main.py \
    --mode rag \
    --model-type qwen2.5 \
    --dataset movie \
    --use-label-desc \
    --api-key <你的API密钥> \
    --base-url https://api.siliconflow.cn/v1
```

### 自定义 API Key

```bash
python3 main.py \
    --mode rag \
    --model-type gpt3.5 \
    --dataset movie \
    --api-key <你的API密钥>
```

### 限制样本数量（用于快速测试）

```bash
python3 main.py \
    --mode rag \
    --model-type gpt3.5 \
    --dataset movie \
    --max-samples 100
```

## 参数说明

### 数据集参数

- `--dataset`：数据集名称，可选值：
  - `movie`：MOVIE 数据集
  - `aapd`：AAPD 数据集
  - `rcv`：RCV 数据集
  - `se`：StackExchange 数据集（默认：`movie`）

**注意**：程序会根据数据集名称自动查找对应的测试文件和标签描述文件（使用绝对路径）。所有数据集的文件格式都是一样的，因此不需要单独指定文件路径。

### 模式参数

- `--mode`：实验模式，可选值：
  - `base`：基础模式，不使用 RAG（用于对比）
  - `rag`：RAG 模式，使用检索增强（默认：`base`）

### 模型相关参数

- `--model-type`：模型类型，可选值：
  - `gpt3.5`：GPT-3.5 Turbo
  - `gpt4o`：GPT-4o
  - `qwen2.5`：Qwen2.5（默认：`gpt3.5`）

- `--model-name`：（可选）自定义模型名称，如果指定将覆盖 `--model-type` 的默认映射

- `--api-key`：（可选）API 密钥，如果不指定则使用默认值

- `--base-url`：（可选）API Base URL，如果不指定则根据模型类型自动选择

- `--system-prompt`：（可选）系统提示词，默认值：`"你是一个严格输出 JSON 的多标签分类器。"`

- `--use-label-desc`：（可选）是否在提示词中使用标签描述，默认：`False`

### 评估相关参数

- `--max-samples`：（可选）限制测试样本数量，用于快速调试（默认处理所有样本）

### 输出相关参数

- `--output-dir`：输出目录，用于保存预测结果和评估指标（绝对路径，默认：根据数据集自动设置为 `{项目根目录}/rag/output/{数据集名称}`）

### 限流配置

- `--request-interval`：每次 API 调用之前的等待时间（秒），用于限流，默认 `0.0` 秒

## RAG 机制说明

### 工作流程

1. **Warmup 阶段**（前 200 个样本）：
   - 使用基础 prompt 进行预测
   - 计算预测置信度
   - 将样本加入缓存（按置信度筛选，最多保留 100 个）

2. **RAG 阶段**（200 个样本之后）：
   - 使用 embedding 相似度检索 top-10 个相似样本
   - 构建包含相似样本示例的 RAG prompt
   - 使用 RAG prompt 进行预测（用于评估指标）
   - 同时使用基础 prompt 预测以获取置信度（用于更新缓存）

### 缓存管理策略

- **缓存大小**：最多保留 100 个样本
- **更新策略**：
  - 如果缓存未满，直接添加新样本
  - 如果缓存已满，用置信度更高的样本替换置信度最低的样本
- **置信度计算**：取所有预测标签的置信度均值（基于 logprobs）

### Embedding 模型

- 使用 `sentence-transformers/all-mpnet-base-v2` 进行文本 embedding
- 使用余弦相似度（归一化的点积）进行检索

## 输出说明

程序运行后会在指定的输出目录（默认：`./output`）下生成以下文件：

### 1. `log.txt`

详细的日志文件，包含：
- 模型加载信息
- RAG 缓存状态
- 每个样本的预测结果和检索信息
- 实时评估指标（第 10 个样本后，每 100 个样本输出一次）
- 最终评估结果

### 2. `metrics.json`

JSON 格式的评估指标文件，包含：

```json
{
  "micro-f1": 28.02,
  "macro-f1": 22.48,
  "example-f1": 26.54
}
```

### 3. `predictions.csv`

CSV 格式的预测结果文件，包含以下列：
- `id`：样本 ID
- `text`：原始文本
- `gold_labels`：真实标签（用分号分隔）
- `predicted_labels`：预测标签（用分号分隔）

## 评估指标说明

本工具计算三种 F1 指标：

1. **Micro-F1**：全局的精确率和召回率的调和平均数
   - 先计算所有样本的 TP, FP, FN，然后计算全局的 precision 和 recall，最后计算 F1
   - 适用于关注整体性能的场景

2. **Macro-F1**：每个标签的 F1 分数的平均值
   - 先计算每个标签的 F1 分数，然后对所有标签求平均
   - 适用于关注每个标签平衡性能的场景

3. **Example-F1**：每个样本的 F1 分数的平均值（也称为 Sample-F1）
   - 先计算每个样本的 F1 分数，然后对所有样本求平均
   - 适用于关注每个样本分类准确性的场景

## 与 Baseline 方法的对比

本项目与 `baselines/` 目录下的基础方法的主要区别：

| 特性 | Baseline | RAG |
|------|----------|-----|
| 预测方式 | 单次预测 | RAG 增强预测 |
| 样本缓存 | 无 | 有（基于置信度） |
| 相似度检索 | 无 | 有（embedding 相似度） |
| 测试时自适应 | 否 | 是 |

## 注意事项

1. **API 密钥安全**：建议通过环境变量或配置文件设置 API 密钥，而不是在命令行中直接传递

2. **数据路径**：确保测试文件和标签描述文件的路径正确，且格式符合要求

3. **标签一致性**：程序会自动检查测试文件中的标签是否在标签描述文件中，如果覆盖率低于 80% 会发出警告

4. **Embedding 模型下载**：首次使用时会自动下载 embedding 模型，需要一定时间和网络连接

5. **API 限流**：使用 LLM 模型时，注意 API 调用频率限制，必要时可通过 `--request-interval` 参数添加请求间隔

6. **RAG 模式性能**：
   - RAG 模式会产生两次 API 调用（一次用于评估，一次用于置信度计算）
   - 建议在测试时使用 `--max-samples` 限制样本数量进行快速验证

7. **缓存效果**：RAG 的效果依赖于缓存的样本质量，建议在 warmup 阶段使用高质量的数据

## 故障排除

### 问题：JSON 解析失败

**解决方案**：程序已内置 JSON 修复和重试机制，如果仍然失败，可以尝试：
- 检查 API 返回的原始响应
- 调整系统提示词，明确要求输出格式
- 使用 `--system-prompt` 参数自定义提示词

### 问题：Embedding 模型加载失败

**解决方案**：
- 检查网络连接（首次使用需要下载模型）
- 确认已安装 `sentence-transformers` 库
- 检查是否有足够的磁盘空间

### 问题：标签覆盖率警告

**解决方案**：
- 检查标签名称是否完全匹配（注意大小写和空格）
- 确认测试文件和标签描述文件对应同一个数据集

### 问题：RAG 缓存为空

**解决方案**：
- 确保已经处理了足够多的样本（warmup 阶段需要 200 个样本）
- 检查置信度计算是否正常（查看日志中的置信度信息）

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证

请查看项目根目录的 LICENSE 文件。

