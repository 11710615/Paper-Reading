# CUDA Stream 与 TensorRT 多流优化实战笔记

适用于：PyTorch + TensorRT 推理（如 SAM/SAMV3 多模块模型）

整理自实际工程调试经验

---

## 1. 核心概念

### 1.1 什么是 CUDA Stream

CUDA Stream = **GPU 任务队列**。

所有 GPU 计算（模型推理、张量运算）都必须在某个流中执行。

- CPU：负责**把任务提交到流**

- GPU：负责**按流里的顺序执行任务**

### 1.2 异步 / 同步

- **异步（async）**

CPU 提交任务后**不等待 GPU 跑完**，直接继续执行后面代码。

是 GPU 高性能的基础。

- **同步（synchronize）**

CPU 阻塞，**等待流中所有 GPU 任务完成**，保证数据就绪。

---

## 2. 流的三大铁律

1. **同流内任务：自动串行，无需同步，绝对安全**

2. **异流任务：默认并行执行**

    - 无数据依赖 → 可以提速

    - 有数据依赖 → **必出脏数据 / 非法内存访问**

3. **跨流有依赖时：必须手动同步，否则结果无效**

---

## 3. 单流模式（默认/原始场景）

多个模块**共用同一个流**：

- CPU 串行提交任务

- GPU **链式串行执行**

- 任务顺序：先到先执行，不会并行

典型结构：

```Python

stream = torch.cuda.Stream()

img_encoder = TRTModule(..., stream=stream)
txt_encoder = TRTModule(..., stream=stream)
decoder     = TRTModule(..., stream=stream)

# 全程串行
img_feat = img_encoder(img)
txt_feat = txt_encoder(txt)
out = decoder(img_feat, txt_feat)
```

适用：

模块之间**强数据依赖**，必须按顺序跑（如 encoder → decoder）。

---

## 4. 多流优化（工程提速核心）

### 4.1 适用场景

模块之间**无数据依赖**，可以同时计算。

典型（以 SAM 为例）：

- 图像编码：`image_encoder`

- 文本/prompt 编码：`text_encoder`

两者**互不使用对方的输出**，可以并行。

### 4.2 推荐流划分

- `stream_img`：图像编码 + 解码器（有依赖，串行）

- `stream_txt`：文本编码（独立，可并行）

执行示意图：

```Plain Text

stream_txt:  text_encoder --------------------→
stream_img:  img_encoder ------→ mask_decoder ----→
总耗时 ≈ max(text, img) + decoder
```

---

## 5. 同步位置（最关键：避免脏数据）

### 5.1 规则

**凡是要“拼接/使用多个流的输出”时，必须在这一步之前同步所有相关流。**

典型流程：

1. 图像流异步推理

2. 文本流异步推理

3. **同步两个流**

4. 特征拼接 / 融合

5. 后续解码

### 5.2 同步代码位置

```Python

# 1. 双流并行推理
with torch.cuda.stream(stream_img):
    img_feat = img_encoder(img)

with torch.cuda.stream(stream_txt):
    txt_feat = txt_encoder(txt)

# ==========================
# 🔥 必须在这里同步
# ==========================
torch.cuda.synchronize(stream_img)
torch.cuda.synchronize(stream_txt)

# 然后才能安全拼接
feat = torch.cat([img_feat, txt_feat], dim=-1)
```

---

## 6. 与 TRTModule 结合的两种写法

### 6.1 写法 A：模块内部自带 synchronize（你当前的类）

如果 `TRTModule.__call__` 最后已经做了：

```Python

self.context.execute_async_v3(...)
self.synchronize()
```

那么：

- 模块调用完成 = 流已同步

- **特征拼接前不需要再加同步**

```Python

# 内部已同步，数据直接就绪
img_feat = img_encoder(img)
txt_feat = txt_encoder(txt)

# 直接拼接
feat = torch.cat([img_feat, txt_feat], dim=-1)
```

### 6.2 写法 B：高性能异步提交（最后统一同步）

去掉模块内部同步，只在**融合前统一同步**：

```Python

# 只提交任务，不等待
with torch.cuda.stream(stream_img):
    img_encoder(img)

with torch.cuda.stream(stream_txt):
    txt_encoder(txt)

# 等全部做完
torch.cuda.synchronize()
```

---

## 7. 常见坑总结

1. **多流 + 跨流数据依赖 + 不同步**

→ 脏数据、NaN、非法地址、程序崩溃

1. **流太多**

→ GPU 调度开销变大，反而变慢

1. **以为“多流一定更快”**

→ 只有无依赖模块才能提速；有依赖的必须同流串行

1. **混淆 ** **`execute_async_v3`** ** 与同步**

    - `async_v3` = 异步提交（快）

    - `synchronize` = 等结果对（准）

    - 工程上必须搭配使用

---

## 8. 极简总结

1. **同流 = 串行 = 安全**

2. **异流 = 并行 = 快，但只能用在无依赖模块**

3. **多流结果要融合 → 先同步再使用**

4. TensorRT 推理标准范式：**异步提交 + 按需同步**
> （注：文档部分内容可能由 AI 生成）
