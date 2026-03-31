# Paper-Reading

工程实践笔记，记录 GPU 推理优化与模型量化相关技术要点，整理自实际项目调试经验。

---

## 目录

| 笔记 | 主题摘要 |
| :--- | :--- |
| [CUDA Stream 与 TensorRT 多流优化](./cuda_stream.md) | CUDA 流机制、单流 vs 多流、同步时机、TensorRT 异步推理范式 |
| [量化（AWQ）](./quantization.md) | AWQ 量化原理、激活感知缩放因子、W4A16 推理流程 |