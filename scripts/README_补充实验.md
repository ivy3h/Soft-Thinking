# Soft-Thinking 补充实验脚本

本文档列出了需要补充的实验及对应的提交脚本。所有脚本使用 overlap 分区，12小时时限。

## 优先级1：完成部分评测

### 1. MGSM Qwen3-4B-Instruct Soft-Thinking 继续
- **脚本**: `mgsm_qwen3_4b_instruct_st_continue.sh`
- **任务**: 继续完成 Qwen3-4B-Instruct MGSM Soft-Thinking 评测
- **状态**: 已完成 en, es；需要完成剩余 9 种语言
- **提交命令**:
```bash
cd /nethome/jhe478/flash/Soft-Thinking
sbatch scripts/mgsm_qwen3_4b_instruct_st_continue.sh
```

### 2. MGSM Qwen3-8B 缺失的 te 语言
- **脚本**: `mgsm_qwen3_8b_missing_te.sh`
- **任务**: 完成 Qwen3-8B MGSM 非ST模式下缺失的 te (Telugu) 语言
- **状态**: 10/11 语言完成，缺 te
- **提交命令**:
```bash
sbatch scripts/mgsm_qwen3_8b_missing_te.sh
```

### 3. XReasoning Qwen3-4B-Instruct GPQA
- **脚本**: `xreasoning_qwen3_4b_instruct_gpqa.sh`
- **任务**: 评测 Qwen3-4B-Instruct 在 GPQA 数据集上的表现（非ST）
- **状态**: 未开始
- **提交命令**:
```bash
sbatch scripts/xreasoning_qwen3_4b_instruct_gpqa.sh
```

### 4. XReasoning Qwen3-8B-Base 完成
- **脚本**: `xreasoning_qwen3_8b_base_continue.sh`
- **任务**: 完成 Qwen3-8B-Base 的 XReasoning 评测
  - AIME 2024: 完成第5次运行
  - AIME 2025: 5次运行
  - GPQA: 1次运行
- **状态**: AIME 2024 完成 4/5
- **提交命令**:
```bash
sbatch scripts/xreasoning_qwen3_8b_base_continue.sh
```

### 5. XReasoning Qwen3-8B 完成
- **脚本**: `xreasoning_qwen3_8b_continue.sh`
- **任务**: 完成 Qwen3-8B 的 XReasoning 评测
  - AIME 2024: 完成剩余3次运行
  - AIME 2025: 5次运行
  - GPQA: 1次运行
- **状态**: AIME 2024 完成 2/5
- **提交命令**:
```bash
sbatch scripts/xreasoning_qwen3_8b_continue.sh
```

## 优先级2：开始 Soft-Thinking 评测

### 6. MGSM Qwen3-8B-Base Soft-Thinking
- **脚本**: `mgsm_qwen3_8b_base_st.sh`
- **任务**: 开始 Qwen3-8B-Base MGSM Soft-Thinking 评测（11种语言）
- **状态**: 未开始
- **提交命令**:
```bash
sbatch scripts/mgsm_qwen3_8b_base_st.sh
```

### 7. MGSM Qwen3-8B Soft-Thinking
- **脚本**: `mgsm_qwen3_8b_st.sh`
- **任务**: 开始 Qwen3-8B MGSM Soft-Thinking 评测（11种语言）
- **状态**: 未开始
- **提交命令**:
```bash
sbatch scripts/mgsm_qwen3_8b_st.sh
```

## 批量提交脚本

可以使用以下命令按优先级批量提交：

### 优先级1（最紧急）
```bash
cd /nethome/jhe478/flash/Soft-Thinking
sbatch scripts/mgsm_qwen3_4b_instruct_st_continue.sh
sbatch scripts/mgsm_qwen3_8b_missing_te.sh
sbatch scripts/xreasoning_qwen3_4b_instruct_gpqa.sh
```

### 优先级1（XReasoning 完成）
```bash
sbatch scripts/xreasoning_qwen3_8b_base_continue.sh
sbatch scripts/xreasoning_qwen3_8b_continue.sh
```

### 优先级2（Soft-Thinking 新评测）
```bash
sbatch scripts/mgsm_qwen3_8b_base_st.sh
sbatch scripts/mgsm_qwen3_8b_st.sh
```

### 全部提交
```bash
cd /nethome/jhe478/flash/Soft-Thinking
for script in scripts/mgsm_qwen3_4b_instruct_st_continue.sh \
              scripts/mgsm_qwen3_8b_missing_te.sh \
              scripts/xreasoning_qwen3_4b_instruct_gpqa.sh \
              scripts/xreasoning_qwen3_8b_base_continue.sh \
              scripts/xreasoning_qwen3_8b_continue.sh \
              scripts/mgsm_qwen3_8b_base_st.sh \
              scripts/mgsm_qwen3_8b_st.sh; do
    sbatch $script
done
```

## 日志文件

所有日志文件将保存在 `/coc/pskynet6/jhe478/Soft-Thinking/logs/` 目录下，文件名格式为：
- `mgsm_qwen3_4b_instruct_st_continue_<job_id>.log`
- `mgsm_qwen3_8b_missing_te_<job_id>.log`
- `xreasoning_qwen3_4b_instruct_gpqa_<job_id>.log`
- `xreasoning_qwen3_8b_base_continue_<job_id>.log`
- `xreasoning_qwen3_8b_continue_<job_id>.log`
- `mgsm_qwen3_8b_base_st_<job_id>.log`
- `mgsm_qwen3_8b_st_<job_id>.log`

## 注意事项

1. 所有脚本使用 `--resume` 参数，支持断点续传
2. 使用 overlap 分区，12小时时限
3. 单GPU配置（`--gres=gpu:1`）
4. 确保 conda 环境 `st` 已正确配置
5. MGSM 脚本会自动处理所有语言，无需指定
6. XReasoning 脚本中的多个 dataset 调用会顺序执行

## 预计完成时间

根据之前的运行经验：
- MGSM 单个模型（11种语言）：约 6-8 小时
- XReasoning 单个数据集（1次运行）：约 1-2 小时
- XReasoning 单个数据集（5次运行）：约 5-10 小时

建议分批提交，避免同时占用过多资源。
