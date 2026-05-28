# CDSS — 医学知识蒸馏 + 模型训练监控 + 临床推理演示平台

围绕以下六个工作流组织的端到端平台：

```
数据导入 → 知识抽取 → 数据集构建
                    ↓
       继续预训练 (CPT) → SFT 微调 → 临床助手 / 模型评估
```

CPT（continued pre-training，可选但推荐）会先用因果语言建模在原始医学文档上注入领域知识，再由 SFT 教模型 QA 格式。

## 技术栈

- **后端**：FastAPI + SQLAlchemy（SQLite）+ HuggingFace Transformers / PEFT / TRL + vLLM
- **前端**：React 19 + Vite + Tailwind CSS 4 + Recharts + React Router
- **本地模型服务**：vLLM 与后端共用同一个 conda env（`vllm_manager._resolve_vllm_executable()` 取 `sys.executable` 同目录的 `vllm` 可执行文件，确保后端进程 spawn 的子进程拿到同样的依赖栈）

## 项目结构

```
cdss/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI 入口
│   │   ├── database.py         # SQLAlchemy 引擎 + run_migrations()
│   │   ├── config.py           # pydantic-settings 读取 .env
│   │   ├── models/models.py    # 所有 ORM 模型集中在此文件
│   │   ├── api/                # 每个资源一个 router
│   │   │   ├── documents.py    # 文档管理（含持久化删除）
│   │   │   ├── extraction.py   # 知识抽取任务
│   │   │   ├── datasets.py     # 数据集构建/导出
│   │   │   ├── pretraining.py  # CPT 实验
│   │   │   ├── training.py     # SFT 实验
│   │   │   ├── evaluations.py  # LLM-as-Judge 评估
│   │   │   ├── assistants.py   # LLM 助手 CRUD（本地 vllm + 远程 API）
│   │   │   └── assistant.py    # 临床演示端点
│   │   └── services/
│   │       ├── data_loader.py        # xlsx / md → documents 表
│   │       ├── extractor.py          # 抽取 job runner
│   │       ├── trainer.py            # SFT 子进程 + 监控线程
│   │       ├── train_script.py       # 独立 SFT 训练脚本
│   │       ├── pretrainer.py         # CPT 子进程 + 监控线程
│   │       ├── pretrain_script.py    # 独立 CPT 训练脚本
│   │       ├── evaluator.py          # LLM-as-Judge 运行器
│   │       ├── vllm_manager.py       # vllm 子进程生命周期 + 健康检查
│   │       ├── llm_client.py         # resolve_assistant() + chat_completion()
│   │       └── _subprocess_env.py    # 给子进程注入 LD_LIBRARY_PATH 等
│   ├── requirements.txt
│   ├── .env.example
│   └── cdss.db                 # SQLite 数据库（不入仓库）
├── frontend/
│   ├── src/
│   │   ├── api/client.js       # axios 客户端
│   │   ├── pages/              # 每个一级路由一个文件
│   │   └── components/         # 通用组件（MarkdownView 等）
│   ├── package.json
│   └── vite.config.js          # 代理 /api → http://localhost:8001
├── crawler/                    # 病例报告 xlsx 数据（不入仓库）
├── guideline/                  # 临床指南 md 数据（不入仓库）
├── CLAUDE.md                   # Claude Code 协作指引（可入仓库）
└── README.md
```

## 环境准备

### 1. 后端 Python 环境（conda 推荐）

```bash
conda create -n cdss python=3.10 -y
conda activate cdss

cd backend
pip install -r requirements.txt
# 训练相关包按平台选择 CUDA 版本，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 5090/H100 等新卡
# 或 V100 等老卡：
# pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft trl accelerate datasets
pip install bitsandbytes        # 仅 Linux + CUDA，4-bit 量化所需
```

### 2. vLLM 安装（与后端**同一**conda env）

后端 spawn vllm 时取的是 `sys.executable` 同目录的 `vllm` 可执行文件——也就是说后端环境里必须能 `vllm --help`，不要装到单独的 env。

```bash
conda activate cdss
# 版本按 GPU 架构选：
#   5090 / H100 / A100 → vllm 最新版
#   V100  / T4         → vllm <= 0.6.x（FlashAttention-2 不支持 Volta）
pip install vllm
```

如果之前在另一个 env 装过 vllm，要么 `pip install vllm` 装进 cdss，要么改 `vllm_manager._resolve_vllm_executable()` 走绝对路径。

### 3. 前端

```bash
cd frontend
npm install
```

### 4. CUDA 运行库（关键，bitsandbytes / vllm 都依赖）

`backend/app/services/_subprocess_env.py` 会在 spawn 子进程时把 conda env 下的 CUDA 库目录前置到 `LD_LIBRARY_PATH`。原因：`ctypes.CDLL` 在进程启动时缓存路径，Python 内部修改 `os.environ` 来不及——必须在 `subprocess.Popen(env=...)` 时设。所以无需手动 export，确保 `conda install -c nvidia cuda-toolkit=<匹配版本>` 装好即可。

## 配置

复制示例文件：

```bash
cd backend
cp .env.example .env
```

按需修改：

```ini
# 默认全局 LLM（被抽取/评估等模块作为 fallback；可被「助手」覆盖）
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=
LLM_MODEL=Qwen2.5-7B-Instruct

# 原始数据目录（相对 backend/ 的路径）
CRAWLER_DATA_PATH=../crawler
GUIDELINE_DATA_PATH=../guideline

MAX_CONCURRENT_EXTRACTIONS=50
```

## 启动

```bash
# Terminal 1 — 后端
conda activate cdss
cd backend
uvicorn app.main:app --reload --port 8001   # Vite 代理写死了 8001，不要改

# Terminal 2 — 前端
cd frontend
npm run dev                                  # 默认 5173

# Terminal 3+ — vllm 本地助手由后端自动 spawn（不需手动起服务）
#   也可作为调试参考：
#   vllm serve /path/to/model --served-model-name <name> --host 127.0.0.1 --port 8011
```

浏览器打开 `http://localhost:5173`。首次进入 → 文档管理 → 加载数据文件，从 `crawler/` 和 `guideline/` 导入原始文档。

## 不同 GPU 架构兼容性说明

| 架构 | 卡型示例 | 注意事项 |
|---|---|---|
| **Ampere+ / Hopper+** | A100, RTX 3090/4090, H100, RTX 5090 | 全功能可用；推荐 `--dtype bfloat16`、vllm 最新版 |
| **Volta / Turing** | V100, T4, RTX 2080 | **不支持 bfloat16**：助手 `extra_vllm_args` 必须加 `--dtype half`<br>**不支持 FlashAttention-2**：vllm 会自动回退到 XFormers backend<br>使用 vllm <= 0.6.x；torch 用 cu118 / cu121 编译版 |

混型 GPU 主机（如 V100-DGXS + V100-PCIE 共存）已在 `_subprocess_env.py` 默认设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，确保 `CUDA_VISIBLE_DEVICES=N` 选中的卡与 `nvidia-smi` 显示一致。

### 多卡 vllm（tensor parallel）

后端检测到 `gpu_ids` 数量 > 1 时自动：
- 注入 `--tensor-parallel-size <N>`
- 把环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 喂进子进程（**重要**：默认 `fork` 在 CUDA 下会与 worker 进程死锁，症状是日志停在 `Started engine process` 之后再无任何输出直到我们的监控超时）
- 把 `NCCL_IB_DISABLE=1` 喂进子进程（避开 InfiniBand 探测的 30~120s 超时；不影响 NVLink 吞吐）
- 用 `start_new_session=True` 让 vllm leader + 所有 TP worker 在同一会话中；停止时用 `killpg(SIGTERM → SIGKILL)` 一锅端，避免 V100 上 NCCL 死锁导致的 worker 僵尸（占着每卡 ~700 MB CUDA context 不释放）
- 监控超时按 GPU 数线性放大：单卡 10 分钟，2 卡 15 分钟，N 卡 = 600 + 300×(N-1) 秒

### V100 异构 GPU 主机（重点）

如果服务器同时插了 V100-DGXS（带 NVLink）+ V100-PCIE / V100S-PCIE 等不同 SKU，**NCCL 在多卡初始化 `ncclCommInitRank` 时会卡死**——因为它会对所有 GPU 对做 P2P 探测，而异构 PCIe 桥之间 P2P 实际不可用却走死等协议。日志症状：

```
INFO ... pynccl.py:63] vLLM is using nccl==2.20.5
（之后再无任何输出，监控线程一直 still waiting）
```

在助手编辑表单的"额外环境变量"里加：

```
NCCL_P2P_DISABLE=1
```

这会让 NCCL 走 CPU 中转代替 GPU 直连，单节点同 PCIe 主机性能损失约 30~50%，但功能正确不再卡死。如果还有问题再加 `NCCL_DEBUG=INFO` 把 NCCL 的初始化过程打印到日志里精确定位。

### 助手健康检查匹配规则（V100 用户重点）

老版本 vllm 在 `/v1/models` 返回的 `id` 字段大小写、路径前缀可能与你填的 `model_name` 略有差异。`vllm_manager._ping_vllm` 已升级为三档匹配：
1. 严格相等 → `ok`
2. 大小写不敏感 / basename / 子串 → `ok-loose:<actual>`
3. 服务在响应但名字完全不匹配 → `ok-any` （仍标为 running，但 `error_message` 提示请修正助手的 model_name）

每次轮询的结果会写进 `backend/vllm_logs/assistant_<id>.log`，可以 `tail -f` 跟踪。

## 常见问题

| 现象 | 排查 |
|---|---|
| 助手按钮卡在「启动中」 | `tail -f backend/vllm_logs/assistant_<id>.log`，找 `[manager] still waiting` 行查最后一次 ping 失败原因 |
| 多卡助手日志停在 `Started engine process` 后无任何输出 | 多进程死锁；确认 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 已生效（本项目自动设），若无效再按上文「多卡 vllm」段试 NCCL_P2P_DISABLE / NCCL_IB_DISABLE |
| 多卡助手日志停在 `vLLM is using nccl==X.Y.Z` 后无输出 | V100 异构 PCIe 主机的 NCCL P2P 探测死锁；助手「额外环境变量」加 `NCCL_P2P_DISABLE=1` |
| 停止多卡助手后 `nvidia-smi` 仍有进程占着每卡 ~700MB | 老版本会出现 — 升级到本版后 stop() 已改 killpg 整组终止；若仍有残留，`pkill -9 -f vllm` 兜底 |
| `libnvJitLink.so.13: cannot open shared object file` | 没装匹配版本的 `cuda-toolkit`，或 `_subprocess_env.py` 没发现 conda env；用 `conda install -c nvidia cuda-toolkit=<X>.0` 装上 |
| CPT eval 阶段 OOM | 已通过 `prediction_loss_only=True` + `eval_do_concat_batches=False` 缓解；如仍 OOM，调小 `block_size`（默认 2048） |
| 训练 loss 图不显示 | Recharts 要求 `<Line>` 必须直接作为 `<LineChart>` 子节点，不能包 `<>...</>` Fragment |
| `'X' is an invalid keyword argument for Y` | ORM 模型加字段时忘了同步在 `database.py:run_migrations()` 注册迁移 |

## 数据库

- `backend/cdss.db` 是所有页面的真实数据来源（包括文档、抽取知识、训练实验、助手配置）
- 启动时 `run_migrations()` 会幂等地 `ALTER TABLE` 补字段，无 Alembic
- 同时也会把上次未正常关停的本地助手状态从 `running`/`starting` 重置回 `stopped`（避免引用上个 backend 进程已经孤儿化的 vllm 子进程）
- **没有自动备份**：跨机器迁移时建议手动 `scp backend/cdss.db` 或在 `pretrain_runs/` `training_runs/` 同步前导出关键实验

## 数据源说明

- `crawler/*.xlsx`：丁香园/医脉通等爬下来的病例报告，`*_cleaned.xlsx` 是清洗后的文件
- `guideline/**/*.md`：MinerU 解析的临床指南 markdown，按学科子目录组织
- 这两个目录都不入仓库（体量大、版权敏感），跨机器迁移用 `rsync` 单独同步

---

更详细的架构说明、模块边界、惯例约定请参考根目录的 `CLAUDE.md`。
