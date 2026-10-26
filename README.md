# Text2SQL 脚本训练项目

训练路线为 Qwen3-8B + QLoRA + SFT，详见 [技术路线](docs/技术路线.md)。提供下载、清洗、专项样本、冻结划分、推理评测和训练脚本，不依赖 LlamaFactory。下载环境与训练环境分开。

第 3～7 步的完整命令、数据协议、评测口径与恢复训练说明见 [训练运行说明](docs/训练运行说明.md)。

从下载到最终验收、交付的完整步骤及产物路径总表见 [微调执行流程](docs/微调执行流程.md)。第 10 步使用 `scripts/compare_reports.py`，阈值模板为 `configs/acceptance.example.json`；未填写的条件不会默认通过。

## 项目结构

```text
.
├── scripts/                      # 下载、数据处理、推理、训练和验收脚本
├── configs/acceptance.example.json # 验收阈值模板
├── tests/                        # 无 GPU 的自动测试与命令行串联测试
├── docs/                         # 项目要求、技术路线与云端执行说明
├── requirements-download.txt     # 下载环境依赖
├── requirements-data.txt         # 数据处理、评测与验收依赖
└── requirements-train.txt        # 云端 GPU 训练与推理依赖
```

仓库只保存代码、配置模板、测试和文档。模型、数据、检查点、日志及报告保存到持久化数据盘（文档默认 `/data/text2sql`），不提交 Git。原始要求见 [项目要求](docs/项目要求.md)。

## 云服务器运行

推荐 Linux、Python 3.11 或 3.12。在项目根目录执行，下载目录应放在云服务器的持久化数据盘上。

```bash
python3.11 -m venv .venv-download
source .venv-download/bin/activate
python -m pip install -r requirements-download.txt

# 默认：ModelScope 上的 Qwen3-8B + 作者完整版 CSpider
python scripts/download_assets.py --output-dir /data/text2sql
```

模型下载的是原始 BF16 权重，约 16.4 GB；4-bit 量化在后续训练加载时进行。建议先预留至少 30 GB 空闲磁盘，后续训练检查点还需额外空间。下载无需 GPU。

可单独下载或更换模型下载源：

```bash
python scripts/download_assets.py --only model --output-dir /data/text2sql
python scripts/download_assets.py --only data --output-dir /data/text2sql

# 网络能访问 Hugging Face 时使用；同一模型目录保持同一下载源和 revision
python scripts/download_assets.py --only model --model-source huggingface --output-dir /data/text2sql

# 下载可选的 SeSQL 中文补充数据
python scripts/download_assets.py --only data --include-sesql --output-dir /data/text2sql
```

需要固定模型版本时传 `--revision`（分支、标签或提交 ID）；默认 ModelScope 为 `master`，Hugging Face 为 `main`。下载信息文件记录请求的版本，移动分支的确切提交由 SDK 本地缓存元数据保留；正式实验请指定固定提交。私有访问凭据通过 SDK 的环境变量或登录机制配置，不写入脚本。

下载中断后重复原命令：模型 SDK 和 gdown 使用各自缓存及临时文件续传。已成功安装的数据会先重新验证，再跳过；不会覆盖已有数据目录。请勿同时向同一输出目录启动多个实例。

## CSpider 网络与文件检查

使用 [作者主页](https://taolusi.github.io/CSpider-explorer/)中 Google Drive 文件夹内的 `full_CSpider.zip`（文件 ID `1ecF1YUhbvFgN6lXzzIQeEwdiuhWj_o32`），不是只含 JSON 的 Paddle 重打包版本。作者包约 110 MB，包含 206 个 SQLite 文件。

云服务器需要能够访问 Google Drive/usercontent。若无法访问，可从作者主页的官方入口自行下载完整版并上传到服务器，再执行：

```bash
python scripts/download_assets.py --only data \
  --cspider-archive /data/uploads/full_CSpider.zip \
  --output-dir /data/text2sql
```

脚本只接受已核验的作者完整版；旧版、网盘包装目录不同或文件不完整会报错。CSpider 和 SeSQL 均核对预先实测的 SHA256。在线下载包保存在 `downloads/`，手动指定的包保持原位置。解压在临时目录进行，ZIP 路径、CRC 和关键数据文件检查通过后才发布目录。失败时原始下载包保留，便于检查；若报告包大小或 SHA256 错误，应移走该损坏包后重新下载。

```text
/data/text2sql/
├── models/Qwen3-8B/          # 权重、tokenizer、配置、SDK 缓存
├── downloads/               # 原始 ZIP 包，便于复核
└── data/raw/
    ├── cspider/
    │   ├── train.json
    │   ├── dev.json
    │   ├── tables.json
    │   ├── database/
    │   ├── test_data/        # 最终测试，不得加入训练
    │   └── test_database/
    └── sesql/               # 仅 --include-sesql 时下载
        ├── tables.json
        ├── db_content.json
        └── single-round-question-completed/
```

模型检查包括 JSON 配置、索引引用的分片、safetensors 头与张量数据长度；不加载模型，不读取全部张量做数值验证。CSpider 检查每个 split 的问题/SQL、Schema 引用和对应 SQLite 文件头；这不是 SQL 标注质量审核或完整数据库 integrity_check。`download_info.json` 记录下载来源、归档 SHA256 或模型分片统计。

SeSQL 使用作者 GitHub 的固定提交版本，并验证预先核实的 SHA256。它提供 JSON 表内容，尚需后续转换才能在 SQLite 上执行，不能将其直接视作现成 SQLite 数据库。CSpider 数据按作者 CC BY-SA 4.0 条款使用；SeSQL 保留原始发布文件，数据使用前核对原项目条款。

## 本地自检

自检仅使用临时小文件、SQLite 测试库和模拟 SDK，不下载模型或数据。运行全套测试前安装轻量数据依赖：

```bash
python -m pip install -r requirements-data.txt
python -m unittest discover -s tests -v
python scripts/download_assets.py --help
```

下载依赖与训练环境分开。下载脚本不会清洗数据、重划分训练集或启动训练；后续步骤由独立脚本执行。

## 验证记录

2026-10-24，本地 Python 3.12：8 项自检通过；实测下载并验证作者 CSpider 完整包及 SeSQL 包，重复运行通过。该 CSpider 包包含 train 8659 / dev 1034 / test 2147 条记录，引用 206 个 SQLite 文件；SeSQL 单轮版为 train 21454 / dev 2279 / test 3279 条。保留包内版本，不将这些数量套给其他发行版。

ModelScope 已实测仅下载 Qwen3-8B 的 config.json；没有下载完整权重。Hugging Face SDK 可导入，但本地网络的配置下载发生 TLS 连接错误，未完成在线验证。云服务器网络与完整权重下载需在实际运行时确认。

2026-10-26：28 项自动测试通过，覆盖数据处理、评测和验收的命令行串联、完整性与安全检查；已使用真实 Qwen tokenizer 验证非思考模板。完整模型推理、CUDA 训练、显存与 GPU 断点恢复仍需云服务器验证。
