# 交付验证记录

日期：2026-11-10；执行者：助手。本轮没有租 GPU，没有执行任何模型 `train`、`generate`、forward、backward 或 optimizer step，也没有创建 SwanLab 云端实验。

## 已执行

- 阅读原仓库 `scripts/train_grpo_smoke.py`、奖励/冻结评分器、配置、SFT adapter 配置、历史报告和当前 Notion 计划。确认旧训练接线只使用 correctness，未将 format reward 加入总奖励。
- 从远程 main `100b7ce3dae460e58b52a2e0d1d706ffb01c7e4f` 创建独立工作目录及 `feat/reward-restart-autodl` 分支；旧本地目录、`environment-backup.txt` 和历史结果保持原状。
- 查询 GitHub 并阅读固定 TRL 0.24.0 `GRPOConfig` / `GRPOTrainer`、Open-R1 rewards；核对 generation_batch、跨卡奖励 gather、LoRA 禁用后的 reference 行为和 checkpoint 入口。
- 从 HF 固定 revision 实际下载 GSM8K TRAIN parquet，7473 条。转成本地原序 JSONL，执行 `python -m restart.prepare_data --source local --raw-jsonl ...`，重建 RL1000/dev500。两集合内容/顺序/答案与旧本地实际数据规范化指纹一致；跨 SFT/RL/dev 题目精确规范化重复检查通过。
- `python -m unittest discover -s restart/tests -v`：6 项测试通过，含多个奖励攻击反例、错误 gold 对齐、1/2/3 卡有效 batch、跨 split 重复、逐题比较与唯一奖励干预。
- 新代码 `compileall` 和两个 Bash 脚本 `bash -n` 通过；`git diff --check` 通过。
- 在独立 `E:/qwen/restart-check-env` 中安装固定直接依赖；`pip check` 通过，未修改原 nlp 环境。此检查环境为 Windows/Python 3.12/已有 PyTorch 2.14.0+cu126，与目标 Linux/Python 3.11/PyTorch 2.6.0+cu124 不同。
- 实际 import GRPOConfig/GRPOTrainer、Qwen3ForCausalLM、PEFT、SwanLabCallback 成功；CPU 构建 GRPOConfig，generation_batch_size=12 通过。SwanLabCallback 仅构造，未初始化云端 run。
- 用随机初始化的极小 Qwen3 构建 CPU GRPOTrainer + LoRA + 4 行合成 Dataset；直接调用 reward hook 验证 Dataset 的 gold 传递及 reward gather，返回 `[[1,1],[0,1],[0,0],[1,1]]`；可训练参数全为 LoRA、reference 使用禁用新 adapter 的路径。没有运行模型 forward、生成或训练；不是训练 smoke，也不是历史 1.7B 资产加载验证。
- `python -m scripts.audit_final_evaluation` 通过，旧六组 official / 四组 dev 的已存证据与配对计数仍可审核；原历史证据没有被改写。
- SFT 权重打包 `E:/qwen/restart-assets/qwen3-sft-final-adapter.zip`，ZIP 完整性检查通过，解压内容中的权重与原文件字节一致；不含 `training_args.bin`，没有上传 Git 权重。ZIP SHA256：`47d9e2b9b1bd8cf2f5afd7cf56add281120d7ab99e373292a711e3f804b5fdec`。

## 尚未验证

AutoDL Linux 安装、驱动兼容、固定 ModelScope revision 的实际大模型下载、真实 Base+SFT 加载/merge、GPU 显存、速度、三卡 NCCL/DDP、GPU 保存/恢复以及 SwanLab 云端曲线上传，均未运行。执行 README 的 inspect → 20 步 → 两组对照 → greedy dev 评测，保留实际日志后才能更新结论。

没有新训练结果，没有新的独立 held-out 评测，没有稳定提升/显著性结论。格式辅助奖励是待检验假设；不从历史聚合错误率推断退化因果，也不把旧 test 再称为未见测试。
