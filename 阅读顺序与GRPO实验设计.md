# 阅读顺序与GRPO定向实验

> 本文保留阅读路线和实施前计划。实际单卡已完成，各组 476 题、batch size 8、476 次更新（2 epoch），终评 batch size 16。[实际配置、命令和结果](历史结果/grpo_screening_4090/README.md)

## 待审查的最终实验方案

本节替代下文旧顺序中的「首轮 baseline vs token_norm」。第一轮优先检验用户提出的假设：专项 SFT 可能让一些 RL 题稳定全对，使二元奖励很少提供组内对比。两组训练共同使用 `--arm baseline`，归一化改动留作独立后续实验。

|环节|实施|看什么|
|---|---|---|
|起点诊断|专项 SFT42 对 RL1000 每题采 4 次。可选同规模官方 Qwen3-1.7B 在预先固定的 100 道同题上采 4 次|按题记录答对 0～4 次的分布，重点看 Instruct 混合→专项 SFT 全对，以及反向与混合→全错。并列报告格式失败及截断|
|筛题对照|A 从 RL1000 均匀不放回抽 N 题，B 使用所有答对 1～3 次的 N 题。两份数据由同一次筛题生成，训练起点都为同一 Base + SFT42|训练时重新在线采样。比较 A/B 的混合组比例、生成 token 数与更新情况。随机组允许抽中部分正确题，报告重叠数|
|开发评测|SFT 起点、A 和 B 都在原 dev500 上用 greedy512 与同一严格评分器|主比较为第 200 步 B−A 的正确题数和逐题 gained/lost。同时比较 B−SFT，判断有无净收益|

诊断固定使用同一份原始题面、G4、temperature 0.8、top-p 0.95、关闭 top-k、256 token 上限。种子按题固定。官方 Qwen3-1.7B 通常走聊天模板，并支持思考模式。本对照采用项目原始提示词，结论限定于这套输入与预算。官方 1.7B 的配方包含蒸馏，A/B 起点诊断无法隔离「通用 SFT 数据覆盖面」这个因果变量。真正隔离它需要从同一个 Base 自己训练通用与专项 SFT，并匹配训练预算。[官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B) · [Qwen3 技术报告 §4.5](https://arxiv.org/html/2505.09388v1#S4.SS5)

单卡 RTX 4090 24 GB 上，两组固定 G4、KL 0.04、学习率 1e-6、每步 12 条回答、200 步，默认训练 seed42。每次生成一题的 4 条回答，一次反传处理这 4 条，梯度累计 3 次后更新。相比一次生成 12 条，这样减少一次生成所需的显存。两组名义上各 600 次题目采样和 2,400 条回答，生成 token 数与实际题目覆盖另行记录。保留 BF16、梯度检查点和 `bnpo`。`bnpo` 按当前微批的有效 token 数归一化，因此从微批 1 改成 4 也改变了 loss 的归一化范围。A、B 两组使用完全相同的微批配置，历史 GRPO 数字只作参考。采样统一 temperature 0.8、top-p 0.95、top-k 关闭、回答最多 256 tokens。筛题标签来自一次 G4，可能受采样噪声影响，所以要看训练中的混合组比例是否仍更高。若 N 很小或两组重叠很高，先审查重复覆盖与对照区分度，再决定是否运行完整 200 步。后续如测 DAPO 的 token 归一化，先重新匹配生成批次。本脚本的 DAPO 分支保留单次 12 条，因为其分母依赖生成批次。

- **B 比 A 多答对，且超过 SFT**，才观察到本轮开发集的净提升。先记录逐题变化，再固定方案换训练 seed123 复跑，沿用相同筛题集以单独检验训练种子影响。
- **B 比 A 好，但仍低于 SFT**，说明筛题减轻了本轮退化。若 B 训练中混合组更多、dev 却没提高，则组内信号增加没有转化为开发集收益。
- 若后续要测 `token_norm`，用同一份已经冻结的问题集只换 `--arm`。本轮决策只看开发集，历史官方 test 的曝光状态保持原样。

上云命令从项目根目录运行。`MODEL`、`SFT`、`INSTRUCT` 指向云机上的完整 Base、原 SFT42 adapter、同规模官方后训练权重目录。准备 Python 3.11 和支持 CUDA 的 PyTorch，例如原项目使用的 2.6.0/cu124，然后安装 `requirements.txt`。先确认 `nvidia-smi` 能看到 4090、PyTorch 能用 CUDA BF16。没有官方后训练权重时可以先完成两组主实验。

```bash
# 单卡只生成和评分。若要同时跑固定 100 题的起点诊断，加上 --instruct "$INSTRUCT"
export CUDA_VISIBLE_DEVICES=0
python train_grpo_experiment.py --screen --model "$MODEL" --sft "$SFT" --output outputs/final_grpo/screen42

# 核对筛题结果及训练计划后，两组各跑 5 步短检，检查 loss、梯度、显存和 checkpoint
for pool in random informative; do
  DATA="outputs/final_grpo/screen42/$pool.jsonl"
  python train_grpo_experiment.py --data "$DATA" --plan --world-size 1
  python train_grpo_experiment.py --model "$MODEL" --sft "$SFT" --data "$DATA" --output "outputs/final_grpo/smoke-$pool" --steps 5 --save-steps 5
done

# 短检确认梯度、显存和 checkpoint 后，正式训练各 200 步
for pool in random informative; do
  python train_grpo_experiment.py --arm baseline --model "$MODEL" --sft "$SFT" --data "outputs/final_grpo/screen42/$pool.jsonl" --output "outputs/final_grpo/${pool}42" --seed 42
  CUDA_VISIBLE_DEVICES=0 python train_grpo_experiment.py --eval-dev --model "$MODEL" --sft "$SFT" --grpo "outputs/final_grpo/${pool}42/final_adapter" --output "outputs/final_grpo/eval/$pool.jsonl"
done
CUDA_VISIBLE_DEVICES=0 python train_grpo_experiment.py --eval-dev --model "$MODEL" --sft "$SFT" --output outputs/final_grpo/eval/sft.jsonl
python evaluate.py --before outputs/final_grpo/eval/random.jsonl --after outputs/final_grpo/eval/informative.jsonl --output outputs/final_grpo/eval/screening_effect.json
python evaluate.py --before outputs/final_grpo/eval/sft.jsonl --after outputs/final_grpo/eval/informative.jsonl --output outputs/final_grpo/eval/net_vs_sft.json
```

筛题产物 `screen_summary.json` 记录 N 和重叠，`paired_probe.json` 记录可选的起点诊断，`manifest.json` 锁定源数据、两份训练集与筛题时的 Base/SFT 指纹。现有 GRPO 与评测入口保持原用法。实验审查、真实 GPU 短检与正式训练均待执行。

---

下文保存上一轮阅读路线、历史诊断及归一化实验备选。涉及执行优先级时以上方最终方案为准。

本轮只新增 `train_grpo_experiment.py`、`test_grpo_experiment.py` 和本文。原 `train.py/common.py/data.py/sample.py/evaluate.py`、README、历史证据都不修改。当前实际路径是 **E:/qwen_gsm8k**。

## 先按数据流阅读，最后才读训练框架

不要先从头啃完common.py或TRL源码。先看一条真实数据，读入口main，再追它调用的少数函数。

|顺序|文件与位置|这一遍只弄清楚什么|
|---|---|---|
|1|README.md的文件树、架构图和V1/V2结论|Base→SFT→GRPO→评测，已有结果是什么，哪些新实验还没做|
|2|prompt.txt；data/rl.jsonl、data/sft.jsonl各一行|模型看到prompt；SFT另有completion；gold用于监督/奖励，不拼进生成输入|
|3|data.py：先main，再build|固定train来源→按split_manifest选ID→清洗答案→SFT2000/RL1000/dev500；不重随机切分|
|4|common.py：extract_answer→parse_reference→correctness_reward→build_result|一段输出如何被判正确、错误或格式失败；暂时跳过模型加载和文件指纹|
|5|train.py：main，再build_trainer的sft分支|读prompt/completion→Base→LoRA→completion-only loss→SFTTrainer→保存adapter|
|6|sample.py：main，再select|冻结SFT每题抽4次，保留答对1–3次的原RL题，并取同数量随机对照|
|7|train.py的grpo分支；回看common.load_model|合并同一SFT→新建GRPO LoRA→在线采样→严格正确性reward→更新；G4与全局batch12是什么|
|8|evaluate.py：生成流程与paired|相同dev、greedy512、原始逐题预测、gained/lost；训练reward不是正式正确率|
|9|test_core.py|用小反例检查严格格式、数据隔离、筛题和配对是否理解正确|
|10|train_grpo_experiment.py：ARMS→arm_config→main→make_trainer|先比较配置差异，再看新训练流程；最后才读DiagnosticTrainer的跨rank统计|

第一条RL记录目前是ID3392、gold为16。可以顺着它想：prompt是题目与固定模板；生成`Final answer: 16`得到1，`Final answer: 20`得到0，只有“答案是16”也得到0，因为不满足统一输出协议。

一个G4组若是`[1,1,1,0]`，组均值为0.75，去均值后的优势是`[0.25,0.25,0.25,-0.75]`。原配置还会除以组内标准差；`no_std`只去掉这一步，不会让全对/全错组凭空产生相对正确性信号。KL项仍可能有梯度，不能把“全同分”等同“所有参数一定不更新”。

阅读时每次只串清楚“输入是什么→做了什么→输出到哪里→如何知道没有出错”。第一遍不需要理解SwanLab、云打包、旧平台命令或历史备份。

## 先回到本项目的证据

|实际证据|能支持的判断|不能推出的结论|
|---|---|---|
|V1：SFT42为902/1319，GRPO42为893；SFT123为897，GRPO123为882|两种子在同一official test上均退化；dev各+0.40pp未泛化|不能凭这些差异断言退化机制、普遍方法优劣或统计显著|
|历史50题G4预采样：20组mixed、26组全对、4组全错|启动时该小样本60%组没有组内二元正确性差异|不能外推整个训练始终60%无信号，更不能把它叫唯一原因|
|200条预采样：格式合法199条；有害截断1条|格式奖励、长度加倍不应无证据地成为第一轮主要干预|不能保证在线策略训练后仍然几乎不截断|
|当前代码明确设bnpo、每卡microbatch=1、G4、beta=.04、256 tokens|可以定位具体loss归一化、采样和KL设置|不能只凭配置猜测模型一定长度漂移或熵坍缩|
|V2 G8比G4 test净多9题，但dev净少6题，且仍低于SFT42|单纯增大G没有形成稳定增益证据|不能把耗时较短当作严格加速实验|

原始位置：`历史结果/reports/day11/grpo_go_no_go.md`、`历史结果/results/group_stats.json`、`历史结果/reports/final/verification.json`、`历史结果/reports/v2/research_report.md`。

### 本轮重新做了逐题配对复核

这次实际读取四份1319行的SFT/GRPO原始预测，对齐ID和reference后重新统计，而不是只摘README的总分：

|seed|改善题数|退化题数|退化后为数值错误|退化后为格式失败|退化题中被截断|SFT→GRPO平均输出tokens|
|---|---:|---:|---:|---:|---:|---:|
|42|36|45|45|0|0|90.81→91.90|
|123|32|47|43|4|2|93.14→94.40|

两seed长度中位数分别维持79与82。也就是说，**seed42原本正确却丢失的45题全部是格式合法的数值错误**；不是补格式奖励就能直接解决。seed123也主要是原本会做的题变成数值错误。总体平均长度只增加约1token，不支持“已观测到明显长度膨胀”的说法。

因此本轮优先检验优化目标/梯度权重和策略漂移，而不把格式或截断当作主要已知根因。token归一化是由源码定位的可测机制假设；即使没有明显平均长度变化，token/题目的梯度权重仍可能不同，但这不是已被证明的退化原因。以上分析来自已经曝光的历史test，只作回顾性诊断，不能伪装为独立验证。

## 具体参考了哪些源头

### 1. Dr. GRPO：拆开长度与难度归一化

[原作者项目](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87)及[实际训练源码](https://github.com/sail-sg/understand-r1-zero/blob/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87/train_zero_math.py)中，`ZeroMathLearner`分别实现固定常数分母，以及去掉组内标准差缩放的MC优势。

在本项目中用TRL现成的`loss_type='dr_grpo'`、`scale_rewards='none'`适配这两点，不引入Oat框架，不重写优化器。保留本项目SFT起点、模板、二元严格奖励、KL=.04和G4。原作者的模型、数据、KL及框架与这里不同，因此这是思想适配，不能称完整论文复现。

重要限制：改常数分母、去标准差也会改变有效梯度尺度；同一个名义学习率不等于相同更新幅度。即使Dr分支更好，也不能自动把原因归到“去偏”，还需看实际更新、KL、长度和学习率对照。

### 2. DAPO：全局token归一化与有效组诊断

[作者仓库](https://github.com/BytedTsinghua-SIA/DAPO/tree/33fe3176f0bb212588e84fc8ccf50dd554975144)提出token级策略损失与动态采样等组合。其旧verl训练分支链接当前未能读取，不假装已经审查该失效路径；本轮实际落地和逐行核对的是[固定TRL0.24实现](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py#L1730)。

这个源码有一个与本项目直接相关的区别：

- `bnpo`：用当前本地microbatch的有效token数归一化。
- `dapo`：用全局累积batch的有效token数归一化。
- 当前microbatch=1时，bnpo等价于按单条回答长度平均；不是仅把配置名改成bnpo就消除了长度权重差异。这一点也在[对应版本GRPOConfig文档](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py#L184)明确说明。

因此第一轮优先只改`loss_type='dapo'`，其余不动。新增统计完整跨rank组的全对/全错/mixed比例，以及正确/错误回答长度、格式失败和截断；原TRL继续记录reward_std、KL、entropy、clip ratio。

这里只适配全局token归一化与诊断，没有同时加入Clip-Higher、在线重抽补齐或soft-overlong reward，不能将本入口命名成“完整DAPO”。原`sample.py`是固定SFT的**离线**筛题，不是DAPO随策略变化的在线动态采样；在文档和结果中严格分开。

### 3. Open-R1：分开观察正确性与格式

[Open-R1固定版本rewards.py](https://github.com/huggingface/open-r1/blob/5b6ff22b3fb7aa069c54866e517f39dfc3160e09/src/open_r1/rewards.py)将accuracy/format等功能拆开。这里借鉴可解释的分项诊断，继续使用本项目原严格答案协议；不搬`<think>/<answer>`模板，不奖励步骤数/关键词，也不额外给“错但格式合法”奖励。

这样避免格式指标已经接近饱和时，通过新格式reward制造一个难以解释的混合干预。

本机安装的TRL训练源码Git blob为`352a0144ef9c4c02033cf57b8452ad19efaed6d1`，与官方v0.24.0返回的源文件blob一致。私有统计hook只支持这个版本，未随意套用新版本参数。

## 新入口与实验分支

新增文件`train_grpo_experiment.py`复用原项目的数据、模型加载、SFT合并和layout，另建实验Trainer。不是在原train.py上反复改参数。`--plan`只读数据与打印配置，不导入大模型依赖、不训练。

|arm|loss_type|scale_rewards|学习率|解释|
|---|---|---|---:|---|
|baseline|bnpo|group|1e-6|必须重新跑的同条件控制，不能拿历史数字代替|
|no_std|bnpo|none|1e-6|只去掉组内标准差缩放|
|token_norm|dapo|group|1e-6|只改全局累积token归一化，首选对照|
|token_no_std|dapo|none|1e-6|与前3组构成2×2归一化对照|
|dr_grpo|dr_grpo|none|1e-6|固定常数分母+不除组内std；与token_no_std比较分母方式|
|low_lr|bnpo|group|3e-7|降低更新幅度的保守控制，不是精确梯度匹配|

其他共同条件：冻结SFT42 adapter、同一Base revision、原RL1000、seed42、G4、temperature=.8、top_p=.95、最大256 tokens、beta=.04、全局completion batch12、默认200次更新。单机1/2/3卡均使generation batch=12；三卡每卡batch1、梯度累积4，每update三题。

每25步保存一次完整checkpoint，不自动删。200步最多8个checkpoint/组，正式运行时记录磁盘与token用量。原训练入口此前默认50步保存，此处只改变记录粒度，作为两组共同设置。长度/optimizer/lr schedule等也要固定记录，不能只给表现差的一组延长训练。

新入口要求原RL1000或已完整生成、按源行逐条一致的`sample.py`子集。拒绝dev/test、改写prompt/gold、重复ID、未完成筛题和改动过的筛题证据。每个generation batch至少需要三题。

共同停止条件：若某个全局generation batch的回答全部截断且被mask，已没有有效loss token，入口会先保留诊断再停止，避免dapo分母为0导致NaN。不会偷偷为某组重抽或增加长度；这与“全错但正常结束、仍可能有KL梯度”是不同情况。

## 先跑哪两组

1. **第一轮仅baseline vs token_norm**，均用RL1000、同一个SFT42与seed42，先做短跑，再各200步。比较25/50/100/200步的轨迹；不要同时加长度、G、格式奖励或改KL。
2. 若需要区分难度归一化，再补no_std、token_no_std，完成2×2对照。若怀疑只是更新幅度变小，再看low_lr；dr_grpo是后续分母对照，不先把六组全部上云。
3. 数据实验单独进行：原sample.py得到N题informative与N题random，固定训练方法与所有条件比较二者。不能将RL1000与N题筛选直接比较后把效果全归到筛题。
4. 最后对两组配对重复GRPO seed123，仍使用同一SFT42。它是GRPO采样/训练种子重复，不是独立SFT种子。保留所有结果，不能只报告最好的seed。

先看相同dev的correct/n和逐题改善/退化，再看mixed组、正确/错误长度、截断、KL、entropy和clip ratio。训练reward上升而dev下降不算成功。`low_lr`和Dr分支的有效梯度尺度不同，必须保留这条混杂说明。

如果只有一次seed42上的小幅提升，结论写单次探索性改善，不称稳定涨点。当前dev和official test均有历史使用记录，本轮不使用official test选方案；真正独立泛化验证需要另行明确并冻结，不能把旧test重新命名成未见测试。

## 本地验证与原文件保护

本轮已通过：

- 26项原有与新增测试；覆盖分支差异、严格奖励、全局组/ID、数据来源隔离、有效子集与格式边界。
- 六个arm的真实CPU微型随机Qwen3训练：使用合成二元奖励作**梯度探针**，每组LoRA均发生更新，冻结基座不变；保存、重载、生成通过。不把探针奖励称数学正确率。
- 同seed、同optimizer的微型单步基线，与原build_trainer产生的LoRA参数逐元素相同。这只是该受控CPU案例，不能外推所有模型/多卡/恢复路径的完全等价。
- 完整checkpoint包含optimizer/scheduler/RNG/Trainer状态，实际从step1恢复运行至step2；不承诺跨中断逐token重放。
- 三进程CPU Gloo测试：全局G4组的ID一致、dapo token归一化反向运行、三rank LoRA指纹一致。Windows torchrun直接启动先遇到本机构建缺libuv，改用测试专用FileStore完成该CPU测试；没有修改环境库或生产代码来绕过云端问题。
- 新评测写盘流程、逐行预测、run/summary通过微型fixture验证；不修改真实dev数据。

实际发现一个旧parser边界：`Final answer: ,`可能被原正则匹配后触发Decimal异常。原评分器保持原样；新实验入口在训练和`--eval-dev`中将这类非法数字记0/format_fail。正常格式仍逐字复用原严格规则，未放宽答案标准。两组及SFT基线统一使用新`--eval-dev`，避免一边异常退出、一边继续计数。

原common.provenance会扫描根目录Python文件，因此新增文件会进入**新run**指纹。不要将新增入口拿去续接历史run目录，不能承诺“文件没改”就能恢复旧实验。每组开全新目录，续训仍要求该组配置/代码/资产一致。

### 验证边界

本机为Windows、Python3.12、Torch2.14.0+cu126、TRL0.24.0；计划云端为Linux、Python3.11、Torch2.6.0/cu124及原requirements.txt。CPU与3进程Gloo通过，不等于真实1.7B、BF16显存、CUDA/NCCL、三卡吞吐或数学提升通过。

**当前结论：本地实现与验证完成，可以准备云端工程检查；尚不能宣布“一切云端问题都已排除”，也没有新模型指标。** 本轮没有租机、下载Base、启动云训练、提交或推送Git。

## 下一步命令（尚未执行云训练）

Windows本地从E:/qwen_gsm8k运行，只打印计划：

```powershell
& ./.venv/Scripts/python.exe train_grpo_experiment.py --arm baseline --world-size 3 --plan
& ./.venv/Scripts/python.exe train_grpo_experiment.py --arm token_norm --world-size 3 --plan
& ./.venv/Scripts/python.exe -m pytest test_core.py test_grpo_experiment.py -q
```

未来Linux云机：上传**当前整个源码目录的新文件**、data/与SFT42 adapter；当前GitHub尚没有这次新增文件，不能直接clone后假定存在。按原requirements.txt建立环境，准备固定来源Base，保存环境与下载记录后：

```bash
export MODEL=/root/autodl-tmp/models/Qwen3-1.7B-Base
export SFT=outputs/sft_full/final_adapter

# 先3卡真实构建，随后5步工程短跑；两组都检查
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm baseline --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/inspect-base --inspect-only
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm token_norm --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/inspect-token --inspect-only
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm baseline --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/gpu-smoke-base --steps 5 --save-steps 5
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm token_norm --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/gpu-smoke-token --steps 5 --save-steps 5
```

这一步应实际检查：Base来源与配置/权重、相同SFT指纹、可训练参数仅新LoRA、非零梯度与参数变化、ref为合并SFT、全局G4分组、损失有限、显存、checkpoint重载、生成能EOS结束。当前构建只凭架构config不能证明Base来源；run会记录实际权重指纹，仍需固定下载日志。工程检查通过后才开始下面正式对照，不将5步smoke当效果实验。

```bash
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm baseline --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/baseline42 --steps 200 --seed 42
torchrun --standalone --nproc_per_node=3 train_grpo_experiment.py --arm token_norm --world-size 3 --model "$MODEL" --sft "$SFT" --output outputs/targeted/token42 --steps 200 --seed 42

# SFT和两实验组统一greedy512评测；先看整套同条件结果
CUDA_VISIBLE_DEVICES=0 python train_grpo_experiment.py --eval-dev --model "$MODEL" --sft "$SFT" --output outputs/targeted/eval/sft.jsonl
CUDA_VISIBLE_DEVICES=0 python train_grpo_experiment.py --eval-dev --model "$MODEL" --sft "$SFT" --grpo outputs/targeted/baseline42/final_adapter --output outputs/targeted/eval/base.jsonl
CUDA_VISIBLE_DEVICES=0 python train_grpo_experiment.py --eval-dev --model "$MODEL" --sft "$SFT" --grpo outputs/targeted/token42/final_adapter --output outputs/targeted/eval/token.jsonl
python evaluate.py --before outputs/targeted/eval/base.jsonl --after outputs/targeted/eval/token.jsonl --output outputs/targeted/eval/paired.json
```

过程轨迹可以把同样命令的`--grpo`换为各自checkpoint-25/50/100/200，输出不同文件名。两组按同一预设步数比较，不因看到某一组效果不好就单独改变长度/数据/训练步数；使用dev挑选checkpoint必须明确写为开发集选择，不是独立泛化结论。

输出包括原Trainer指标、`online_diagnostics.jsonl`、`run.json`、每25步checkpoint、`final_adapter`、最终metrics与日志。每次generation batch的统计是在当前update之前采集，不能误当成独立dev评测。
