# 对比结果、推理测速与复现

## 1. 直接查看已经完成的对比

[完整结果表](../benchmarks/2026-10-02/generated/COMPARISON.md)包含原三种子精度、
八种子配对差值、标签协议、RF叠加/跨滤波视图诊断、AG1/AG3/AG5和多频输入，
以及19个配置的推理成本。原三种子主表保留，不能将不同实验的样本量或种子数混算。

本次八种子结果：多频减AG3为 **+0.447 pp**，区间 **[-0.067,+0.961]**；
Gong-BiLSTM减Gong-CNN为 **+0.012 pp**，区间 **[-0.274,+0.297]**。
两项区间均跨零。两个Gong模型相对多频参考的未校正配对区间略高于零，详见表格。
单卡RTX 5090、批32的API吞吐量分别为 Gong-CNN **13,789 RF/s**、
Gong-BiLSTM **1,079 RF/s**、AG3参考 **7,421 RF/s**、多频参考 **5,128 RF/s**。
种子标准差不是新地区泛化置信区间；计时轮次也不是训练种子。

## 2. 不需要数据，一条命令复算汇总表

```bash
git clone https://github.com/cangyeone/rfqc-bench.git
cd rfqc-bench
python -m pip install '.[plots]'
python scripts/reproduce_comparisons.py --plots
```

输出位于 `benchmarks/2026-10-02/generated/`：Markdown表格、CSV、精度/速度图及核验报告。
不需要GPU、不下载模型、不训练。脚本核验源文件哈希，并从每个模型/种子的汇总数和
57条原始计时记录重新计算表格。这不等于从逐条RF重新计算精度：原始波形、标签、
台站/样本目录和逐样本预测没有上传。

## 3. 用自己的数据做精度对比

准备符合[输入格式](DATA.md)的 `my_test.npz`。必须包含二值标签、唯一sample_id；
特征类模型还需要六项提供的特征，FCM需要完整台站池和台站标识。
默认评估17个主比较配置、每个原登记的三个种子，不训练、不按测试集重新选阈值。

```bash
python scripts/compare_models.py --input my_test.npz --output outputs/accuracy.json --device cuda:0
```

只比较指定配置：

```bash
python scripts/compare_models.py --input my_test.npz --models gong_cnn gong_cnn_bilstm reference_ag3 reference_multifilter --seeds 20260928 20260929 20260930 --device cuda:0 --output outputs/four_models.json
```

AG1/AG5仅发布种子20260929。比较它们时显式指定 `--models reference_ag1 reference_ag3
reference_ag5 reference_multifilter --seeds 20260929`，并自行提供同一批具有所需滤波视图的记录。
程序不会悄悄为不同模型删除不同的记录。新数据结果不能标记为原论文结果。
输出仅为每次评价的汇总指标；需要本地逐条预测时使用现有 `rfqc-bench predict`。

## 4. 测量单个模型的推理速度

下面命令在一个新进程里测量一轮。默认选择512条记录，10次预热、100次单条调用和
32次批32调用。输出保留所有计时，不含波形、标签或逐样本分数。

```bash
rfqc-bench benchmark --model gong_cnn --seed 20260929 --input my_test.npz --output outputs/gong_timing.json --device cuda:0 --complete-views
```

`--complete-views` 要求从七视图完整记录中选择相同的样本池，用于和多频模型比较。
选择由sample_id的SHA256排序决定，不使用标签或预测。单独测只有AG3的数据时可省略该选项。
记录不足512条时可显式修改 `--sample-size`，其值必须是 `--batch-size` 的整数倍；
这会改变测速方案，应在结果中注明。CPU模型指定 `--device cpu`。

Python调用：

```python
from rfqc_bench import RFData, RFQCPredictor, benchmark_inference

data = RFData.load('my_test.npz')
model = RFQCPredictor.from_pretrained('gong_cnn', seed=20260929, device='cuda:0')
timing = benchmark_inference(model, data, complete_views=True)
print(timing['latency_p50_ms'], timing['batch_records_per_second'])
```

程序化调用测一轮，并记录调用者的PyTorch inter-op线程数。CLI新进程将其设为1。
API总耗时包含预处理、验证和CPU/GPU传输；纯网络执行另外计时。
读盘、加载模型、RF生成、六项提供特征的提取以及HTTP传输不计入，不能称为完整地震数据处理速度。

## 5. 一次测所有模型，每个三轮

```bash
python scripts/benchmark_models.py --input my_test.npz --output outputs/timing_all --device cuda:0
```

每个模型/轮次启动独立进程，顺序由固定哈希排序决定。所有神经网络在指定的同一设备顺序运行，
不让它们同时抢占GPU；LogReg和FCM自动使用CPU。FCM始终处理完整台站池，不抽成512条，
也不输出不适用的单条延迟。结果不能把FCM整池吞吐量当作批32吞吐量混排。

手动恢复时重复同一命令加 `--resume`，程序检查输入、方案和已完成计时文件哈希。
输入、设备或方案改变时应使用新目录。不会设置开机启动，也不会自动重启训练。

模型首次使用时按需下载。离线使用 `--model-root /path/to/bundles`，其中子目录为
`gong_cnn-seed20260929` 等，包含现有 `bundle.json` 和可选 `weights.safetensors`。

## 6. 训练、物理诊断和完整研究复现

所有19个配置均可使用原有 `fit()` / `rfqc-bench train` 接口；见[训练说明](TRAINING.md)。
这些命令需要显式提供训练与验证数据，测试数据不参与拟合、预处理估计或阈值选择。
包内 `rfqc_bench.metrics` 提供标签一致性及台站bootstrap，`rfqc_bench.diagnostics`
提供人工/自动叠加的形态与振幅比较及跨滤波视图关联。

本次同步还保留了论文的原始分析/核验脚本及来源哈希，见
[`reproduction/`](../benchmarks/2026-10-02/reproduction/)。它们依赖完整研究目录和原始运行记录，
不是无数据演示。当前仓库中可无数据运行的是第2节的汇总复算；使用自己的数据应使用第3–5节接口。
原始训练结果来自固定CUDA环境，不能承诺跨设备逐位复现训练轨迹。

目前没有与每条RF对应的射线参数和事件映射，未执行H–κ或结构反演。本次不发布暂缓的H–κ草稿。
数据公开许可和DOI与软件许可证是不同事项；此仓库当前只发布程序、已有模型资产和对比汇总。
