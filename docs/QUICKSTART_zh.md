# 中文快速使用

这是论文算法的独立 Python 软件包，包含 19 个配置、53 份已训练权重的下载索引。
不包含真实波形、人工标签、数据缓存或逐条测试预测，也不会自动下载训练数据。

## 安装

需要 Python 3.10 或以上。建议先创建虚拟环境；下面命令可直接通过 pip 安装发布的 wheel，不需要 Git：

```bash
python -m pip install https://github.com/cangyeone/rfqc-bench/releases/download/v0.1.0/rfqc_bench-0.1.0-py3-none-any.whl
rfqc-bench doctor
rfqc-bench demo --model gong_cnn
```

最后一步会下载一份已训练模型，对程序生成的合成波形做预测，只用于检查安装。
要启用 HTTP API，可安装带 api 扩展的版本：

```bash
python -m pip install "rfqc-bench[api] @ git+https://github.com/cangyeone/rfqc-bench.git@v0.1.0"
rfqc-bench serve --model gong_cnn
```

打开 `http://127.0.0.1:8000/docs` 即可交互调用。当前不宣称已在 PyPI 发布；
以上两种都是实际可用的 pip 安装方式。仓库提供了后续 PyPI 发布流程。

## Python 调用

```python
import numpy as np
from rfqc_bench import RFQCPredictor

# raw_rf: (记录数, 频段数, 501)，保持原始幅值；同一记录不同频段必须对应同一事件。
raw_rf = np.load("my_waveforms.npy", allow_pickle=False)
model = RFQCPredictor.from_pretrained("reference_multifilter", device="cpu")
result = model.predict(waveforms=raw_rf, gaussians=[1, 1.5, 2, 2.5, 3, 4, 5])
print(result.p_good)       # good 类分数，并非已经校准的物理可信度
print(result.prediction)   # 1=good，0=bad，使用模型保存的验证集阈值
```

单频模型可输入 `(N,501)` 数组，默认 AG3；若使用 AG1 或 AG5，必须显式给出系数。
多频缺 AG5 时只提供已有的六个频段。不同记录频段数不同可用补零数组及 `lengths`；
有效频段必须按高斯系数升序排列，补零部分不会送入参考编码器。

模型只接受 501 点、−10 到 40 秒、dt=0.1 秒的物理时间窗；这些 AG 系数不是 Hz。
特征模型需另给六个原始描述量，顺序见 [数据接口](DATA.md)，程序不会冒充原作者的特征生成方法。
Xiong-FCM 需要同一台站完整输入集合，拆开调用可能改变站内相关性和预测。

## 文件预测、训练和续训

```bash
rfqc-bench predict --model gong_cnn_bilstm --input my_rf.npz --output predictions.csv
rfqc-bench train --model reference_multifilter --train train.npz --validation validation.npz --output runs/reference --device cuda:0
```

中断后重复训练命令并加 `--resume`。默认最多 50 轮，验证 AP 连续 10 轮不提升则早停；
阈值只由验证集确定。两块 GPU 可分别开终端使用 `cuda:0`、`cuda:1` 训练不同任务，
输出目录必须不同。本包不修改已完成的论文实验，也不安装开机启动服务。

默认选最早登记的种子，不按测试精度挑模型。17 个主配置各提供三个种子；
AG1/AG5 仅提供已完成的 20260929。`from_pretrained` 指已完成 RF 训练的权重，
不表示论文采用了相位拾取迁移学习。权重首次下载后可以离线加载。
