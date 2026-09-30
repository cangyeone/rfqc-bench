"""Opt-in weight download; all example inputs are generated, not observations."""
from rfqc_bench import RFQCPredictor, synthetic_data

model = RFQCPredictor.from_pretrained('gong_cnn', seed=20260928)
data = synthetic_data(n=4, multiview=False)
print(model.predict(data).to_dict())
