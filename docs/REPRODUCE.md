# 复查和重跑

推荐先直接阅读results，写报告不需要再次训练。

原环境Python 3.12、PyTorch 2.5.1+cu121，其余依赖见requirements.txt。需按机器平台安装适用CUDA的PyTorch，再安装依赖。首次重跑必须保留重载检查，不能放宽容差来宣称通过。

在仓库根目录下载固定版本基座：

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", local_dir="models/qwen_0.5b", allow_patterns=["*.json", "*.safetensors", "*.txt"])
```

执行已有候选上的评测（输出到新目录，不覆盖已交付结果）：

```sh
python src/eval_rerank.py --model models/qwen_0.5b --adapter adapter --data results/labeled.jsonl --aggregation min --device cuda --max-length 2048 --output reproduced
python src/eval_rerank.py --model models/qwen_0.5b --adapter adapter --data results/labeled.jsonl --aggregation product --device cuda --max-length 2048 --output reproduced
```

Product复用同目录前向评分缓存。跨硬件/软件环境数值可能变化，若重载检查失败应记录并排查，不删除检查。

训练复跑使用新的输出目录：

```sh
python src/train_lora.py --model models/qwen_0.5b --data data/train.jsonl --validation data/validation.jsonl --output reproduced/adapter --device cuda --quantization 4bit --target process --max-length 768
```

正式测试题已消费，不得继续用来调参后再声称独立测试。旧云端诊断结果未作为本仓库最终结果。
