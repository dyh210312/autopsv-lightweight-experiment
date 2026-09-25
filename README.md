# AutoPSV 思路下的轻量过程验证器实验

本仓库保存已完成的轻量化实验及真实负结果，**不是原论文完整复现**。

## 最终结果

200道固定GSM8K测试题，每题5个候选，共1000份答案。

|方法|答对|准确率|
|---|---:|---:|
|Min|68/200|34%|
|Product|67/200|33.5%|
|多数投票|84/200|42%|
|Oracle（至少一个正确候选）|119/200|59.5%|

模型重载检查通过，检查输入保存前后logit均为1.2109375。当前方案未超过多数投票，不能宣称复现论文效果。

## 文件怎么读

- `results/RESULTS.md`：报告用结果说明。
- `results/final_summary.json`：指标与95% Wilson区间。
- `results/per_question.csv`：逐题对比。
- `results/candidates.json`：1000份原始候选。
- `results/evaluation/scored_candidates.json`：逐步评分。
- `data/`：训练、验证和固定测试题。
- `adapter/`：最终本机LoRA权重及训练记录。
- `src/`：训练、评分、提取答案等代码。
- `docs/REPRODUCE.md`：复查结果和重新推理方法。

## 实验设置与限制

Qwen2.5-0.5B-Instruct，4位量化，LoRA r=8、alpha=16、dropout=0.05，q_proj/v_proj。400训练轨迹、5302前缀样本、30更新；最终阈值数据为−0.5。target实际为process，BCE训练，与原论文设定有差异。首步标签按旧实现约定处理，并非人工真值。

生成固定seed=20260914、temperature=0.7、top_p=0.9、最多768新token，每题5候选。训练上下文上限768，正式评测上限2048，保留完整前缀、不静默截断。测试题在评测前经本地历史记录排除并固定；不保证排除基座预训练污染。

152/1000候选无法被固定提取器解析，按错误计；多数投票忽略未解析票，平票按候选ID排序后首次出现者决定。Oracle不是实际模型性能。最终答案正确不等于所有推理步骤正确。

## 资料来源

数据来源：GSM8K（https://github.com/openai/grade-school-math）。模型：Qwen/Qwen2.5-0.5B-Instruct（Hugging Face），revision `7ae557604adf67be50417f59c2c2f167def9a775`。原始数据和模型分别适用其上游许可；本仓库未另行授予整体开源许可。

为便于分享，JSON中的原电脑绝对路径已替换；权重内容和数值结果未改变。原始本地备份保留。未上传账号凭据、云端控制脚本、缓存和基座大模型。
