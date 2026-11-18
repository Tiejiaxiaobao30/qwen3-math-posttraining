# 历史实验结果

这里保留V1/V2原始预测、报告、冻结清单与来源记录，不保留重复训练入口。

- [V1最终报告](reports/final/official_test_report.md)
- [V2报告](reports/v2/research_report.md)
- [推理服务测量](reports/final/vllm_benchmark.md)
- 原源码与相对路径在[原Git版本](https://github.com/Tiejiaxiaobao30/qwen3-math-posttraining/tree/100b7ce3dae460e58b52a2e0d1d706ffb01c7e4f)中完整保留。报告内的旧源码相对路径请在此版本查看。

在新项目根目录执行 `python evaluate.py --audit-history`，会在临时目录读取该版本、运行原V1/V2核验，并清理临时文件。需要完整Git历史；不要用不含该提交的浅克隆。
