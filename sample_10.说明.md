# 10 条测试样本

`sample_10.jsonl` 每行是一条用户提问和模型回答，连同原样本示例中的请求、token、业务和反馈字段。`msg-0004` 是请求失败，没有 assistant 回复；`msg-0009` 是空回复。其他样本用于验证点赞、点踩、重新生成、慢请求和长输入等信号。

从最初的 10 条单轮样本恢复，并修正了几处混用了不同业务的 `sessionId`。`msg-0005/0006` 改成相同的对照测速问题：前者忽略证据、被点踩，后者依据证据改正、被点赞，供纠错配对测试。财务、教育和缓存样本使用各自的会话 ID。原始文件保存在 `samples_backup/sample_10.single_turn.jsonl`。

这些示例的对话、反馈和性能数值是人工测试数据。运行 `python flywheel.py sample_10.jsonl --output output/processed_10.jsonl`，结果会写到 `output/processed_10.jsonl` 和 `output/processed_10.summary.json`。
