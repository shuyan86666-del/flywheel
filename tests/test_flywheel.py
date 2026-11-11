import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from flywheel import content_fingerprint, judge_with_api, prepare, run, select_increment


ROOT = Path(__file__).resolve().parents[1]


class FlywheelTest(unittest.TestCase):
    def test_ten_samples_pass_all_five_operators(self):
        rows = [json.loads(line) for line in (ROOT / "sample_10.jsonl").read_text(encoding="utf-8").splitlines()]
        with patch.dict(os.environ, {"JUDGE_API_URL": "", "JUDGE_MODEL": ""}):
            results = run(rows)
        by_id = {row["messageId"]: row for row in results}
        self.assertEqual(len(results), 10)
        self.assertTrue(all(len(row["conversations"]) <= 2 and "turnMetrics" not in row for row in results))
        self.assertTrue(all(all(key in row for key in ("prep", "effect", "hardCase", "candidate", "increment")) for row in results))
        self.assertTrue(all(set(row["effect"]) == {"service", "performanceSignals", "contentRuleFindings", "contentReview"} for row in results))
        self.assertEqual(by_id["msg-0004"]["effect"]["service"], "failed")
        self.assertEqual(by_id["msg-0009"]["effect"]["service"], "empty")
        self.assertEqual(by_id["msg-0005"]["candidate"]["status"], "合格")
        self.assertEqual(by_id["msg-0005"]["candidate"]["useAs"], "hard_case")
        self.assertEqual(by_id["msg-0005"]["increment"]["pairedWith"], "msg-0006")
        self.assertEqual(by_id["msg-0006"]["candidate"]["useAs"], "positive_answer")
        self.assertIn("slow_total", by_id["msg-0008"]["effect"]["performanceSignals"])
        self.assertEqual(by_id["msg-0007"]["candidate"]["status"], "待核验")
        self.assertIn("unsupported_result_claim", by_id["msg-0007"]["effect"]["contentRuleFindings"])
        self.assertEqual(sum(row["increment"]["selected"] for row in results), 6)

    def test_preparation_redacts_and_deduplicates_content(self):
        base = {
            "sessionId": "s1", "messageId": "m1", "success": True,
            "inputTokens": 30, "outputTokens": 30,
            "conversations": [
                {"from": "user", "value": "联系 13800138000，邮箱 qa@example.com，姓名：测试甲。请回答。"},
                {"from": "assistant", "value": "收到，这是一个足够长的测试回答内容。"},
            ],
            "userFeedback": {"thumbs": "up", "comment": "API_KEY=sk-example-secret-12345"},
        }
        duplicate = json.loads(json.dumps(base, ensure_ascii=False))
        duplicate["messageId"] = "m2"
        processed = prepare([base, duplicate])
        self.assertTrue(processed[0]["prep"]["passed"])
        self.assertGreaterEqual(processed[0]["prep"]["redactedCount"], 4)
        self.assertNotIn("13800138000", json.dumps(processed[0]["conversations"], ensure_ascii=False))
        self.assertNotIn("qa@example.com", json.dumps(processed[0]["conversations"], ensure_ascii=False))
        self.assertNotIn("sk-example-secret-12345", processed[0]["userFeedback"]["comment"])
        self.assertEqual(processed[1]["prep"]["duplicateOf"], "m1")

    def test_failed_or_empty_answers_do_not_enter_content_pool(self):
        rows = [
            {"sessionId": "s1", "messageId": "failed", "success": False, "inputTokens": 10, "outputTokens": 0,
             "conversations": [{"from": "user", "value": "问题"}]},
            {"sessionId": "s2", "messageId": "empty", "success": True, "inputTokens": 10, "outputTokens": 0,
             "conversations": [{"from": "user", "value": "另一个问题"}, {"from": "assistant", "value": ""}]},
        ]
        with patch.dict(os.environ, {"JUDGE_API_URL": "", "JUDGE_MODEL": ""}):
            results = run(rows)
        self.assertEqual([r["candidate"]["status"] for r in results], ["不合格", "不合格"])
        self.assertTrue(all(r["hardCase"]["isHard"] for r in results))

    def test_existing_content_is_excluded_from_increment(self):
        rows = [json.loads(line) for line in (ROOT / "sample_10.jsonl").read_text(encoding="utf-8").splitlines()]
        with patch.dict(os.environ, {"JUDGE_API_KEY": ""}):
            results = run(rows, {content_fingerprint(rows[0])})
        self.assertFalse(results[0]["increment"]["selected"])
        self.assertIn("already_in_existing_dataset", results[0]["increment"]["reasons"])

    def test_qwen_request_configuration(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"verdict":"good","reason":"回应问题","taskComplexity":"high","complexityReason":"多项约束","sceneRarity":"rare","rarityReason":"较少见的场景"}'}}]}
        response.raise_for_status.return_value = None
        with patch.dict(os.environ, {"JUDGE_API_KEY": "dummy"}, clear=True), patch("flywheel.requests.post", return_value=response) as post:
            review = judge_with_api("问题", "回答")
        self.assertEqual(review["verdict"], "good")
        self.assertEqual(review["taskComplexity"], "high")
        self.assertEqual(review["sceneRarity"], "rare")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "qwen3.5-plus")
        self.assertIs(kwargs["json"]["enable_thinking"], False)
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["timeout"], (10, 120))

    def test_increment_selects_negative_complex_and_rare(self):
        def item(message_id, question, role, thumbs, complexity, rarity):
            return {
                "sessionId": message_id, "messageId": message_id,
                "conversations": [{"from": "user", "value": question}, {"from": "assistant", "value": "回答" + message_id}],
                "candidate": {"status": "合格", "useAs": role},
                "userFeedback": {"thumbs": thumbs},
                "effect": {"contentReview": {"taskComplexity": complexity, "sceneRarity": rarity}},
            }
        rows = [
            item("negative", "问题一", "hard_case", "down", "normal", "common"),
            item("complex", "问题二", "positive_answer", "unknown", "high", "common"),
            item("rare", "问题三", "positive_answer", "unknown", "normal", "rare"),
        ]
        select_increment(rows)
        self.assertTrue(all(row["increment"]["selected"] for row in rows))
        self.assertIn("explicit_feedback_down", rows[0]["increment"]["reasons"])
        self.assertIn("complex_task", rows[1]["increment"]["reasons"])
        self.assertIn("rare_scene_estimate", rows[2]["increment"]["reasons"])


if __name__ == "__main__":
    unittest.main()
