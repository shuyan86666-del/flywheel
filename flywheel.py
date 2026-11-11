"""Five small operators for one-request-per-row interaction logs.

Run: python flywheel.py sample_10.jsonl --output output/processed_10.jsonl
For the default Qwen review API, set JUDGE_API_KEY in the process environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import requests

PLACEHOLDER = "[REDACTED]"
REQUIRED = ("sessionId", "messageId", "success", "inputTokens", "outputTokens", "conversations")
SENSITIVE = (
    re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{12,}|(?:api[_-]?key|token|password)\s*[:=]\s*\S+)"),
    re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)"),
    re.compile(r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"(?:姓名|地址|病历)\s*[:：]\s*[^，。；;\n]+"),
)


def redact(value: str) -> tuple[str, int]:
    count = 0
    for pattern in SENSITIVE:
        value, n = pattern.subn(PLACEHOLDER, value)
        count += n
    return value, count


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).strip())


def content_fingerprint(row: dict) -> str:
    question, answer = qa(row)
    content = json.dumps([canonical_text(question), canonical_text(answer)], ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def qa(row: dict) -> tuple[str, str]:
    messages = row.get("conversations") or []
    if not isinstance(messages, list):
        return "", ""
    user = next((m.get("value", "") for m in messages if isinstance(m, dict) and m.get("from") == "user" and isinstance(m.get("value"), str)), "")
    answer = next((m.get("value", "") for m in messages if isinstance(m, dict) and m.get("from") == "assistant" and isinstance(m.get("value"), str)), "")
    return user, answer


def validate(row: dict) -> list[str]:
    problems = []
    if any(name not in row for name in REQUIRED):
        return ["missing_required_field"]
    if not isinstance(row["sessionId"], str) or not isinstance(row["messageId"], str):
        problems.append("invalid_id")
    if not isinstance(row["success"], bool):
        problems.append("invalid_success")
    if any(not isinstance(row[key], int) or row[key] < 0 for key in ("inputTokens", "outputTokens")):
        problems.append("invalid_token_count")
    messages = row["conversations"]
    if not isinstance(messages, list) or not messages or len(messages) > 2:
        problems.append("invalid_conversation")
    elif any(not isinstance(m, dict) or m.get("from") not in ("user", "assistant") or not isinstance(m.get("value"), str) for m in messages):
        problems.append("invalid_message")
    elif messages[0]["from"] != "user" or (len(messages) == 2 and messages[1]["from"] != "assistant"):
        problems.append("invalid_role_order")
    elif not messages[0]["value"].strip():
        problems.append("empty_user_input")
    return problems


def ngrams(text: str, n: int = 3) -> set[str]:
    text = canonical_text(text)
    return {text[i : i + n] for i in range(max(1, len(text) - n + 1))} if text else set()


def jaccard(a: str, b: str) -> float:
    x, y = ngrams(a), ngrams(b)
    return len(x & y) / len(x | y) if x or y else 1.0


def prepare(rows: list[dict]) -> list[dict]:
    """Operator 1: normalize, redact, filter, deduplicate."""
    seen_ids: set[str] = set()
    seen_fingerprints: dict[str, str] = {}
    prior: list[dict] = []
    output = []
    for original in rows:
        row = copy.deepcopy(original)
        problems = validate(row)
        redacted = 0
        messages = row.get("conversations")
        for message in messages if isinstance(messages, list) else []:
            if isinstance(message, dict) and isinstance(message.get("value"), str):
                message["value"], count = redact(message["value"])
                redacted += count
        feedback = row.get("userFeedback")
        if isinstance(feedback, dict) and isinstance(feedback.get("comment"), str):
            feedback["comment"], count = redact(feedback["comment"])
            redacted += count
        row_id = row.get("messageId")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                problems.append("duplicate_message_id")
            seen_ids.add(row_id)
        duplicate_of = None
        near_duplicate_of = None
        if not problems:
            question, answer = qa(row)
            digest = content_fingerprint(row)
            duplicate_of = seen_fingerprints.get(digest)
            if duplicate_of:
                problems.append("duplicate_content")
            else:
                seen_fingerprints[digest] = row_id
                # Check answer as well as the prompt, so distinct retries survive.
                for other in prior:
                    q2, a2 = qa(other)
                    if jaccard(question, q2) >= 0.9 and jaccard(answer, a2) >= 0.9:
                        near_duplicate_of = other["messageId"]
                        problems.append("near_duplicate_content")
                        break
                if not near_duplicate_of:
                    prior.append(row)
        row["prep"] = {"passed": not problems, "reasons": problems,
                       "redactedCount": redacted, "duplicateOf": duplicate_of or near_duplicate_of}
        output.append(row)
    return output


def judge_with_api(question: str, answer: str, scene: str = "") -> dict:
    """Qwen OpenAI-compatible review; input is already redacted."""
    url = os.getenv("JUDGE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    model = os.getenv("JUDGE_MODEL", "qwen3.5-plus")
    key = os.getenv("JUDGE_API_KEY")
    if not url or not model or not key:
        return {"verdict": "unknown", "reason": "review_not_configured"}
    body = {
        "model": model,
        "temperature": 0,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "仅根据给出的信息评审，必须输出 JSON 对象。字段：verdict=good/bad/unknown，reason=回答质量的具体理由；taskComplexity=high/normal/unknown，complexityReason=任务复杂度理由；sceneRarity=rare/common/unknown，rarityReason=该场景在生产交互中是否可能稀缺的理由。回答质量检查关键要求、交付形式、矛盾和无依据断言；复杂任务指多约束或多步骤推理。场景稀缺是估计，无法判断用 unknown。"},
            {"role": "user", "content": json.dumps({"question": question, "answer": answer, "scene": scene}, ensure_ascii=False)},
        ],
    }
    try:
        response = requests.post(url, json=body,
                                 headers={"Authorization": f"Bearer {key}"},
                                 timeout=(10, 120))
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        review = json.loads(content)
        if review.get("verdict") not in ("good", "bad", "unknown"):
            raise ValueError("invalid review verdict")
        complexity = review.get("taskComplexity", "unknown")
        rarity = review.get("sceneRarity", "unknown")
        return {"verdict": review["verdict"], "reason": str(review.get("reason", "")),
                "taskComplexity": complexity if complexity in ("high", "normal", "unknown") else "unknown",
                "complexityReason": str(review.get("complexityReason", "")),
                "sceneRarity": rarity if rarity in ("rare", "common", "unknown") else "unknown",
                "rarityReason": str(review.get("rarityReason", ""))}
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        return {"verdict": "unknown", "reason": f"review_error:{type(error).__name__}"}


def analyze_effect(row: dict) -> dict:
    """Operator 3: service, performance, usage and content feedback."""
    question, answer = qa(row)
    duration = row.get("durationMs")
    ttft = row.get("ttftMs")
    input_tokens = row.get("inputTokens")
    output_tokens = row.get("outputTokens")
    service = "failed" if row.get("success") is False else ("empty" if not answer.strip() else "ok")
    performance = []
    if isinstance(ttft, (int, float)) and ttft >= 3000:
        performance.append("slow_first_token")
    if isinstance(duration, (int, float)) and duration >= 60000:
        performance.append("slow_total")
    if isinstance(input_tokens, int) and input_tokens >= 2000:
        performance.append("long_input")
    if isinstance(output_tokens, int) and output_tokens >= 2000:
        performance.append("long_output")
    rule_findings = []
    if (re.search(r"(?:扫描|检索|查询).{0,18}(?:命中|查到|发现)\s*\d+", answer)
            and not re.search(r"(?:扫描|检索|查询).{0,18}(?:命中|查到|发现)\s*\d+", question)
            and not any(row.get(name) for name in ("toolResults", "retrievalResults", "observations"))):
        rule_findings.append("unsupported_result_claim")
    scene = str(row.get("X-Ai-Scene-Name", row.get("category", "")))
    review = judge_with_api(question, answer, scene) if service == "ok" and row["prep"]["passed"] else {"verdict": "unknown", "reason": "no_reviewable_answer"}
    return {"service": service, "performanceSignals": performance,
            "contentRuleFindings": rule_findings, "contentReview": review}


def mine_hard_case(row: dict) -> dict:
    """Operator 2: mark difficult requests using effect evidence."""
    effect = row["effect"]
    feedback = row.get("userFeedback") or {}
    thumbs = feedback.get("thumbs") if isinstance(feedback, dict) else None
    events = row.get("userInteractionEvents") or []
    reasons = []
    if effect["service"] == "failed":
        reasons.append("request_failed")
    elif effect["service"] == "empty":
        reasons.append("empty_answer")
    if thumbs == "down":
        reasons.append("negative_feedback")
    if "regenerate" in events:
        reasons.append("regenerated")
    if "stop_generation" in events:
        reasons.append("generation_stopped")
    if effect["contentReview"]["verdict"] == "bad":
        reasons.append("content_problem")
    reasons.extend(effect["contentRuleFindings"])
    reasons.extend(effect["performanceSignals"])
    return {"isHard": bool(reasons), "reasons": reasons}


def screen_candidate(row: dict) -> dict:
    """Operator 4: admit usable good answers and evidence-backed bad answers."""
    prep, effect = row["prep"], row["effect"]
    feedback = row.get("userFeedback") or {}
    thumbs = feedback.get("thumbs") if isinstance(feedback, dict) else None
    comment = feedback.get("comment") if isinstance(feedback, dict) else None
    events = row.get("userInteractionEvents") or []
    if not prep["passed"]:
        return {"status": "不合格", "useAs": None, "reason": ",".join(prep["reasons"])}
    if effect["service"] in ("failed", "empty"):
        return {"status": "不合格", "useAs": None, "reason": effect["service"]}
    review = effect["contentReview"]["verdict"]
    if thumbs == "down":
        evidence = bool(isinstance(comment, str) and comment.strip()) and "regenerate" in events
        if evidence or review == "bad":
            return {"status": "合格", "useAs": "hard_case", "reason": "negative_feedback_with_evidence"}
        return {"status": "待核验", "useAs": "hard_case", "reason": "negative_feedback_needs_evidence"}
    if effect["contentRuleFindings"]:
        return {"status": "待核验", "useAs": "hard_case", "reason": ",".join(effect["contentRuleFindings"])}
    if review == "bad":
        return {"status": "待核验", "useAs": "hard_case", "reason": "content_review_bad"}
    if thumbs == "up" or review == "good":
        return {"status": "合格", "useAs": "positive_answer", "reason": "positive_feedback_or_review"}
    return {"status": "待核验", "useAs": None, "reason": "insufficient_quality_evidence"}


def select_increment(rows: list[dict], existing_fingerprints: set[str] | None = None) -> None:
    """Operator 5: select feedback, corrections, complex and rare scenarios."""
    positive_by_prompt: dict[tuple[str, str], dict] = {}
    hard_by_prompt: dict[tuple[str, str], dict] = {}
    existing_fingerprints = existing_fingerprints or set()
    for row in rows:
        if row["candidate"]["status"] == "合格" and row["candidate"]["useAs"] == "positive_answer":
            question, _ = qa(row)
            positive_by_prompt[(row["sessionId"], canonical_text(question))] = row
        elif row["candidate"]["status"] == "合格" and row["candidate"]["useAs"] == "hard_case":
            question, _ = qa(row)
            hard_by_prompt[(row["sessionId"], canonical_text(question))] = row
    for row in rows:
        role = row["candidate"]["useAs"]
        selected = False
        pair = None
        reasons = []
        score = 0
        if row["candidate"]["status"] == "合格":
            effect = row["effect"]
            feedback = row.get("userFeedback") or {}
            thumbs = feedback.get("thumbs") if isinstance(feedback, dict) else None
            if thumbs in ("up", "down"):
                score += 2
                reasons.append("explicit_feedback_" + thumbs)
            if effect["contentReview"].get("taskComplexity") == "high":
                score += 1
                reasons.append("complex_task")
            if effect["contentReview"].get("sceneRarity") == "rare":
                score += 1
                reasons.append("rare_scene_estimate")
            question, answer = qa(row)
            prompt_key = (row["sessionId"], canonical_text(question))
            counterpart = hard_by_prompt.get(prompt_key) if role == "positive_answer" else positive_by_prompt.get(prompt_key)
            if counterpart and qa(counterpart)[1] != answer:
                score += 2
                pair = counterpart["messageId"]
                reasons.append("corrected_success" if role == "positive_answer" else "corrected_pair")
            selected = score > 0
        if selected and content_fingerprint(row) in existing_fingerprints:
            selected = False
            reasons.append("already_in_existing_dataset")
        row["increment"] = {"selected": selected, "useAs": role if selected else None,
                            "pairedWith": pair, "priorityScore": score, "reasons": reasons}


def run(rows: list[dict], existing_fingerprints: set[str] | None = None) -> list[dict]:
    rows = prepare(rows)
    for row in rows:
        row["effect"] = analyze_effect(row)
        row["hardCase"] = mine_hard_case(row)
        row["candidate"] = screen_candidate(row)
    select_increment(rows, existing_fingerprints)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five operators on single-turn API logs.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/processed_10.jsonl"))
    parser.add_argument("--existing", type=Path, help="Existing JSONL data to exclude from the increment")
    args = parser.parse_args()
    rows = []
    invalid_json_lines = []
    for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected JSON object")
            rows.append(row)
        except (json.JSONDecodeError, ValueError):
            invalid_json_lines.append(number)
    existing_fingerprints = set()
    if args.existing:
        for line in args.existing.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_fingerprints.add(content_fingerprint(json.loads(line)))
    results = run(rows, existing_fingerprints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    summary = {
        "total": len(results),
        "invalidJsonLines": invalid_json_lines,
        "prepared": sum(row["prep"]["passed"] for row in results),
        "hardCases": sum(row["hardCase"]["isHard"] for row in results),
        "candidate": dict(Counter(row["candidate"]["status"] for row in results)),
        "incrementSelected": sum(row["increment"]["selected"] for row in results),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
