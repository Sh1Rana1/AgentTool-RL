"""Deterministic scenario generation for train, validation, and test splits."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .validation import validate_scenario

GENERATOR_VERSION = "0.2.0"
SUPPORTED_SPLITS = ("train", "validation", "test")
ROOT_CAUSES = (
    "deployment_regression",
    "database_connection_pool_exhausted",
    "cache_unavailable",
    "upstream_timeout",
    "certificate_expired",
)
DOMAINS = (
    "payment",
    "order",
    "catalog",
    "checkout",
    "profile",
    "shipment",
    "notification",
    "billing",
)
SPLIT_OFFSETS = {"train": 0, "validation": 120, "test": 240}
SPLIT_SEEDS = {"train": 11, "validation": 23, "test": 37}


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class ScenarioGenerator:
    """Generate balanced, reproducible incidents without sharing task IDs across splits."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate(self, split: str, count: int) -> list[dict[str, Any]]:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
        if count < 1:
            raise ValueError("count must be positive")

        rng = random.Random(self.seed * 1009 + SPLIT_SEEDS[split])
        scenarios = []
        for index in range(count):
            root_cause = ROOT_CAUSES[index % len(ROOT_CAUSES)]
            scenario = self._build(split, index, root_cause, rng)
            validate_scenario(scenario)
            scenarios.append(scenario)
        return scenarios

    def _build(
        self,
        split: str,
        index: int,
        root_cause: str,
        rng: random.Random,
    ) -> dict[str, Any]:
        domain = rng.choice(DOMAINS)
        suffix = f"{split[:3]}-{index:04d}"
        service = f"{domain}-{suffix}-api"
        started_at = datetime(2026, 1, 1, 8, tzinfo=UTC) + timedelta(
            days=SPLIT_OFFSETS[split], minutes=index * 17
        )
        task_id = f"{split}_{root_cause}_{index:05d}"

        builders = {
            "deployment_regression": self._deployment_regression,
            "database_connection_pool_exhausted": self._database_pool,
            "cache_unavailable": self._cache_outage,
            "upstream_timeout": self._upstream_timeout,
            "certificate_expired": self._certificate_expired,
        }
        return builders[root_cause](task_id, service, started_at, rng)

    @staticmethod
    def _base(
        task_id: str,
        service: str,
        started_at: datetime,
        alert: str,
        user_request: str,
        root_cause: str,
        relevant_services: list[str],
        required_evidence: list[str],
        mitigations: list[str],
        tool_data: dict[str, list[dict[str, Any]]],
        expected_path: list[tuple[str, str]],
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "user_request": user_request,
            "incident": {
                "service": service,
                "alert": alert,
                "started_at": _iso(started_at),
            },
            "hidden_state": {
                "root_cause": root_cause,
                "relevant_services": relevant_services,
                "required_evidence": required_evidence,
                "acceptable_mitigations": mitigations,
                "tool_data": tool_data,
            },
            "expected_path": [
                {"tool": tool, "purpose": purpose} for tool, purpose in expected_path
            ],
        }

    def _deployment_regression(
        self, task_id: str, service: str, started_at: datetime, rng: random.Random
    ) -> dict[str, Any]:
        version = f"v2026.{started_at.month:02d}.{started_at.day:02d}.{rng.randint(2, 9)}"
        deploy_id = f"deploy:{service}:{version}"
        log_id = f"log:{service}:null-pointer-{task_id[-5:]}"
        return self._base(
            task_id,
            service,
            started_at,
            "error_rate > 8%",
            f"{service} 错误率突然升高，请定位根因并给出处置建议。",
            "deployment_regression",
            [service],
            [deploy_id, log_id],
            [f"rollback {version}", "disable the newly released feature flag"],
            {
                "logs": [
                    {
                        "service": service,
                        "timestamp": _iso(started_at + timedelta(minutes=3)),
                        "level": "ERROR",
                        "message": "NullPointerException in the newly released request path",
                        "keywords": ["null", "release", "exception"],
                        "evidence_id": log_id,
                    }
                ],
                "metrics": [
                    {
                        "service": service,
                        "metric": "error_rate",
                        "timestamp": _iso(started_at + timedelta(minutes=4)),
                        "value": round(rng.uniform(8.1, 15.0), 2),
                        "unit": "percent",
                        "evidence_id": f"metric:{service}:error-rate:{task_id[-5:]}",
                    }
                ],
                "dependencies": [],
                "deployments": [
                    {
                        "service": service,
                        "version": version,
                        "deployed_at": _iso(started_at - timedelta(minutes=5)),
                        "status": "completed",
                        "evidence_id": deploy_id,
                    }
                ],
            },
            [
                ("get_recent_deployments", "检查告警前的发布变更"),
                ("query_logs", "确认新版本对应的错误签名"),
                ("submit_diagnosis", "提交有证据支持的根因"),
            ],
        )

    def _database_pool(
        self, task_id: str, service: str, started_at: datetime, rng: random.Random
    ) -> dict[str, Any]:
        database = service.removesuffix("-api") + "-db"
        dep_id = f"dep:{service}:{database}"
        metric_id = f"metric:{database}:connection-pool:{task_id[-5:]}"
        return self._base(
            task_id,
            service,
            started_at,
            "latency_p95 > 3s",
            f"{service} 延迟和超时同时升高，请检查依赖并判断故障来源。",
            "database_connection_pool_exhausted",
            [service, database],
            [dep_id, metric_id],
            ["increase connection pool capacity", "terminate leaked database sessions"],
            {
                "logs": [
                    {
                        "service": service,
                        "timestamp": _iso(started_at + timedelta(minutes=2)),
                        "level": "WARN",
                        "message": "Timed out waiting for a database connection",
                        "keywords": ["database", "connection", "timeout"],
                        "evidence_id": f"log:{service}:db-timeout:{task_id[-5:]}",
                    }
                ],
                "metrics": [
                    {
                        "service": database,
                        "metric": "connection_pool_usage",
                        "timestamp": _iso(started_at + timedelta(minutes=3)),
                        "value": rng.randint(96, 100),
                        "unit": "percent",
                        "evidence_id": metric_id,
                    }
                ],
                "dependencies": [
                    {"source": service, "target": database, "evidence_id": dep_id}
                ],
                "deployments": [],
            },
            [
                ("get_dependencies", "定位数据库依赖"),
                ("query_metrics", "检查连接池利用率"),
                ("submit_diagnosis", "提交数据库连接池根因"),
            ],
        )

    def _cache_outage(
        self, task_id: str, service: str, started_at: datetime, rng: random.Random
    ) -> dict[str, Any]:
        cache = service.removesuffix("-api") + "-redis"
        dep_id = f"dep:{service}:{cache}"
        metric_id = f"metric:{service}:cache-hit:{task_id[-5:]}"
        return self._base(
            task_id,
            service,
            started_at,
            "latency_p95 > 1.5s",
            f"{service} 负载正常但延迟升高，请定位相关依赖异常。",
            "cache_unavailable",
            [service, cache],
            [dep_id, metric_id],
            [f"restore {cache}", "temporarily bypass cache with rate limiting"],
            {
                "logs": [
                    {
                        "service": service,
                        "timestamp": _iso(started_at + timedelta(minutes=2)),
                        "level": "ERROR",
                        "message": f"Connection refused by {cache}",
                        "keywords": ["redis", "cache", "connection"],
                        "evidence_id": f"log:{service}:cache-refused:{task_id[-5:]}",
                    }
                ],
                "metrics": [
                    {
                        "service": service,
                        "metric": "cache_hit_rate",
                        "timestamp": _iso(started_at + timedelta(minutes=3)),
                        "value": rng.randint(0, 5),
                        "unit": "percent",
                        "evidence_id": metric_id,
                    }
                ],
                "dependencies": [
                    {"source": service, "target": cache, "evidence_id": dep_id}
                ],
                "deployments": [],
            },
            [
                ("get_dependencies", "识别缓存依赖"),
                ("query_metrics", "验证缓存命中率异常"),
                ("submit_diagnosis", "提交缓存不可用根因"),
            ],
        )

    def _upstream_timeout(
        self, task_id: str, service: str, started_at: datetime, rng: random.Random
    ) -> dict[str, Any]:
        upstream = f"pricing-{task_id.split('_')[0][:3]}-{task_id[-5:]}-api"
        dep_id = f"dep:{service}:{upstream}"
        metric_id = f"metric:{upstream}:latency:{task_id[-5:]}"
        return self._base(
            task_id,
            service,
            started_at,
            "HTTP 504 rate > 12%",
            f"{service} 大量返回 504，请沿调用链查找超时来源。",
            "upstream_timeout",
            [service, upstream],
            [dep_id, metric_id],
            [f"scale {upstream}", "enable the upstream fallback"],
            {
                "logs": [
                    {
                        "service": service,
                        "timestamp": _iso(started_at + timedelta(minutes=2)),
                        "level": "ERROR",
                        "message": f"{upstream} exceeded the upstream deadline",
                        "keywords": ["upstream", "timeout", "deadline"],
                        "evidence_id": f"log:{service}:upstream-timeout:{task_id[-5:]}",
                    }
                ],
                "metrics": [
                    {
                        "service": upstream,
                        "metric": "latency_p95",
                        "timestamp": _iso(started_at + timedelta(minutes=3)),
                        "value": rng.randint(3500, 6000),
                        "unit": "milliseconds",
                        "evidence_id": metric_id,
                    }
                ],
                "dependencies": [
                    {"source": service, "target": upstream, "evidence_id": dep_id}
                ],
                "deployments": [],
            },
            [
                ("get_dependencies", "沿调用链定位上游服务"),
                ("query_metrics", "确认上游延迟异常"),
                ("submit_diagnosis", "提交上游超时根因"),
            ],
        )

    def _certificate_expired(
        self, task_id: str, service: str, started_at: datetime, rng: random.Random
    ) -> dict[str, Any]:
        del rng
        vendor = f"message-vendor-{task_id[-5:]}"
        log_id = f"log:{service}:x509-expired:{task_id[-5:]}"
        dep_id = f"dep:{service}:{vendor}"
        return self._base(
            task_id,
            service,
            started_at,
            "tls_handshake_errors > 100/min",
            f"{service} 调用供应商接口时 TLS 握手持续失败，请确认根因。",
            "certificate_expired",
            [service, vendor],
            [log_id, dep_id],
            ["rotate the expired client certificate", "switch to the valid backup certificate"],
            {
                "logs": [
                    {
                        "service": service,
                        "timestamp": _iso(started_at + timedelta(minutes=2)),
                        "level": "ERROR",
                        "message": "TLS handshake failed: x509 certificate has expired",
                        "keywords": ["tls", "x509", "certificate", "expired"],
                        "evidence_id": log_id,
                    }
                ],
                "metrics": [
                    {
                        "service": service,
                        "metric": "tls_handshake_errors",
                        "timestamp": _iso(started_at + timedelta(minutes=3)),
                        "value": 120,
                        "unit": "per_minute",
                        "evidence_id": f"metric:{service}:tls-errors:{task_id[-5:]}",
                    }
                ],
                "dependencies": [
                    {"source": service, "target": vendor, "evidence_id": dep_id}
                ],
                "deployments": [],
            },
            [
                ("query_logs", "提取证书错误"),
                ("get_dependencies", "确认失败的供应商依赖"),
                ("submit_diagnosis", "提交证书过期根因"),
            ],
        )


def write_dataset(
    output_dir: Path,
    *,
    seed: int = 42,
    split_sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Generate JSONL splits and a checksum manifest."""

    sizes = dict(split_sizes or {"train": 100, "validation": 20, "test": 20})
    if set(sizes) != set(SUPPORTED_SPLITS):
        raise ValueError(f"split_sizes must contain exactly {SUPPORTED_SPLITS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = ScenarioGenerator(seed)
    manifest: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "splits": {},
    }
    all_ids: set[str] = set()
    for split in SUPPORTED_SPLITS:
        scenarios = generator.generate(split, sizes[split])
        ids = {item["task_id"] for item in scenarios}
        if all_ids & ids:
            raise RuntimeError("generated task IDs overlap across splits")
        all_ids.update(ids)
        payload = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in scenarios
        )
        path = output_dir / f"{split}.jsonl"
        path.write_text(payload, encoding="utf-8", newline="\n")
        manifest["splits"][split] = {
            "count": len(scenarios),
            "file": path.name,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
