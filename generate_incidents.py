"""
Synthetic incident generator for the agentic RCA eval harness.

Generates fake but realistic incidents, each consisting of:
  - A set of log lines
  - A metrics snapshot
  - A distributed trace
  - A labeled ground-truth root cause (used by the eval layer)

Run:
    python generate_incidents.py --count 20 --output data/incidents.json
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

SERVICES = ["api-gateway", "auth-service", "payment-service", "db-proxy", "cache-service"]

ROOT_CAUSES = [
    {
        "cause": "Database connection pool exhausted",
        "service": "db-proxy",
        "log_signal": "ERROR db-proxy - connection pool exhausted: max_connections=100 active=100",
        "metric_signal": {"metric": "db.connection_pool.active", "value": 100, "threshold": 100},
        "trace_signal": "db-proxy span timed out after 30000ms",
    },
    {
        "cause": "Memory leak in payment-service causing OOM restarts",
        "service": "payment-service",
        "log_signal": "FATAL payment-service - OutOfMemoryError: Java heap space",
        "metric_signal": {"metric": "payment_service.memory.used_mb", "value": 3900, "threshold": 4096},
        "trace_signal": "payment-service span failed: connection reset by peer",
    },
    {
        "cause": "Auth service latency spike due to expired cache",
        "service": "auth-service",
        "log_signal": "WARN auth-service - cache miss rate=0.98 falling back to DB",
        "metric_signal": {"metric": "auth_service.cache.hit_rate", "value": 0.02, "threshold": 0.80},
        "trace_signal": "auth-service span latency=4200ms (p99 baseline=120ms)",
    },
    {
        "cause": "API gateway rate limiter misconfiguration dropping requests",
        "service": "api-gateway",
        "log_signal": "ERROR api-gateway - rate limit applied to internal service traffic: rule=global_limit",
        "metric_signal": {"metric": "api_gateway.requests.dropped_rate", "value": 0.34, "threshold": 0.05},
        "trace_signal": "api-gateway span returned 429 Too Many Requests to payment-service",
    },
    {
        "cause": "Cache service eviction storm under high load",
        "service": "cache-service",
        "log_signal": "WARN cache-service - eviction rate critical: evicted=45000 keys in last 60s",
        "metric_signal": {"metric": "cache_service.eviction_rate", "value": 45000, "threshold": 1000},
        "trace_signal": "cache-service span returned MISS for 94% of keys",
    },
        {
        "cause": "Network packet loss causing service timeouts",
        "service": "api-gateway",
        "log_signal": "ERROR api-gateway - upstream timeout: packet loss detected on network interface eth0",
        "metric_signal": {"metric": "api_gateway.network.packet_loss_rate", "value": 0.18, "threshold": 0.01},
        "trace_signal": "api-gateway span retried 3 times before failing: network unreachable",
    },
    {
        "cause": "Disk I/O saturation on database host",
        "service": "db-proxy",
        "log_signal": "ERROR db-proxy - disk I/O wait critical: iowait=94% read_latency=4200ms",
        "metric_signal": {"metric": "db_proxy.disk.iowait_percent", "value": 94, "threshold": 20},
        "trace_signal": "db-proxy span blocked on disk read for 8200ms before timeout",
    },
    {
        "cause": "TLS certificate expiry causing auth failures",
        "service": "auth-service",
        "log_signal": "ERROR auth-service - TLS handshake failed: certificate expired 2 days ago",
        "metric_signal": {"metric": "auth_service.tls.handshake_failure_rate", "value": 0.91, "threshold": 0.01},
        "trace_signal": "auth-service span failed: SSL: CERTIFICATE_VERIFY_FAILED",
    },
    {
        "cause": "Thread pool exhaustion in payment-service under load",
        "service": "payment-service",
        "log_signal": "ERROR payment-service - thread pool exhausted: active_threads=200 max_threads=200 queue_depth=4500",
        "metric_signal": {"metric": "payment_service.thread_pool.queue_depth", "value": 4500, "threshold": 100},
        "trace_signal": "payment-service span queued for 12000ms waiting for available thread",
    },
    {
        "cause": "DNS resolution failures causing intermittent connectivity",
        "service": "cache-service",
        "log_signal": "ERROR cache-service - DNS resolution failed: Temporary failure in name resolution for upstream-db.internal",
        "metric_signal": {"metric": "cache_service.dns.resolution_failure_rate", "value": 0.43, "threshold": 0.01},
        "trace_signal": "cache-service span failed: getaddrinfo ENOTFOUND upstream-db.internal",
    },
]

LOG_NOISE_TEMPLATES = [
    "INFO {service} - health check ok",
    "DEBUG {service} - processed request in {ms}ms",
    "INFO {service} - connected to upstream",
    "DEBUG {service} - cache hit for key={key}",
    "INFO {service} - request_id={req_id} status=200",
]


def _random_timestamp(base: datetime, jitter_seconds: int = 60) -> str:
    offset = random.randint(-jitter_seconds, jitter_seconds)
    return (base + timedelta(seconds=offset)).isoformat()


def _generate_logs(root_cause: dict, base_time: datetime, noise_lines: int = 8) -> list[dict]:
    logs = []

    # Inject noise lines from various services
    for _ in range(noise_lines):
        template = random.choice(LOG_NOISE_TEMPLATES)
        service = random.choice(SERVICES)
        logs.append({
            "timestamp": _random_timestamp(base_time),
            "level": "INFO" if "INFO" in template or "DEBUG" in template else "WARN",
            "service": service,
            "message": template.format(
                service=service,
                ms=random.randint(5, 300),
                key=str(uuid.uuid4())[:8],
                req_id=str(uuid.uuid4())[:12],
            ),
        })

    # Inject the signal line
    logs.append({
        "timestamp": _random_timestamp(base_time, jitter_seconds=5),
        "level": "ERROR" if "ERROR" in root_cause["log_signal"] else "WARN" if "WARN" in root_cause["log_signal"] else "FATAL",
        "service": root_cause["service"],
        "message": root_cause["log_signal"],
    })

    # Sort by timestamp
    logs.sort(key=lambda x: x["timestamp"])
    return logs


def _generate_metrics(root_cause: dict, base_time: datetime) -> list[dict]:
    signal = root_cause["metric_signal"]
    metrics = []

    # Baseline healthy readings from other services
    for service in SERVICES:
        if service != root_cause["service"]:
            metrics.append({
                "timestamp": _random_timestamp(base_time),
                "service": service,
                "metric": f"{service.replace('-', '_')}.latency_ms",
                "value": random.randint(20, 150),
                "status": "normal",
            })

    # The signal metric — at or above threshold
    metrics.append({
        "timestamp": _random_timestamp(base_time, jitter_seconds=10),
        "service": root_cause["service"],
        "metric": signal["metric"],
        "value": signal["value"],
        "threshold": signal["threshold"],
        "status": "critical",
    })

    return metrics


def _generate_trace(root_cause: dict, base_time: datetime) -> dict:
    trace_id = str(uuid.uuid4())
    spans = []

    # Root span from api-gateway
    root_span_id = str(uuid.uuid4())[:8]
    spans.append({
        "span_id": root_span_id,
        "parent_span_id": None,
        "service": "api-gateway",
        "operation": "handle_request",
        "start_time": base_time.isoformat(),
        "duration_ms": random.randint(4000, 8000),
        "status": "error",
    })

    # Child spans from intermediate services
    for service in SERVICES:
        if service in ["api-gateway", root_cause["service"]]:
            continue
        spans.append({
            "span_id": str(uuid.uuid4())[:8],
            "parent_span_id": root_span_id,
            "service": service,
            "operation": "process",
            "start_time": _random_timestamp(base_time, jitter_seconds=2),
            "duration_ms": random.randint(10, 200),
            "status": "ok",
        })

    # Failing span from the root-cause service
    spans.append({
        "span_id": str(uuid.uuid4())[:8],
        "parent_span_id": root_span_id,
        "service": root_cause["service"],
        "operation": "process",
        "start_time": _random_timestamp(base_time, jitter_seconds=2),
        "duration_ms": random.randint(3000, 6000),
        "status": "error",
        "error_message": root_cause["trace_signal"],
    })

    return {"trace_id": trace_id, "spans": spans}


def generate_incident(index: int) -> dict:
    root_cause = random.choice(ROOT_CAUSES)
    base_time = datetime.now(timezone.utc) - timedelta(minutes=random.randint(5, 120))

    return {
        "incident_id": f"incident-{str(uuid.uuid4())[:8]}",
        "index": index,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "logs": _generate_logs(root_cause, base_time),
        "metrics": _generate_metrics(root_cause, base_time),
        "trace": _generate_trace(root_cause, base_time),
        "ground_truth": {
            "root_cause": root_cause["cause"],
            "affected_service": root_cause["service"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RCA incidents")
    parser.add_argument("--count", type=int, default=20, help="Number of incidents to generate")
    parser.add_argument("--output", type=str, default="data/incidents.json", help="Output file path")
    args = parser.parse_args()

    incidents = [generate_incident(i + 1) for i in range(args.count)]

    with open(args.output, "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"Generated {args.count} incidents -> {args.output}")
    print(f"Root causes represented: {len(set(i['ground_truth']['root_cause'] for i in incidents))}/{len(ROOT_CAUSES)}")


if __name__ == "__main__":
    main()
