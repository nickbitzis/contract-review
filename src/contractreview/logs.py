"""Structured JSON logs: one JSON object per line, ready for Loki, CloudWatch, or Datadog."""

from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": round(record.created, 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def get_logger(name: str = "contractreview") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def event(log: logging.Logger, msg: str, **fields) -> None:
    log.info(msg, extra={"fields": fields})


def now_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
