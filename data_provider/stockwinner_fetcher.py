# -*- coding: utf-8 -*-
"""StockWinner local K-line fetcher.

StockWinner (http://127.0.0.1:8080) serves its own local quote/K-line store
(`data/kline.db`, tens of GB) behind a bearer-token API. Its UI endpoints are
protected by session auth, so this fetcher logs in with the configured
account credentials, caches the token, and re-authenticates on 401.

Endpoints used:
    POST /api/auth/login                                -> {"token": "..."}
    GET  /api/v1/ui/{account_id}/stocks/{code}/kline-local?limit=N
    GET  /api/v1/ui/{account_id}/market/quote/{code}    (stock name)

`kline-local` ignores explicit start/end parameters, so the requested window is
estimated into a row count and then sliced client-side.
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, normalize_stock_code

logger = logging.getLogger(__name__)

_MAX_KLINE_BARS = 800
_MIN_KLINE_BARS = 60


def _estimate_bars(start_date: str, end_date: str) -> int:
    """Estimate how many trading bars cover the requested window."""
    try:
        start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date)[:10], "%Y-%m-%d").date()
    except Exception:
        return _MIN_KLINE_BARS
    if end < start:
        return _MIN_KLINE_BARS
    calendar_days = (end - start).days + 1
    # ~5 trading days per 7 calendar days, plus a buffer for holidays.
    bars = int(calendar_days * 5 / 7) + 20
    return max(_MIN_KLINE_BARS, min(_MAX_KLINE_BARS, bars))


def _bare_code(stock_code: str) -> str:
    """Normalize to the bare 6-digit form StockWinner expects."""
    code = normalize_stock_code(str(stock_code or "").strip())
    return code.split(".")[0]


class StockWinnerFetcher(BaseFetcher):
    """Daily K-line fallback backed by a locally running StockWinner service."""

    name = "StockWinnerFetcher"
    priority = 5
    allow_empty_daily_data = True

    _HTTP_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        base_url: Optional[str] = None,
        account_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        priority: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._account_id = account_id or ""
        self._username = username or ""
        self._password = password or ""
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()
        if priority is not None:
            self.priority = priority
        if timeout:
            self._HTTP_TIMEOUT_SECONDS = timeout

    # ------------------------------------------------------------------
    # credentials / availability
    # ------------------------------------------------------------------
    @staticmethod
    def has_configured_credentials(config: Any) -> bool:
        base_url = (getattr(config, "stockwinner_base_url", None) or "").strip()
        username = (getattr(config, "stockwinner_username", None) or "").strip()
        password = (getattr(config, "stockwinner_password", None) or "").strip()
        return bool(base_url and username and password)

    @property
    def is_available(self) -> bool:
        return bool(self._base_url and self._username and self._password)

    # ------------------------------------------------------------------
    # auth + HTTP
    # ------------------------------------------------------------------
    def _login(self) -> Optional[str]:
        try:
            resp = requests.post(
                f"{self._base_url}/api/auth/login",
                json={"name": self._username, "password": self._password},
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("StockWinnerFetcher login failed: %s", exc)
            return None
        token = payload.get("token") or (payload.get("data") or {}).get("token")
        if not token:
            logger.warning("StockWinnerFetcher login returned no token")
            return None
        return str(token)

    def _get(self, path: str) -> Optional[Dict[str, Any]]:
        """GET with bearer auth; re-authenticate once on 401."""
        if not self.is_available:
            return None
        for attempt in (1, 2):
            with self._token_lock:
                token = self._token
            if not token:
                token = self._login()
                with self._token_lock:
                    self._token = token
                if not token:
                    return None
            try:
                resp = requests.get(
                    f"{self._base_url}{path}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self._HTTP_TIMEOUT_SECONDS,
                )
                if resp.status_code == 401 and attempt == 1:
                    with self._token_lock:
                        self._token = None
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                logger.warning("StockWinnerFetcher GET %s failed: %s", path, exc)
                return None
        return None

    # ------------------------------------------------------------------
    # BaseFetcher contract
    # ------------------------------------------------------------------
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = _bare_code(stock_code)
        if not code:
            raise DataFetchError(f"StockWinnerFetcher unsupported stock code: {stock_code}")
        if not self.is_available:
            raise DataFetchError("StockWinnerFetcher not configured")

        limit = _estimate_bars(start_date, end_date)
        payload = self._get(
            f"/api/v1/ui/{self._account_id}/stocks/{code}/kline-local?limit={limit}"
        )
        if not payload or not payload.get("success"):
            message = (payload or {}).get("message", "no response")
            raise DataFetchError(f"StockWinnerFetcher kline-local failed: {message}")

        rows = payload.get("kline") or []
        if not rows:
            logger.info("StockWinnerFetcher empty daily history for %s", stock_code)
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

        df = pd.DataFrame(rows)
        if "trade_date" not in df.columns:
            raise DataFetchError("StockWinnerFetcher response missing trade_date")
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = df["date"].astype(str).str[:10]
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        start = str(start_date)[:10]
        end = str(end_date)[:10]
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            logger.info(
                "StockWinnerFetcher history outside requested range for %s: %s~%s",
                stock_code, start, end,
            )
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "amount" not in normalized.columns:
            normalized["amount"] = 0.0
        if "pct_chg" not in normalized.columns:
            normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100
        return normalized[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        code = _bare_code(stock_code)
        if not code:
            return None
        payload = self._get(f"/api/v1/ui/{self._account_id}/market/quote/{code}")
        if not payload or not payload.get("success"):
            return None
        data = payload.get("data") or {}
        name = (data.get("stock_name") or "").strip()
        return name or None
