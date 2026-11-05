import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

from vendorops.tools.registry import ToolDefinition, ToolRegistry

Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[float, float], float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetVendorInput(StrictModel):
    vendor_id: str = Field(min_length=1)


class GetVendorOutput(StrictModel):
    id: str
    name: str
    status: str
    note: str


class ListInvoicesInput(StrictModel):
    vendor_id: str = Field(min_length=1)


class Invoice(StrictModel):
    id: str
    amount: float
    expected_amount: float | None = None
    currency: str
    status: str


class ListInvoicesOutput(StrictModel):
    vendor_id: str
    invoices: list[Invoice]


class VendorAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 2.0,
        max_attempts: int = 3,
        retry_base_delay: float = 0.1,
        retry_max_delay: float = 1.0,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter = random.uniform,
    ) -> None:
        if timeout <= 0 or max_attempts < 1:
            raise ValueError("timeout must be positive and max_attempts must be at least 1")
        if retry_base_delay < 0 or retry_max_delay < 0:
            raise ValueError("retry delays must be non-negative")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep
        self._jitter = jitter
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_json(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.get(url, timeout=self._timeout)
                response.raise_for_status()
            except httpx.TransportError:
                if attempt == self._max_attempts:
                    raise
                await self._sleep(self._retry_delay(attempt))
                continue
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 429 and error.response.status_code < 500:
                    raise
                if attempt == self._max_attempts:
                    raise
                await self._sleep(self._retry_delay(attempt))
                continue

            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Vendor API response must be a JSON object")
            return payload
        raise RuntimeError("Vendor API retry loop exited without a response")

    def _retry_delay(self, attempt: int) -> float:
        ceiling = min(self._retry_max_delay, self._retry_base_delay * (2 ** (attempt - 1)))
        return self._jitter(0.0, ceiling)


def build_vendor_registry(api: VendorAPIClient) -> ToolRegistry:
    async def get_vendor(args: dict[str, Any]) -> dict[str, Any]:
        vendor_id = str(args["vendor_id"])
        return await api.get_json(f"vendors/{quote(vendor_id, safe='')}")

    async def list_invoices(args: dict[str, Any]) -> dict[str, Any]:
        vendor_id = str(args["vendor_id"])
        return await api.get_json(f"vendors/{quote(vendor_id, safe='')}/invoices")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_vendor",
            description="Get vendor details by vendor ID.",
            input_model=GetVendorInput,
            output_model=GetVendorOutput,
            handler=get_vendor,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_invoices",
            description="List invoices for a vendor by vendor ID.",
            input_model=ListInvoicesInput,
            output_model=ListInvoicesOutput,
            handler=list_invoices,
        )
    )
    return registry
