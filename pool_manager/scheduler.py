import asyncio
from pool_manager.endpoint_state import EndpointState
from contextlib import asynccontextmanager
from datetime import datetime

class AllEndpointsExhausted(Exception):
    """All Endpoints are Exhausted"""   

class LLMScheduler:
    def __init__(self,
                endpoints: list[EndpointState],
                max_wait_seconds: float = 120,
                poll_interval: float = 1.0
    ):
        self.endpoints = endpoints
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval = poll_interval

    async def _get_best_endpoint(self) -> EndpointState:
        start = datetime.now()
        while True:
            now = datetime.now()
            candidates = [ep for ep in self.endpoints if ep._is_available(now)]

            if candidates:
                for ep in sorted(candidates, key=lambda e: e.inflight):
                    if ep.reserve():
                        return ep
                    
            if (now - start).total_seconds() >= self.max_wait_seconds:
                raise AllEndpointsExhausted(
                    f"No endpoint became available within {self.max_wait_seconds}s "
                    f"({len(self.endpoints)} endpoints configured)."
                )

            await asyncio.sleep(self.poll_interval)
            
    @asynccontextmanager
    async def allocate_endpoint(self):

        endpoint = await self._get_best_endpoint()

        try:
            yield endpoint
        finally:
            endpoint.release()

            
