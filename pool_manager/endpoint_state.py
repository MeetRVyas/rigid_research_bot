from dataclasses import dataclass, field
from langchain_core.language_models.chat_models import BaseChatModel
from datetime import datetime, timedelta


@dataclass
class EndpointState:
    id: str
    provider: str
    model:    str
    rpm_limit: int
    rpd_limit: int
    client: BaseChatModel

    daily_total:  int = 0
    minute_total: int = 0
    inflight: int = 0

    minute_window:  datetime = field(default_factory=datetime.now)
    daily_window:   datetime = field(default_factory=datetime.now)
    cooldown_until: datetime = field(default_factory=datetime.now)

    def _refresh_window(self, now: datetime):
        if now - self.minute_window >= timedelta(minutes=1):
            self.minute_total  = 0
            self.minute_window = now

        if now - self.daily_window >= timedelta(days=1):
            self.daily_total  = 0
            self.daily_window = now

    def is_in_cooldown(self, now: datetime) -> bool:
        if now < self.cooldown_until:
            return True
        return False
    
    def mark_rate_limited(self):
            self.cooldown_until = datetime.now() + timedelta(seconds=60)
    
    def _is_available(self, now: datetime) -> bool:
        self._refresh_window(now)

        return (
            not self.is_in_cooldown(now)
            and self.minute_total < self.rpm_limit
            and self.daily_total < self.rpd_limit
        )
    
    def reserve(self) -> bool:
        now = datetime.now()    

        if not self._is_available(now):
            return False
        
        self.inflight     += 1
        self.minute_total += 1
        self.daily_total  += 1

        return True
        
    def release(self):
        self.inflight = max(0, self.inflight - 1)
