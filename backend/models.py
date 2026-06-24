from pydantic import BaseModel
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    ticker: str
    start_date: str   # YYYY-MM-DD
    end_date: str
    test_size: float = 0.05
    lag_begin: int = 1
    lag_end: int = 7
    model_type: str = "multiplicative"
    period_value: int = 30
    smoothings: List[int] = [7, 30, 90]