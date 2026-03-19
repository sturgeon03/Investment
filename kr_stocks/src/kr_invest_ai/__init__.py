"""Korea-market adapter helpers layered on top of the shared research core."""

from kr_invest_ai.calendar import (
    SessionWindow,
    align_filing_timestamp_to_session_date,
    build_regular_session_window,
    is_regular_trading_day,
)
from kr_invest_ai.dart_adapter import (
    NormalizedDARTFiling,
    classify_dart_filing_category,
    normalize_dart_filing,
    normalize_dart_filings,
)
from kr_invest_ai.dart_client import (
    DARTAPIError,
    DARTListFilingsRequest,
    DARTOpenAPIClient,
    build_dart_filing_url,
    map_dart_list_row_to_payload,
)
from kr_invest_ai.data_bundle import (
    KRResearchDataBundle,
    KRResearchDataRequest,
    build_kr_research_data_bundle,
)
from kr_invest_ai.features import build_kr_feature_frame
from kr_invest_ai.ml_strategy import KRMLModelConfig, generate_ridge_target_weights
from kr_invest_ai.walkforward import KRWalkForwardConfig
from kr_invest_ai.ml_strategy import generate_ridge_walkforward_target_weights
from kr_invest_ai.research import (
    KRResearchRun,
    run_kr_research_backtest,
    run_kr_research_pipeline,
    save_kr_research_outputs,
)
from kr_invest_ai.market_config import (
    FeeTaxConfig,
    KRMarketAdapterConfig,
    MarketAssumptions,
    TickerConventions,
    TradingCalendarConfig,
    load_kr_market_adapter_config,
)
from kr_invest_ai.market_data_client import (
    KRDailyOHLCVRequest,
    KRHistoricalMarketDataClient,
    KRMarketDataError,
    NormalizedDailyOHLCVBar,
    bars_to_frame,
)
from kr_invest_ai.pipeline import (
    build_request_signature,
    load_cached_bundle,
    load_corp_codes_csv,
    run_kr_data_pipeline,
    save_bundle_outputs,
)
from kr_invest_ai.strategy import KRStrategyConfig, generate_target_weights
from kr_invest_ai.tickers import CanonicalKRTicker, canonicalize_kr_ticker, normalize_listing_code

SHARED_CORE_MODULES = (
    "invest_ai_core.config",
    "invest_ai_core.market_data",
)

__all__ = [
    "CanonicalKRTicker",
    "DARTAPIError",
    "DARTListFilingsRequest",
    "DARTOpenAPIClient",
    "FeeTaxConfig",
    "KRDailyOHLCVRequest",
    "KRHistoricalMarketDataClient",
    "KRMarketAdapterConfig",
    "KRMarketDataError",
    "KRMLModelConfig",
    "KRResearchDataBundle",
    "KRResearchDataRequest",
    "KRResearchRun",
    "KRStrategyConfig",
    "KRWalkForwardConfig",
    "MarketAssumptions",
    "NormalizedDailyOHLCVBar",
    "NormalizedDARTFiling",
    "SessionWindow",
    "SHARED_CORE_MODULES",
    "TickerConventions",
    "TradingCalendarConfig",
    "align_filing_timestamp_to_session_date",
    "build_regular_session_window",
    "build_dart_filing_url",
    "build_kr_research_data_bundle",
    "build_kr_feature_frame",
    "build_request_signature",
    "generate_target_weights",
    "generate_ridge_target_weights",
    "generate_ridge_walkforward_target_weights",
    "canonicalize_kr_ticker",
    "classify_dart_filing_category",
    "is_regular_trading_day",
    "bars_to_frame",
    "load_cached_bundle",
    "load_corp_codes_csv",
    "map_dart_list_row_to_payload",
    "normalize_dart_filing",
    "normalize_dart_filings",
    "load_kr_market_adapter_config",
    "normalize_listing_code",
    "run_kr_data_pipeline",
    "run_kr_research_backtest",
    "run_kr_research_pipeline",
    "save_bundle_outputs",
    "save_kr_research_outputs",
]
