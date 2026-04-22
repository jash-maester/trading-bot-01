# 06 — Paper Broker and Live Adapter Stub

## Goals

- A realistic simulated broker that executes orders at next-day open
  with slippage and real Indian fees, persisting state to Postgres.
- A clean `Broker` abstract base that the trained policy can use for
  both paper and eventual live trading.
- A live `ZerodhaBroker` that implements the interface but is **not
  wired to any credentials or any real endpoint** in v1.

## `Broker` interface

```python
class Broker(ABC):
    def get_nav(self) -> float: ...
    def get_cash(self) -> float: ...
    def get_positions(self) -> dict[str, Position]: ...
    def place_order(self, order: OrderRequest) -> OrderAck: ...
    def cancel_order(self, client_order_id: str) -> None: ...
    def get_order(self, client_order_id: str) -> Order: ...
    def on_market_close(self, date: date) -> PortfolioSnapshot: ...
    def history(
        self, start: date, end: date
    ) -> pl.DataFrame: ...
```

Data classes (Pydantic):

```python
class OrderRequest(BaseModel):
    client_order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: float | None = None
    strategy_id: str
    tag: str | None = None

class Position(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    last_mark: float

class PortfolioSnapshot(BaseModel):
    ts: datetime
    cash: float
    equity_value: float
    total_value: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    turnover: float
    positions: list[Position]
```

## Postgres schema (`ledger` schema)

```sql
CREATE TABLE ledger.strategy_runs (
  id             BIGSERIAL PRIMARY KEY,
  strategy_id    TEXT NOT NULL,
  started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at       TIMESTAMPTZ,
  config_hash    TEXT NOT NULL,
  mlflow_run_id  TEXT,
  initial_cash   NUMERIC(18,2) NOT NULL,
  mode           TEXT NOT NULL CHECK (mode IN ('paper','live-stub','live')),
  notes          TEXT
);

CREATE TABLE ledger.orders (
  id               BIGSERIAL PRIMARY KEY,
  strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
  client_order_id  TEXT UNIQUE NOT NULL,
  ts_submitted    TIMESTAMPTZ NOT NULL,
  symbol          TEXT NOT NULL,
  side            TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  quantity        INTEGER NOT NULL CHECK (quantity > 0),
  order_type      TEXT NOT NULL,
  limit_price     NUMERIC(18,4),
  status          TEXT NOT NULL CHECK (status IN
                     ('PENDING','FILLED','PARTIAL','CANCELLED','REJECTED')),
  reject_reason   TEXT,
  tag             TEXT
);
CREATE INDEX ON ledger.orders (strategy_run_id, ts_submitted);

CREATE TABLE ledger.fills (
  id          BIGSERIAL PRIMARY KEY,
  order_id    BIGINT REFERENCES ledger.orders(id) NOT NULL,
  ts          TIMESTAMPTZ NOT NULL,
  quantity    INTEGER NOT NULL CHECK (quantity > 0),
  price       NUMERIC(18,4) NOT NULL,
  fees        NUMERIC(18,4) NOT NULL DEFAULT 0,
  slippage    NUMERIC(18,6) NOT NULL DEFAULT 0
);
CREATE INDEX ON ledger.fills (order_id);

CREATE TABLE ledger.positions (
  id               BIGSERIAL PRIMARY KEY,
  strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
  symbol           TEXT NOT NULL,
  quantity         INTEGER NOT NULL,
  avg_price        NUMERIC(18,4) NOT NULL,
  last_mark        NUMERIC(18,4),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(strategy_run_id, symbol)
);

CREATE TABLE ledger.portfolio_snapshots (
  id               BIGSERIAL PRIMARY KEY,
  strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
  ts               TIMESTAMPTZ NOT NULL,
  cash             NUMERIC(18,2) NOT NULL,
  equity_value     NUMERIC(18,2) NOT NULL,
  total_value      NUMERIC(18,2) NOT NULL,
  realized_pnl     NUMERIC(18,2) NOT NULL,
  unrealized_pnl   NUMERIC(18,2) NOT NULL,
  fees_paid        NUMERIC(18,2) NOT NULL,
  turnover         NUMERIC(18,2) NOT NULL,
  metrics_json     JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX ON ledger.portfolio_snapshots (strategy_run_id, ts);

CREATE TABLE ledger.pnl_daily (
  strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
  date             DATE NOT NULL,
  realized         NUMERIC(18,2) NOT NULL,
  unrealized       NUMERIC(18,2) NOT NULL,
  fees             NUMERIC(18,2) NOT NULL,
  total            NUMERIC(18,2) NOT NULL,
  PRIMARY KEY (strategy_run_id, date)
);
```

Migrations managed by Alembic. Initial migration `0001_initial.py`
creates `market.*` and `ledger.*` schemas and the tables above.

## `PaperBroker` behavior

Execution model (daily):

1. `place_order` during bar `t-1`'s close adds a `PENDING` row.
2. At the open of bar `t`, each pending order is filled:
   - `fill_price = open_t * (1 + slippage)`, where
     `slippage = k * (atr_14 / close) * sign(qty) *
                 sqrt(|qty| / adv_20)`. Clamp to ±3%.
   - Fees computed from `trader.env.costs.*CostModel`.
   - Emit a `fills` row and a position update.
3. At the close of bar `t`, `on_market_close` is called:
   - Mark positions to close.
   - Compute and persist a `portfolio_snapshots` row.
   - Upsert `pnl_daily` for the date.
4. Lot sizes and tick sizes from `market.instruments` are respected;
   a target weight is quantized to integer lots (nearest, not
   truncated, with a hard cap of cash available).

T+1 cash settlement: proceeds from a SELL are not available as BUY
power until the next trading day. Implemented with a `pending_settlement`
bucket in the cash account.

Rejection conditions:
- Insufficient cash after modeled costs.
- Negative quantity resulting from rounding.
- Symbol not in today's universe snapshot.

## `ZerodhaBroker` stub

- Implements the same `Broker` ABC.
- Constructor reads `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_ACCESS_TOKEN`
  from env, but methods raise `NotImplementedError("live trading not
  enabled; see docs/going_live.md")`.
- A TODO comment references the Kite Connect Python SDK with links to:
  - `kiteconnect.KiteConnect` for REST.
  - `kiteconnect.KiteTicker` for streaming ticks.
- `configs/broker/zerodha.yaml` exists with placeholder keys and a
  big `enabled: false` field that training/paper-run code checks.

Going live is a separate, explicit step — not a default — with its own
runbook.

## Running a paper session

```bash
python scripts/paper_run.py \
    broker=paper \
    data=universe_v1 \
    model.checkpoint=mlruns/<run_id>/artifacts/model.pt \
    paper.start_date=2024-01-01 \
    paper.end_date=2024-12-31 \
    paper.initial_cash=1_000_000
```

Effects:
- Inserts a `strategy_runs` row.
- Iterates trading days: loads features → policy forward → target
  allocation → broker orders → fills → snapshot.
- All state in Postgres; nothing lives in RAM beyond the process.

## Ledger tooling

```bash
python scripts/ledger.py show --run 42
python scripts/ledger.py report --run 42 --format pdf --out report.pdf
python scripts/ledger.py compare --runs 42,43,44
```

Generates a report with equity curve, drawdown, monthly PnL heatmap,
sector exposure, and the baselines evaluated on the same dates for
comparison.

## Invariants

- `sum(positions.value) + cash ≈ portfolio_snapshots.total_value`
  within 0.01 INR per snapshot.
- No order rows without a matching strategy_run row (FK enforced).
- No fills exceeding their parent order's quantity.
- `pnl_daily.total` equals the delta of `portfolio_snapshots.total_value`
  between the last snapshot of `date-1` and the last snapshot of
  `date`.

## Acceptance criteria for Phase 5

- Alembic `upgrade head` creates all tables.
- `PaperBroker` passes a golden-file integration test: a fixed seed and
  fixed dates produce fixed fills, fixed NAV, fixed snapshot rows.
- `scripts/paper_run.py` runs a 1-year paper session end-to-end in
  under 30 minutes with a trained checkpoint.
- `scripts/ledger.py report` produces a PDF identical across reruns
  from the same DB state.
- `ZerodhaBroker` import does not touch the network and all methods
  raise `NotImplementedError` cleanly.
