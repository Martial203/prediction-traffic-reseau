export type ModelId = 'random_forest' | 'xgboost' | 'lstm' | 'seasonal_naive' | 'persistence';

export interface Metrics {
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
  n: number;
}

export interface ModelSummary {
  id: ModelId;
  label: string;
  kind: 'ml' | 'baseline';
  fit_seconds: number;
  metrics: Metrics;
}

export interface Overview {
  dataset: {
    rows: number;
    cells: string[];
    start: string;
    end: string;
    valid_start: string;
    test_start: string;
    test_ratio: number;
    lookback_hours: number;
    train_rows: number;
    valid_rows: number;
    test_rows: number;
  };
  trained_at: string;
  best_model: ModelId;
  models: ModelSummary[];
  kpis: {
    mean_traffic: number;
    peak_traffic: number;
    mean_users: number;
    events: Record<string, number>;
  };
}

export type MetricsBy = Record<ModelId, Record<string, Metrics>>;

export interface CellInfo {
  id: string;
  mean_traffic: number;
  peak_traffic: number;
  mean_users: number;
  mean_rsrp: number;
  mean_sinr: number;
  congestion_hours: number;
  handover_hours: number;
}

export interface CellSeries {
  cell: string;
  test_start: string;
  timestamps: string[];
  actual: (number | null)[];
  active_users: number[];
  rsrp: (number | null)[];
  sinr: (number | null)[];
  event: string[];
  predictions: Record<ModelId, (number | null)[]>;
}

export interface Forecast {
  cell: string;
  model: ModelId;
  timestamps: string[];
  values: number[];
  lower: number[];
  upper: number[];
}

export interface LstmHistory {
  train_loss: number[];
  valid_loss: number[];
}
