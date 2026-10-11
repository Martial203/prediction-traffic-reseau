import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ChartConfiguration } from 'chart.js';

import { ModelId } from '../../core/api.models';
import { ApiService } from '../../core/api.service';
import { ThemeService } from '../../core/theme.service';
import { ChartComponent } from '../../shared/chart.component';
import { FONT, baseOptions } from '../../shared/chart-defaults';

const FEATURE_LABELS: Record<string, string> = {
  hour: 'Heure',
  dayofweek: 'Jour de la semaine',
  is_weekend: 'Week-end',
  hour_sin: 'Heure (sin)',
  hour_cos: 'Heure (cos)',
  dow_sin: 'Jour (sin)',
  dow_cos: 'Jour (cos)',
  cell_code: 'Cellule',
  traffic_lag1: 'Trafic H-1',
  traffic_lag2: 'Trafic H-2',
  traffic_lag3: 'Trafic H-3',
  traffic_lag6: 'Trafic H-6',
  traffic_lag12: 'Trafic H-12',
  traffic_lag24: 'Trafic J-1 (H-24)',
  traffic_roll_mean6: 'Moyenne trafic 6 h',
  traffic_roll_mean24: 'Moyenne trafic 24 h',
  traffic_roll_std24: 'Écart-type trafic 24 h',
  active_users_lag1: 'Utilisateurs H-1',
  rsrp_lag1: 'RSRP H-1',
  sinr_lag1: 'SINR H-1',
  congestion_lag1: 'Congestion H-1',
  handover_lag1: 'Handover H-1',
};
const TOP = 12;

@Component({
  selector: 'app-models-page',
  imports: [ChartComponent],
  templateUrl: './models.page.html',
})
export class ModelsPage {
  private readonly api = inject(ApiService);
  private readonly theme = inject(ThemeService);

  readonly overview = toSignal(this.api.overview$);
  readonly rf = toSignal(this.api.featureImportance('random_forest'));
  readonly xgb = toSignal(this.api.featureImportance('xgboost'));
  readonly history = toSignal(this.api.lstmHistory$);

  readonly rfChart = computed(() => this.importanceChart('random_forest', this.rf()));
  readonly xgbChart = computed(() => this.importanceChart('xgboost', this.xgb()));

  readonly epochs = computed(() => this.history()?.valid_loss?.length ?? 0);
  readonly bestEpoch = computed(() => {
    const v = this.history()?.valid_loss ?? [];
    if (!v.length) return 0;
    return v.indexOf(Math.min(...v)) + 1;
  });

  readonly lossChart = computed<ChartConfiguration<'line'> | null>(() => {
    const h = this.history();
    if (!h?.train_loss?.length) return null;
    const c = this.theme.colors();
    const opts = baseOptions<'line'>(c, 'Erreur quadratique (normalisée)');
    return {
      type: 'line',
      data: {
        labels: h.train_loss.map((_, i) => String(i + 1)),
        datasets: [
          {
            label: 'Entraînement',
            data: h.train_loss,
            borderColor: c.inkSecondary,
            backgroundColor: c.inkSecondary,
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 0,
          },
          {
            label: 'Validation',
            data: h.valid_loss,
            borderColor: c.model.lstm,
            backgroundColor: c.model.lstm,
            borderWidth: 2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        ...opts,
        scales: {
          ...opts.scales,
          x: { ...opts.scales!['x'], title: { display: true, text: 'Époque', color: c.muted } },
        },
      },
    };
  });

  private importanceChart(
    id: ModelId,
    imp?: Record<string, number>,
  ): ChartConfiguration<'bar'> | null {
    if (!imp) return null;
    const c = this.theme.colors();
    const top = Object.entries(imp).slice(0, TOP);
    const opts = baseOptions<'bar'>(c);
    return {
      type: 'bar',
      data: {
        labels: top.map(([k]) => FEATURE_LABELS[k] ?? k),
        datasets: [
          {
            label: 'Importance (%)',
            data: top.map(([, v]) => v * 100),
            backgroundColor: c.model[id],
            borderRadius: 4,
            borderSkipped: 'start',
            barThickness: 14,
          },
        ],
      },
      options: {
        ...opts,
        indexAxis: 'y',
        interaction: { mode: 'nearest', intersect: true, axis: 'y' },
        plugins: { ...opts.plugins, legend: { display: false } },
        scales: {
          x: {
            ...opts.scales!['y'],
            beginAtZero: true,
            title: { display: true, text: 'Importance (%)', color: c.muted },
          },
          y: {
            ...opts.scales!['x'],
            ticks: { color: c.inkSecondary, font: FONT, autoSkip: false },
          },
        },
      },
    };
  }
}
