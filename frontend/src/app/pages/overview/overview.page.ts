import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { ChartConfiguration } from 'chart.js';
import { combineLatest } from 'rxjs';

import { MetricsBy, ModelId, ModelSummary, Overview } from '../../core/api.models';
import { ApiService } from '../../core/api.service';
import { ChartColors, ThemeService } from '../../core/theme.service';
import { ChartComponent } from '../../shared/chart.component';
import { baseOptions } from '../../shared/chart-defaults';
import { asLoad } from '../../shared/load';

const ML: ModelId[] = ['random_forest', 'xgboost', 'lstm'];
const EVENTS = [
  { id: 'normal', label: 'Normal' },
  { id: 'handover', label: 'Handover' },
  { id: 'congestion', label: 'Congestion' },
];

interface Data {
  overview: Overview;
  byCell: MetricsBy;
  byEvent: MetricsBy;
}

@Component({
  selector: 'app-overview-page',
  imports: [ChartComponent, DecimalPipe, DatePipe, RouterLink],
  templateUrl: './overview.page.html',
})
export class OverviewPage {
  private readonly api = inject(ApiService);
  private readonly theme = inject(ThemeService);

  readonly events = EVENTS;
  readonly state = toSignal(
    asLoad(
      combineLatest({
        overview: this.api.overview$,
        byCell: this.api.metricsByCell$,
        byEvent: this.api.metricsByEvent$,
      }),
    ),
    { initialValue: { status: 'loading' } as const },
  );

  readonly data = computed(() => {
    const s = this.state();
    return s.status === 'ok' ? s.data : null;
  });

  readonly ranked = computed(() =>
    [...(this.data()?.overview.models ?? [])].sort((a, b) => a.metrics.rmse - b.metrics.rmse),
  );

  readonly best = computed(() => {
    const d = this.data();
    return d ? d.overview.models.find((m) => m.id === d.overview.best_model)! : null;
  });

  /** Réduction de RMSE du meilleur modèle par rapport au naïf saisonnier. */
  readonly gainVsNaive = computed(() => {
    const d = this.data();
    const best = this.best();
    const naive = d?.overview.models.find((m) => m.id === 'seasonal_naive');
    if (!best || !naive) return null;
    return (1 - best.metrics.rmse / naive.metrics.rmse) * 100;
  });

  readonly cells = computed(() => this.data()?.overview.dataset.cells ?? []);
  readonly mlModels = computed(() =>
    ML.map((id) => this.data()?.overview.models.find((m) => m.id === id)).filter(
      (m): m is ModelSummary => !!m,
    ),
  );

  readonly rmseChart = computed<ChartConfiguration<'bar'> | null>(() => {
    const models = this.ranked();
    if (!models.length) return null;
    const c = this.theme.colors();
    const opts = baseOptions<'bar'>(c, 'RMSE (Mbps)');
    return {
      type: 'bar',
      data: {
        labels: models.map((m) => m.label),
        datasets: [
          {
            label: 'RMSE sur la période de test',
            data: models.map((m) => m.metrics.rmse),
            backgroundColor: models.map((m) => c.model[m.id]),
            borderRadius: 4,
            borderSkipped: 'start',
            barThickness: 18,
          },
        ],
      },
      options: {
        ...opts,
        indexAxis: 'y',
        interaction: { mode: 'nearest', intersect: true },
        plugins: { ...opts.plugins, legend: { display: false } },
        scales: {
          x: { ...opts.scales!['y'], beginAtZero: true },
          y: {
            ...opts.scales!['x'],
            ticks: { ...opts.scales!['x']!.ticks, color: c.inkSecondary },
          },
        },
      },
    };
  });

  readonly cellChart = computed<ChartConfiguration<'bar'> | null>(() => {
    const d = this.data();
    if (!d) return null;
    const c = this.theme.colors();
    return {
      type: 'bar',
      data: {
        labels: d.overview.dataset.cells,
        datasets: this.mlModels().map((m) => ({
          label: m.label,
          data: d.overview.dataset.cells.map((cell) => d.byCell[m.id][cell].rmse),
          backgroundColor: c.model[m.id],
          borderColor: c.surface,
          borderWidth: { left: 1, right: 1 },
          borderRadius: 4,
          borderSkipped: 'start',
          maxBarThickness: 22,
        })),
      },
      options: {
        ...baseOptions<'bar'>(c, 'RMSE (Mbps)'),
        interaction: { mode: 'index', intersect: false },
        plugins: {
          ...baseOptions<'bar'>(c).plugins,
          legend: { ...baseOptions<'bar'>(c).plugins!.legend, labels: { ...legendRect(c) } },
        },
      },
    };
  });

  isBest(id: ModelId) {
    return this.data()?.overview.best_model === id;
  }
}

function legendRect(c: ChartColors) {
  return {
    color: c.inkSecondary,
    boxWidth: 12,
    boxHeight: 12,
    useBorderRadius: true,
    borderRadius: 2,
  };
}
