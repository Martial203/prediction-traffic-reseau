import { DecimalPipe } from '@angular/common';
import { Component, computed, effect, inject, input, signal } from '@angular/core';
import { toObservable, toSignal } from '@angular/core/rxjs-interop';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { ChartConfiguration, ChartDataset, LegendItem, TooltipItem } from 'chart.js';
import { combineLatest, switchMap } from 'rxjs';

import { ModelId } from '../../core/api.models';
import { ApiService } from '../../core/api.service';
import { ThemeService } from '../../core/theme.service';
import { ChartComponent } from '../../shared/chart.component';
import { baseOptions, nf, shortStamp } from '../../shared/chart-defaults';
import { asLoad } from '../../shared/load';

const MODEL_ORDER: ModelId[] = [
  'random_forest',
  'xgboost',
  'lstm',
  'seasonal_naive',
  'persistence',
];
const BASELINES: ModelId[] = ['seasonal_naive', 'persistence'];
/** Heures d'historique affichées avant le début du test en mode "Période de test". */
const CONTEXT_HOURS = 48;

@Component({
  selector: 'app-cell-page',
  imports: [ChartComponent, DecimalPipe, RouterLink, RouterLinkActive],
  templateUrl: './cell.page.html',
})
export class CellPage {
  private readonly api = inject(ApiService);
  private readonly theme = inject(ThemeService);

  readonly id = input.required<string>();

  readonly overview = toSignal(this.api.overview$);
  readonly cells = toSignal(this.api.cells$, { initialValue: [] });
  readonly metricsByCell = toSignal(this.api.metricsByCell$);

  readonly series = toSignal(
    toObservable(this.id).pipe(switchMap((id) => asLoad(this.api.series(id)))),
    {
      initialValue: { status: 'loading' } as const,
    },
  );
  readonly data = computed(() => {
    const s = this.series();
    return s.status === 'ok' ? s.data : null;
  });

  readonly info = computed(() => this.cells().find((c) => c.id === this.id()));
  readonly models = computed(() => {
    const ov = this.overview();
    return ov ? MODEL_ORDER.map((id) => ov.models.find((m) => m.id === id)!).filter(Boolean) : [];
  });

  readonly visible = signal<Set<ModelId>>(new Set(['random_forest', 'xgboost', 'lstm']));
  readonly range = signal<'test' | 'all'>('test');
  readonly forecastModel = signal<ModelId>('xgboost');
  readonly horizon = signal(24);
  readonly horizons = [12, 24, 48, 72];

  constructor() {
    // Par défaut, la prévision utilise le meilleur modèle.
    effect(() => {
      const best = this.overview()?.best_model;
      if (best) this.forecastModel.set(best);
    });
  }

  toggle(id: ModelId) {
    const next = new Set(this.visible());
    next.has(id) ? next.delete(id) : next.add(id);
    this.visible.set(next);
  }

  readonly forecast = toSignal(
    combineLatest([
      toObservable(this.id),
      toObservable(this.forecastModel),
      toObservable(this.horizon),
    ]).pipe(switchMap(([id, model, h]) => asLoad(this.api.forecast(id, model, h)))),
    { initialValue: { status: 'loading' } as const },
  );

  readonly cellMetrics = computed(() => {
    const by = this.metricsByCell();
    return by
      ? this.models()
          .map((m) => ({ ...m, cell: by[m.id][this.id()] }))
          .sort((a, b) => a.cell.rmse - b.cell.rmse)
      : [];
  });

  private readonly window = computed(() => {
    const d = this.data();
    if (!d) return null;
    const testIdx = d.timestamps.indexOf(d.test_start);
    const from = this.range() === 'test' ? Math.max(testIdx - CONTEXT_HOURS, 0) : 0;
    const cut = <T>(a: T[]) => a.slice(from);
    return { d, from, testIdx: testIdx - from, cut };
  });

  readonly mainChart = computed<ChartConfiguration<'line'> | null>(() => {
    const w = this.window();
    if (!w) return null;
    const { d, cut } = w;
    const c = this.theme.colors();
    const actual = cut(d.actual);
    const events = cut(d.event);
    const marker = (ev: string) => actual.map((v, i) => (events[i] === ev ? v : null));
    const label = (id: ModelId) => this.models().find((m) => m.id === id)?.label ?? id;

    const datasets: ChartDataset<'line'>[] = [
      {
        label: 'Trafic réel',
        data: actual,
        borderColor: c.actual,
        backgroundColor: c.actual,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        order: 0,
      },
      ...MODEL_ORDER.filter((id) => this.visible().has(id) && d.predictions[id]).map(
        (id): ChartDataset<'line'> => ({
          label: label(id),
          data: cut(d.predictions[id]),
          borderColor: c.model[id],
          backgroundColor: c.model[id],
          borderWidth: 2,
          borderDash: BASELINES.includes(id) ? [5, 4] : undefined,
          pointRadius: 0,
          tension: 0.25,
          spanGaps: false,
          order: 1,
        }),
      ),
      eventDataset('Congestion', marker('congestion'), c.critical, 'triangle', c.surface),
      eventDataset('Handover', marker('handover'), c.warning, 'rectRot', c.surface),
    ];

    const opts = baseOptions<'line'>(c, 'Trafic (Mbps)');
    return {
      type: 'line',
      data: { labels: cut(d.timestamps).map(shortStamp), datasets },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          zone: {
            from: w.testIdx,
            color: c.testZone,
            label: 'Période de test',
            labelColor: c.muted,
          },
        },
      },
    };
  });

  readonly usersChart = computed(() =>
    this.kpiChart('active_users', 'Utilisateurs actifs', 'utilisateurs'),
  );
  readonly sinrChart = computed(() => this.kpiChart('sinr', 'SINR', 'dB'));

  private kpiChart(
    key: 'active_users' | 'sinr',
    label: string,
    unit: string,
  ): ChartConfiguration<'line'> | null {
    const w = this.window();
    if (!w) return null;
    const c = this.theme.colors();
    const opts = baseOptions<'line'>(c, unit);
    return {
      type: 'line',
      data: {
        labels: w.cut(w.d.timestamps).map(shortStamp),
        datasets: [
          {
            label,
            data: w.cut(w.d[key] as (number | null)[]),
            borderColor: c.inkSecondary,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.25,
          },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          legend: { display: false },
          zone: { from: w.testIdx, color: c.testZone },
        },
      },
    };
  }

  readonly forecastChart = computed<ChartConfiguration<'line'> | null>(() => {
    const f = this.forecast();
    const d = this.data();
    if (f.status !== 'ok' || !d) return null;
    const fc = f.data;
    const c = this.theme.colors();
    const hist = 48;
    const hTs = d.timestamps.slice(-hist);
    const hVals = d.actual.slice(-hist);
    const pad = (n: number) => Array<null>(n).fill(null);
    // Le dernier point réel est répété pour raccorder visuellement la prévision.
    const join = hVals[hVals.length - 1];
    const label = this.models().find((m) => m.id === fc.model)?.label ?? fc.model;
    const opts = baseOptions<'line'>(c, 'Trafic (Mbps)');
    return {
      type: 'line',
      data: {
        labels: [...hTs, ...fc.timestamps].map(shortStamp),
        datasets: [
          {
            label: 'Trafic réel',
            data: [...hVals, ...pad(fc.values.length)],
            borderColor: c.actual,
            backgroundColor: c.actual,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
          },
          {
            label: 'Borne basse ~95 %',
            data: [...pad(hist), ...fc.lower],
            borderWidth: 0,
            pointRadius: 0,
            tension: 0.25,
          },
          {
            label: 'Intervalle ~95 %',
            data: [...pad(hist), ...fc.upper],
            borderWidth: 0,
            borderColor: c.band,
            pointStyle: 'rect',
            pointRadius: 0,
            backgroundColor: c.band,
            fill: '-1',
            tension: 0.25,
          },
          {
            label: `Prévision ${label}`,
            data: [...pad(hist - 1), join, ...fc.values],
            borderColor: c.model[fc.model],
            backgroundColor: c.model[fc.model],
            borderWidth: 2,
            borderDash: [6, 3],
            pointRadius: 0,
            tension: 0.25,
          },
        ],
      },
      options: {
        ...opts,
        plugins: {
          ...opts.plugins,
          legend: {
            ...opts.plugins!.legend,
            labels: {
              ...opts.plugins!.legend!.labels,
              filter: (item: LegendItem) => !item.text.startsWith('Borne basse'),
            },
          },
          tooltip: {
            ...opts.plugins!.tooltip,
            filter: (item: TooltipItem<'line'>) => item.parsed.y != null && item.datasetIndex !== 1,
            callbacks: {
              label: (ctx: TooltipItem<'line'>) => {
                if (ctx.datasetIndex !== 2)
                  return ` ${ctx.dataset.label} : ${nf.format(ctx.parsed.y ?? 0)}`;
                const low = fc.lower[ctx.dataIndex - hist];
                return ` Intervalle ~95 % : ${nf.format(low)} – ${nf.format(ctx.parsed.y ?? 0)}`;
              },
            },
          },
          zone: { from: hist, color: c.testZone, label: 'Prévision', labelColor: c.muted },
        },
      },
    };
  });

  readonly forecastSummary = computed(() => {
    const f = this.forecast();
    if (f.status !== 'ok') return null;
    const v = f.data.values;
    const peak = Math.max(...v);
    const at = f.data.timestamps[v.indexOf(peak)];
    return { mean: v.reduce((a, b) => a + b, 0) / v.length, peak, peakAt: shortStamp(at) };
  });
}

function eventDataset(
  label: string,
  data: (number | null)[],
  color: string,
  pointStyle: 'triangle' | 'rectRot',
  ring: string,
): ChartDataset<'line'> {
  return {
    label,
    data,
    showLine: false,
    pointStyle,
    pointRadius: 5,
    pointHoverRadius: 7,
    pointBackgroundColor: color,
    pointBorderColor: ring,
    pointBorderWidth: 1.5,
    borderColor: color,
    backgroundColor: color,
    order: -1,
  };
}
