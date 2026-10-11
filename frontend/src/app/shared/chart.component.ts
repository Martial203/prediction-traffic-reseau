import { Component, ElementRef, OnDestroy, effect, input, viewChild } from '@angular/core';
import { Chart, ChartConfiguration, ChartType, Plugin, registerables } from 'chart.js';

Chart.register(...registerables);
// Les séries en ligne apparaissent comme un trait dans la légende ; les marqueurs
// d'événements (triangle, losange) gardent leur propre forme.
Chart.defaults.elements.point.pointStyle = 'line';
Chart.defaults.elements.point.hoverRadius = 0;

export type AnyChartConfig = ChartConfiguration<'line'> | ChartConfiguration<'bar'>;

export interface ZoneOptions {
  /** Index (label) à partir duquel la zone commence. */
  from?: number;
  color?: string;
  label?: string;
  labelColor?: string;
}

declare module 'chart.js' {
  interface PluginOptionsByType<TType extends ChartType> {
    zone?: ZoneOptions;
  }
}

/** Ombre la zone [from, fin] de l'axe X (période de test / prévision). */
const zonePlugin: Plugin<ChartType, ZoneOptions> = {
  id: 'zone',
  beforeDatasetsDraw(chart, _args, opts) {
    if (opts?.from == null || opts.from < 0) return;
    const { ctx, chartArea, scales } = chart;
    const x = scales['x'].getPixelForValue(opts.from);
    if (x > chartArea.right) return;
    const left = Math.max(x, chartArea.left);
    ctx.save();
    ctx.fillStyle = opts.color ?? 'rgba(0,0,0,0.04)';
    ctx.fillRect(left, chartArea.top, chartArea.right - left, chartArea.bottom - chartArea.top);
    if (opts.label) {
      ctx.fillStyle = opts.labelColor ?? '#888';
      ctx.font = '12px system-ui, "Segoe UI", sans-serif';
      ctx.fillText(opts.label, left + 6, chartArea.top + 14);
    }
    ctx.restore();
  },
};

/** Fine enveloppe Chart.js : recrée le graphique quand la config change. */
@Component({
  selector: 'app-chart',
  template: `<div class="chart-box" [style.height.px]="height()">
    <canvas #canvas role="img" [attr.aria-label]="ariaLabel()"></canvas>
  </div>`,
  styles: `
    .chart-box {
      position: relative;
      width: 100%;
    }
  `,
})
export class ChartComponent implements OnDestroy {
  readonly config = input.required<AnyChartConfig>();
  readonly height = input(320);
  readonly ariaLabel = input('Graphique');

  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private chart?: Chart;

  constructor() {
    effect(() => {
      const config = this.config();
      this.chart?.destroy();
      this.chart = new Chart(this.canvas().nativeElement, {
        ...config,
        plugins: [...((config.plugins as Plugin[]) ?? []), zonePlugin],
      } as ChartConfiguration);
    });
  }

  ngOnDestroy() {
    this.chart?.destroy();
  }
}
