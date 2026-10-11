import { ChartOptions } from 'chart.js';

type CartesianType = 'line' | 'bar';

import { ChartColors } from '../core/theme.service';

export const FONT = { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12 };

export const nf = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 1 });

/** "2024-01-24T13:00:00" -> "24/01 13h" */
export function shortStamp(iso: string): string {
  return `${iso.slice(8, 10)}/${iso.slice(5, 7)} ${iso.slice(11, 13)}h`;
}

/** Options communes : grille discrète, tooltip sur toutes les séries à l'abscisse survolée. */
export function baseOptions<T extends CartesianType>(
  c: ChartColors,
  yTitle?: string,
): ChartOptions<T> {
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        align: 'start',
        labels: {
          color: c.inkSecondary,
          font: FONT,
          usePointStyle: true,
          pointStyleWidth: 18,
          boxHeight: 9,
          sort: (a, b) => (a.datasetIndex ?? 0) - (b.datasetIndex ?? 0),
        },
      },
      tooltip: {
        backgroundColor: c.surface,
        titleColor: c.ink,
        bodyColor: c.inkSecondary,
        borderColor: c.axis,
        borderWidth: 1,
        padding: 10,
        filter: (item) => item.parsed.y != null,
        titleFont: { ...FONT, weight: 'bold' },
        bodyFont: FONT,
        callbacks: {
          label: (ctx) =>
            ctx.parsed.y == null ? '' : ` ${ctx.dataset.label} : ${nf.format(ctx.parsed.y)}`,
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        border: { color: c.axis },
        ticks: { color: c.muted, font: FONT, maxRotation: 0, autoSkipPadding: 24 },
      },
      y: {
        grid: { color: c.grid },
        border: { display: false },
        ticks: { color: c.muted, font: FONT },
        title: yTitle ? { display: true, text: yTitle, color: c.muted, font: FONT } : undefined,
      },
    },
  };
  return options as ChartOptions<T>;
}
