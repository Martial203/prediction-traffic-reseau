import { Injectable, signal } from '@angular/core';

import { ModelId } from './api.models';

/** Couleurs résolues depuis les variables CSS (styles.scss), pour Chart.js. */
export interface ChartColors {
  ink: string;
  inkSecondary: string;
  muted: string;
  grid: string;
  axis: string;
  surface: string;
  actual: string;
  band: string;
  testZone: string;
  critical: string;
  warning: string;
  model: Record<ModelId, string>;
}

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly media = window.matchMedia('(prefers-color-scheme: dark)');
  readonly colors = signal<ChartColors>(this.read());

  constructor() {
    // Les couleurs des graphiques suivent le thème clair/sombre du système.
    this.media.addEventListener('change', () => this.colors.set(this.read()));
  }

  private read(): ChartColors {
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    return {
      ink: v('--text-primary'),
      inkSecondary: v('--text-secondary'),
      muted: v('--text-muted'),
      grid: v('--grid'),
      axis: v('--axis'),
      surface: v('--surface-1'),
      actual: v('--series-actual'),
      band: v('--series-band'),
      testZone: v('--test-zone'),
      critical: v('--status-critical'),
      warning: v('--status-warning'),
      model: {
        random_forest: v('--series-1'),
        xgboost: v('--series-2'),
        lstm: v('--series-3'),
        seasonal_naive: v('--series-baseline'),
        persistence: v('--series-baseline-2'),
      },
    };
  }
}
