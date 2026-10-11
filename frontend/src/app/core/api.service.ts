import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { shareReplay } from 'rxjs';

import {
  CellInfo,
  CellSeries,
  Forecast,
  LstmHistory,
  MetricsBy,
  ModelId,
  Overview,
} from './api.models';

/** Client de l'API FastAPI, servie sous /api (proxy en développement). */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly base = '/api';

  // Données statiques après l'entraînement : mises en cache pour la session.
  readonly overview$ = this.http.get<Overview>(`${this.base}/overview`).pipe(shareReplay(1));
  readonly cells$ = this.http.get<CellInfo[]>(`${this.base}/cells`).pipe(shareReplay(1));
  readonly metricsByCell$ = this.http
    .get<MetricsBy>(`${this.base}/metrics/by-cell`)
    .pipe(shareReplay(1));
  readonly metricsByEvent$ = this.http
    .get<MetricsBy>(`${this.base}/metrics/by-event`)
    .pipe(shareReplay(1));
  readonly lstmHistory$ = this.http
    .get<LstmHistory>(`${this.base}/lstm/history`)
    .pipe(shareReplay(1));

  featureImportance(model: ModelId) {
    return this.http.get<Record<string, number>>(`${this.base}/feature-importance/${model}`);
  }

  series(cell: string) {
    return this.http.get<CellSeries>(`${this.base}/cells/${encodeURIComponent(cell)}/series`);
  }

  forecast(cell: string, model: ModelId, horizon: number) {
    const params = new HttpParams().set('model', model).set('horizon', horizon);
    return this.http.get<Forecast>(`${this.base}/cells/${encodeURIComponent(cell)}/forecast`, {
      params,
    });
  }
}
