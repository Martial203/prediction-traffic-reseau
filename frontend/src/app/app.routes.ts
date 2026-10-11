import { inject } from '@angular/core';
import { Router, Routes } from '@angular/router';
import { map } from 'rxjs';

import { ApiService } from './core/api.service';

/** /cells redirige vers la première cellule connue de l'API. */
const toFirstCell = () => {
  const router = inject(Router);
  return inject(ApiService).cells$.pipe(
    map((cells) => router.createUrlTree(['/cells', cells[0]?.id ?? ''])),
  );
};

export const routes: Routes = [
  {
    path: '',
    title: "Vue d'ensemble · Trafic 4G",
    loadComponent: () => import('./pages/overview/overview.page').then((m) => m.OverviewPage),
  },
  { path: 'cells', pathMatch: 'full', canActivate: [toFirstCell], children: [] },
  {
    path: 'cells/:id',
    title: 'Cellule · Trafic 4G',
    loadComponent: () => import('./pages/cell/cell.page').then((m) => m.CellPage),
  },
  {
    path: 'models',
    title: 'Modèles · Trafic 4G',
    loadComponent: () => import('./pages/models/models.page').then((m) => m.ModelsPage),
  },
  { path: '**', redirectTo: '' },
];
