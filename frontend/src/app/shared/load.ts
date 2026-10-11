import { HttpErrorResponse } from '@angular/common/http';
import { Observable, catchError, map, of, startWith } from 'rxjs';

export type Load<T> =
  { status: 'loading' } | { status: 'ok'; data: T } | { status: 'error'; message: string };

/** Transforme un appel HTTP en état chargement / succès / erreur affichable. */
export function asLoad<T>(source: Observable<T>): Observable<Load<T>> {
  return source.pipe(
    map((data) => ({ status: 'ok', data }) as Load<T>),
    catchError((err: unknown) => of({ status: 'error', message: errorMessage(err) } as Load<T>)),
    startWith({ status: 'loading' } as Load<T>),
  );
}

export function errorMessage(err: unknown): string {
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0 || err.status === 504) {
      return "L'API ne répond pas. Lancez le backend : cd backend puis uvicorn app.api:app --port 8000";
    }
    return err.error?.detail ?? `Erreur ${err.status}`;
  }
  return String(err);
}
