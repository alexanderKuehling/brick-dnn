import { request, requestFile } from './client';
import type {
  Analysis,
  Compatibility,
  Dataset,
  DatasetRef,
  Example,
  GenerateOptions,
  GenerateResult,
  GraphDocument,
  LayerCatalog,
  Project,
  RunDetail,
  RunInput,
  RunSummary,
  Tokens,
  TrainingStatus,
  User,
} from './types';

export * from './types';
export { ApiError, getTokens, onTokensChange, setTokens, type FileDownload } from './client';

type ProjectInput = Pick<Project, 'name' | 'graph'> & Partial<Pick<Project, 'description'>>;

export const api = {
  layers: () => request<LayerCatalog>('/api/layers/'),
  examples: () => request<Example[]>('/api/examples/'),
  analyze: (graph: GraphDocument, signal?: AbortSignal) =>
    request<Analysis>('/api/analyze/', { method: 'POST', body: { graph }, signal }),
  generate: (graph: GraphDocument, options: GenerateOptions = {}) =>
    request<GenerateResult>('/api/generate/', { method: 'POST', body: { graph, options } }),

  login: (username: string, password: string) =>
    request<Tokens>('/api/auth/token/', { method: 'POST', body: { username, password } }),
  register: (username: string, email: string, password: string) =>
    request<Tokens & { user: User }>('/api/auth/register/', {
      method: 'POST',
      body: { username, email, password },
    }),
  logout: (refresh: string) =>
    request<void>('/api/auth/logout/', { method: 'POST', body: { refresh } }),
  me: () => request<User>('/api/auth/me/', { auth: true }),

  projects: {
    list: () => request<Project[]>('/api/projects/', { auth: true }),
    get: (id: number) => request<Project>(`/api/projects/${id}/`, { auth: true }),
    create: (data: ProjectInput) =>
      request<Project>('/api/projects/', { method: 'POST', body: data, auth: true }),
    update: (id: number, data: Partial<ProjectInput>) =>
      request<Project>(`/api/projects/${id}/`, { method: 'PATCH', body: data, auth: true }),
    remove: (id: number) => request<void>(`/api/projects/${id}/`, { method: 'DELETE', auth: true }),
  },

  training: {
    /** What the server can do (public, no login needed). */
    status: (signal?: AbortSignal) => request<TrainingStatus>('/api/training/', { signal }),
    datasets: {
      list: (signal?: AbortSignal) =>
        request<Dataset[]>('/api/training/datasets/', { auth: true, signal }),
      upload: (file: File, name?: string) => {
        const form = new FormData();
        form.append('file', file);
        if (name) {
          form.append('name', name);
        }
        return request<Dataset>('/api/training/datasets/', {
          method: 'POST',
          body: form,
          auth: true,
        });
      },
      remove: (id: number) =>
        request<void>(`/api/training/datasets/${id}/`, { method: 'DELETE', auth: true }),
    },
    check: (graph: GraphDocument, dataset: DatasetRef, signal?: AbortSignal) =>
      request<Compatibility>('/api/training/check/', {
        method: 'POST',
        body: { graph, dataset },
        auth: true,
        signal,
      }),
    runs: {
      list: (signal?: AbortSignal) =>
        request<RunSummary[]>('/api/training/runs/', { auth: true, signal }),
      get: (id: string, signal?: AbortSignal) =>
        request<RunDetail>(`/api/training/runs/${id}/`, { auth: true, signal }),
      create: (data: RunInput) =>
        request<RunDetail>('/api/training/runs/', { method: 'POST', body: data, auth: true }),
      stop: (id: string) =>
        request<RunDetail>(`/api/training/runs/${id}/stop/`, { method: 'POST', auth: true }),
      remove: (id: string) =>
        request<void>(`/api/training/runs/${id}/`, { method: 'DELETE', auth: true }),
      downloadWeights: (id: string) =>
        requestFile(`/api/training/runs/${id}/weights/`, { auth: true }),
    },
  },
};
