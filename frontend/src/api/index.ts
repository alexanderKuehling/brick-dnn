import { DEMO } from '@/demo';
import { request, requestFile } from './client';
import { createDemoApi } from './demoApi';
import type {
  Analysis,
  Compatibility,
  DataPrep,
  DataPrepInput,
  Dataset,
  DatasetRef,
  DataSource,
  DataStatus,
  Example,
  GenerateOptions,
  GenerateResult,
  GraphDocument,
  ImportOptions,
  LayerCatalog,
  PrepDownloadFormat,
  PreviewRequest,
  PreviewResult,
  Project,
  RunDetail,
  RunInput,
  RunSummary,
  StepCatalog,
  Tokens,
  TrainingStatus,
  User,
} from './types';

export * from './types';
export { ApiError, getTokens, onTokensChange, setTokens, type FileDownload } from './client';

type ProjectInput = Pick<Project, 'name' | 'graph'> & Partial<Pick<Project, 'description'>>;

const serverApi = {
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

  /** Data preparation (the Data tab). */
  data: {
    /** What the server can do (public, no login needed). */
    status: (signal?: AbortSignal) => request<DataStatus>('/api/data/', { signal }),
    /** All step types with their settings (public). */
    steps: (signal?: AbortSignal) => request<StepCatalog>('/api/data/steps/', { signal }),
    sources: {
      list: (signal?: AbortSignal) =>
        request<DataSource[]>('/api/data/sources/', { auth: true, signal }),
      get: (id: number, signal?: AbortSignal) =>
        request<DataSource>(`/api/data/sources/${id}/`, { auth: true, signal }),
      /** Import a CSV file; options left out are detected. */
      upload: (
        file: File,
        options: Partial<ImportOptions> & { name?: string } = {},
        signal?: AbortSignal
      ) => {
        const form = new FormData();
        form.append('file', file);
        for (const [key, value] of Object.entries(options)) {
          if (value === undefined || value === null) {
            continue;
          }
          form.append(key, Array.isArray(value) ? JSON.stringify(value) : String(value));
        }
        return request<DataSource>('/api/data/sources/', {
          method: 'POST',
          body: form,
          auth: true,
          signal,
        });
      },
      rename: (id: number, name: string) =>
        request<DataSource>(`/api/data/sources/${id}/`, {
          method: 'PATCH',
          body: { name },
          auth: true,
        }),
      remove: (id: number) =>
        request<void>(`/api/data/sources/${id}/`, { method: 'DELETE', auth: true }),
    },
    preps: {
      list: (signal?: AbortSignal) =>
        request<DataPrep[]>('/api/data/preps/', { auth: true, signal }),
      get: (id: number, signal?: AbortSignal) =>
        request<DataPrep>(`/api/data/preps/${id}/`, { auth: true, signal }),
      create: (data: DataPrepInput) =>
        request<DataPrep>('/api/data/preps/', { method: 'POST', body: data, auth: true }),
      update: (id: number, data: Partial<DataPrepInput>) =>
        request<DataPrep>(`/api/data/preps/${id}/`, { method: 'PATCH', body: data, auth: true }),
      remove: (id: number) =>
        request<void>(`/api/data/preps/${id}/`, { method: 'DELETE', auth: true }),
      /** Run the (possibly unsaved) pipeline and return reports, profile and a sample. */
      preview: (id: number, body: PreviewRequest = {}, signal?: AbortSignal) =>
        request<PreviewResult>(`/api/data/preps/${id}/preview/`, {
          method: 'POST',
          body,
          auth: true,
          signal,
        }),
      /** Turn the saved preparation into a training dataset (it appears under "Your data"). */
      export: (id: number, name?: string) =>
        request<Dataset>(`/api/data/preps/${id}/export/`, {
          method: 'POST',
          body: name ? { name } : {},
          auth: true,
        }),
      /** The prepared data (csv), the recipe (JSON) or a pandas script (python). */
      download: (id: number, format: PrepDownloadFormat) =>
        requestFile(`/api/data/preps/${id}/download/?format=${format}`, { auth: true }),
    },
  },
};

export type Api = typeof serverApi;

/** In the online demo, the code generator runs in the browser instead of on a server. */
export const api: Api = DEMO ? createDemoApi(serverApi) : serverApi;
