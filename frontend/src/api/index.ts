import { request } from './client';
import type {
  Analysis,
  Example,
  GenerateOptions,
  GenerateResult,
  GraphDocument,
  LayerCatalog,
  Project,
  Tokens,
  User,
} from './types';

export * from './types';
export { ApiError, getTokens, onTokensChange, setTokens } from './client';

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
};
