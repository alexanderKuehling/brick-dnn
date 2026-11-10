import { DEMO } from '@/demo';
import { request, requestFile, type UploadOptions } from './client';
import { createDemoApi } from './demoApi';
import type {
  Account,
  AgentActivityPage,
  AgentActivityQuery,
  AgentActivitySummary,
  Analysis,
  ApiToken,
  ApiTokenInput,
  BillingCheckout,
  BillingInfo,
  BillingState,
  CommentInput,
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
  LegalInfo,
  LegalPage,
  LegalPageName,
  MemberRole,
  NewApiToken,
  OAuthConnection,
  OAuthDecision,
  OAuthRequest,
  PlanChangePreview,
  PredictionForm,
  PredictionRequest,
  PredictionResult,
  PrepDownloadFormat,
  PreviewRequest,
  PreviewResult,
  Project,
  ProjectComment,
  ProjectData,
  ProjectDataKind,
  ProjectMember,
  ProjectMembers,
  ProjectVersion,
  ProjectVersionDetail,
  RunDetail,
  RunInput,
  RunSummary,
  SharedNetwork,
  ShareLink,
  StepCatalog,
  Tokens,
  TrainingConfig,
  TrainingStatus,
  TuningInput,
  Usage,
  User,
} from './types';

export * from './types';
export {
  ApiError,
  apiBaseUrl,
  freshAccessToken,
  getTokens,
  onTokensChange,
  setTokens,
  type FileDownload,
  type UploadOptions,
} from './client';

type ProjectInput = Pick<Project, 'name' | 'graph'> & Partial<Pick<Project, 'description'>>;

/** "?a=1&b=x" for the values that are set, "" for none. */
function queryString(params: object): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      search.set(key, String(value));
    }
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

const serverApi = {
  layers: () => request<LayerCatalog>('/api/layers/'),
  examples: () => request<Example[]>('/api/examples/'),
  analyze: (graph: GraphDocument, signal?: AbortSignal) =>
    request<Analysis>('/api/analyze/', { method: 'POST', body: { graph }, signal }),
  generate: (graph: GraphDocument, options: GenerateOptions = {}) =>
    request<GenerateResult>('/api/generate/', { method: 'POST', body: { graph, options } }),

  login: (username: string, password: string) =>
    request<Tokens>('/api/auth/token/', { method: 'POST', body: { username, password } }),
  /** `acceptTerms` is required when the server has terms of use (LegalInfo). */
  register: (username: string, email: string, password: string, acceptTerms = false) =>
    request<Tokens & { user: User }>('/api/auth/register/', {
      method: 'POST',
      body: { username, email, password, ...(acceptTerms ? { accept_terms: true } : {}) },
    }),
  logout: (refresh: string) =>
    request<void>('/api/auth/logout/', { method: 'POST', body: { refresh } }),
  me: () => request<User>('/api/auth/me/', { auth: true }),

  /** The own account (website login only, not API tokens). */
  account: {
    get: () => request<Account>('/api/auth/account/', { auth: true }),
    changeEmail: (email: string) =>
      request<Account>('/api/auth/account/', { method: 'PATCH', body: { email }, auth: true }),
    /** Ends all other logins; the answer holds new tokens for this one. */
    changePassword: (currentPassword: string, newPassword: string) =>
      request<Tokens>('/api/auth/account/password/', {
        method: 'POST',
        body: { current_password: currentPassword, new_password: newPassword },
        auth: true,
      }),
    /** A ZIP file with everything stored about the account. */
    exportData: () => requestFile('/api/auth/account/export/', { auth: true }),
    /** Deletes the account and everything it owns, for good. */
    remove: (password: string, username: string) =>
      request<void>('/api/auth/account/delete/', {
        method: 'POST',
        body: { password, username },
        auth: true,
      }),
    /** This month's usage (compute and storage), the limits and what is left. */
    usage: (signal?: AbortSignal) => request<Usage>('/api/account/usage/', { auth: true, signal }),
    /** The monthly budget for the own account in CPU minutes; null removes it. */
    setBudget: (cpuMinutes: number | null) =>
      request<Usage>('/api/account/usage/', {
        method: 'PATCH',
        body: { budget_cpu_minutes: cpuMinutes },
        auth: true,
      }),
    /** Compute beyond the plan's allowance (to be billed), up to a cap in CPU hours. */
    setOverage: (enabled: boolean, capCpuHours: number | null) =>
      request<Usage>('/api/account/usage/', {
        method: 'PATCH',
        body: { overage_enabled: enabled, overage_cap_cpu_hours: capCpuHours },
        auth: true,
      }),
  },

  /** Paid plans (Paddle). Only `info` answers on a server that sells none. */
  billing: {
    /** Public: whether the server sells plans, the plans and what the checkout needs. */
    info: () => request<BillingInfo>('/api/billing/'),
    /** The account's plan and subscription (website login only, like all below). */
    state: (signal?: AbortSignal) =>
      request<BillingState>('/api/billing/subscription/', { auth: true, signal }),
    /** What Paddle's checkout needs to sell `plan` to this account. */
    checkout: (plan: string) =>
      request<BillingCheckout>('/api/billing/checkout/', {
        method: 'POST',
        body: { plan },
        auth: true,
      }),
    /** What moving the subscription to `plan` would cost; changes nothing. */
    previewChange: (plan: string) =>
      request<PlanChangePreview>('/api/billing/change-plan/', {
        method: 'POST',
        body: { plan, preview: true },
        auth: true,
      }),
    changePlan: (plan: string) =>
      request<BillingState>('/api/billing/change-plan/', {
        method: 'POST',
        body: { plan },
        auth: true,
      }),
    /** Cancels to the end of the period that is paid for. */
    cancel: () => request<BillingState>('/api/billing/cancel/', { method: 'POST', auth: true }),
    /** Takes a cancellation back. */
    reactivate: () =>
      request<BillingState>('/api/billing/reactivate/', { method: 'POST', auth: true }),
    /** A link to Paddle's customer portal (payment method, invoices), valid for a short time. */
    portal: () => request<{ url: string }>('/api/billing/portal/', { method: 'POST', auth: true }),
    /** Asks Paddle for the subscription now, instead of waiting for its notification. */
    refresh: () => request<BillingState>('/api/billing/refresh/', { method: 'POST', auth: true }),
  },

  /** Imprint, privacy policy and terms of use, as far as the server has them (public). */
  legal: {
    info: () => request<LegalInfo>('/api/legal/'),
    page: (page: LegalPageName) => request<LegalPage>(`/api/legal/${page}/`),
  },

  /** Personal API tokens. Only the normal login can manage them, not the tokens themselves. */
  apiTokens: {
    /** Newest first. */
    list: () => request<ApiToken[]>('/api/auth/api-tokens/', { auth: true }),
    create: (data: ApiTokenInput) =>
      request<NewApiToken>('/api/auth/api-tokens/', { method: 'POST', body: data, auth: true }),
    /** Revoked tokens stop working immediately. */
    revoke: (id: number) =>
      request<void>(`/api/auth/api-tokens/${id}/`, { method: 'DELETE', auth: true }),
    /** The token's monthly compute budget in CPU minutes; null removes it. */
    setBudget: (id: number, cpuMinutes: number | null) =>
      request<ApiToken>(`/api/auth/api-tokens/${id}/`, {
        method: 'PATCH',
        body: { budget_cpu_minutes: cpuMinutes },
        auth: true,
      }),
  },

  /**
   * Signing in apps (OAuth authorization code flow) on the page /oauth/authorize. `query` is
   * the query string of that page without the "?", passed on unchanged. A rejected request
   * fails with HTTP 400 and an OAuthRequestError as `data`.
   */
  oauth: {
    /** Checks the request and says which app asks for what. */
    validate: (query: string) =>
      request<OAuthRequest>(`/api/oauth/authorize/${query ? `?${query}` : ''}`, { auth: true }),
    /** Allows (with or without changes) or denies the request. */
    decide: (query: string, approve: boolean, allowWrite: boolean) =>
      request<OAuthDecision>('/api/oauth/authorize/', {
        method: 'POST',
        body: { query, approve, allow_write: allowWrite },
        auth: true,
      }),
  },

  /** Apps the user allowed through OAuth (see `oauth`). */
  connections: {
    /** Newest first. */
    list: () => request<OAuthConnection[]>('/api/auth/connections/', { auth: true }),
    /** The app loses access immediately (its tokens are revoked). */
    revoke: (id: number) =>
      request<void>(`/api/auth/connections/${id}/`, { method: 'DELETE', auth: true }),
    /** The app's monthly compute budget in CPU minutes; null removes it. */
    setBudget: (id: number, cpuMinutes: number | null) =>
      request<OAuthConnection>(`/api/auth/connections/${id}/`, {
        method: 'PATCH',
        body: { budget_cpu_minutes: cpuMinutes },
        auth: true,
      }),
  },

  /** What AI agents did with the user's API tokens and apps (website login only). */
  agentActivity: {
    /** Newest first; `after` for polling, `before` for older pages. */
    list: (query: AgentActivityQuery = {}, signal?: AbortSignal) =>
      request<AgentActivityPage>(`/api/agent-activity/${queryString(query)}`, {
        auth: true,
        signal,
      }),
    /** The tokens and apps that acted, the tools used and the latest entry. */
    summary: (signal?: AbortSignal) =>
      request<AgentActivitySummary>('/api/agent-activity/summary/', { auth: true, signal }),
  },

  projects: {
    /** Own and shared projects (each with the user's `role`), newest first. */
    list: (scope?: 'mine' | 'shared') =>
      request<Project[]>(`/api/projects/${scope ? `?scope=${scope}` : ''}`, { auth: true }),
    get: (id: number) => request<Project>(`/api/projects/${id}/`, { auth: true }),
    create: (data: ProjectInput) =>
      request<Project>('/api/projects/', { method: 'POST', body: data, auth: true }),
    /**
     * Owners and editors. With `version`, a save based on an older version fails with HTTP 409
     * and a ProjectConflict; without it, the save overwrites.
     */
    update: (id: number, data: Partial<ProjectInput> & { version?: number }) =>
      request<Project>(`/api/projects/${id}/`, { method: 'PATCH', body: data, auth: true }),
    /** Owner only. */
    remove: (id: number) => request<void>(`/api/projects/${id}/`, { method: 'DELETE', auth: true }),
    /** A copy owned by the current user, named "<name> (copy)". Any role. */
    duplicate: (id: number) =>
      request<Project>(`/api/projects/${id}/duplicate/`, { method: 'POST', auth: true }),
    members: {
      list: (projectId: number) =>
        request<ProjectMembers>(`/api/projects/${projectId}/members/`, { auth: true }),
      /** Owner only. `user` is a username or an e-mail address. */
      add: (projectId: number, user: string, role: MemberRole) =>
        request<ProjectMember>(`/api/projects/${projectId}/members/`, {
          method: 'POST',
          body: { user, role },
          auth: true,
        }),
      /** Owner only. */
      update: (projectId: number, memberId: number, role: MemberRole) =>
        request<ProjectMember>(`/api/projects/${projectId}/members/${memberId}/`, {
          method: 'PATCH',
          body: { role },
          auth: true,
        }),
      /** The owner removes anyone; members remove themselves (leave). */
      remove: (projectId: number, memberId: number) =>
        request<void>(`/api/projects/${projectId}/members/${memberId}/`, {
          method: 'DELETE',
          auth: true,
        }),
    },
    /** Training datasets and data preparations shared in a project. */
    data: {
      /** Any role. */
      list: (projectId: number, signal?: AbortSignal) =>
        request<ProjectData>(`/api/projects/${projectId}/data/`, { auth: true, signal }),
      /** Owners and editors of the project share one of their own items there. */
      add: (projectId: number, kind: ProjectDataKind, id: number) =>
        request<ProjectData>(`/api/projects/${projectId}/data/`, {
          method: 'POST',
          body: { kind, id },
          auth: true,
        }),
      /** Stop sharing an item there (its owner or the project's owner). */
      remove: (projectId: number, kind: ProjectDataKind, id: number) =>
        request<void>(`/api/projects/${projectId}/data/${kind}/${id}/`, {
          method: 'DELETE',
          auth: true,
        }),
    },
    /** Owner only: turn the view link on or off, or replace it (the old one stops working). */
    setLink: (projectId: number, enabled: boolean, regenerate = false) =>
      request<ShareLink>(`/api/projects/${projectId}/link/`, {
        method: 'POST',
        body: regenerate ? { enabled, regenerate } : { enabled },
        auth: true,
      }),
    /** The history: every change keeps a copy of the new state. */
    versions: {
      /** Newest first. Any role. */
      list: (projectId: number, signal?: AbortSignal) =>
        request<ProjectVersion[]>(`/api/projects/${projectId}/versions/`, { auth: true, signal }),
      /** One version with its graph. Any role. */
      get: (projectId: number, versionId: number, signal?: AbortSignal) =>
        request<ProjectVersionDetail>(`/api/projects/${projectId}/versions/${versionId}/`, {
          auth: true,
          signal,
        }),
      /** Owner and editors: keep the current state (on the server) under a name. */
      create: (projectId: number, label: string) =>
        request<ProjectVersion>(`/api/projects/${projectId}/versions/`, {
          method: 'POST',
          body: { label },
          auth: true,
        }),
      /** Owner and editors: (re)name a version; named versions are kept for good. */
      rename: (projectId: number, versionId: number, label: string) =>
        request<ProjectVersion>(`/api/projects/${projectId}/versions/${versionId}/`, {
          method: 'PATCH',
          body: { label },
          auth: true,
        }),
      /**
       * Owner and editors: make the version the current state (a new entry in the history).
       * Resolves to the updated project; live editors get a `replaced` message.
       */
      restore: (projectId: number, versionId: number) =>
        request<Project>(`/api/projects/${projectId}/versions/${versionId}/restore/`, {
          method: 'POST',
          auth: true,
        }),
    },
    /** Comments on the network and its layers. Everyone with access may comment. */
    comments: {
      /** Oldest first; `node` limits them to one layer. */
      list: (projectId: number, node?: string, signal?: AbortSignal) =>
        request<ProjectComment[]>(
          `/api/projects/${projectId}/comments/${
            node === undefined ? '' : `?node=${encodeURIComponent(node)}`
          }`,
          { auth: true, signal }
        ),
      create: (projectId: number, data: CommentInput) =>
        request<ProjectComment>(`/api/projects/${projectId}/comments/`, {
          method: 'POST',
          body: data,
          auth: true,
        }),
      /** `text`: the author · `resolved`: the author, editors and the owner. */
      update: (projectId: number, commentId: number, data: { text?: string; resolved?: boolean }) =>
        request<ProjectComment>(`/api/projects/${projectId}/comments/${commentId}/`, {
          method: 'PATCH',
          body: data,
          auth: true,
        }),
      /** The author or the owner; replies are deleted with it. */
      remove: (projectId: number, commentId: number) =>
        request<void>(`/api/projects/${projectId}/comments/${commentId}/`, {
          method: 'DELETE',
          auth: true,
        }),
    },
  },

  /** Networks opened through a share link. */
  shared: {
    /** Public: no login needed. 404 if the link is unknown or turned off. */
    get: (token: string) => request<SharedNetwork>(`/api/shared/${encodeURIComponent(token)}/`),
    /** A copy in the current user's account. */
    duplicate: (token: string) =>
      request<Project>(`/api/shared/${encodeURIComponent(token)}/duplicate/`, {
        method: 'POST',
        auth: true,
      }),
  },

  training: {
    /** What the server can do (public, no login needed). */
    status: (signal?: AbortSignal) => request<TrainingStatus>('/api/training/', { signal }),
    datasets: {
      /** The user's own datasets and the ones shared in their projects (`mine`, `owner`). */
      list: (signal?: AbortSignal) =>
        request<Dataset[]>('/api/training/datasets/', { auth: true, signal }),
      /** A CSV file or a ZIP file of images; `upload` can cancel and follow the upload. */
      upload: (file: File, name?: string, upload: UploadOptions = {}) => {
        const form = new FormData();
        form.append('file', file);
        if (name) {
          form.append('name', name);
        }
        return request<Dataset>('/api/training/datasets/', {
          method: 'POST',
          body: form,
          auth: true,
          ...upload,
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
    /**
     * The check with the planned settings: its `estimate` (CPU time) and `budget` (whether
     * the budgets allow the run) are for exactly these settings.
     */
    estimate: (
      graph: GraphDocument,
      dataset: DatasetRef,
      config: Partial<TrainingConfig>,
      tuning: TuningInput | null,
      signal?: AbortSignal
    ) =>
      request<Compatibility>('/api/training/check/', {
        method: 'POST',
        body: { graph, dataset, config, ...(tuning ? { tuning } : {}) },
        auth: true,
        signal,
      }),
    runs: {
      /** The user's runs and the runs of the projects they can open, newest first. */
      list: (signal?: AbortSignal) =>
        request<RunSummary[]>('/api/training/runs/', { auth: true, signal }),
      get: (id: string, signal?: AbortSignal) =>
        request<RunDetail>(`/api/training/runs/${id}/`, { auth: true, signal }),
      create: (data: RunInput) =>
        request<RunDetail>('/api/training/runs/', { method: 'POST', body: data, auth: true }),
      /** Who started it and the owner and editors of its project; others get HTTP 403. */
      stop: (id: string) =>
        request<RunDetail>(`/api/training/runs/${id}/stop/`, { method: 'POST', auth: true }),
      /** Who started it and the owner of its project; others get HTTP 403. */
      remove: (id: string) =>
        request<void>(`/api/training/runs/${id}/`, { method: 'DELETE', auth: true }),
      downloadWeights: (id: string) =>
        requestFile(`/api/training/runs/${id}/weights/`, { auth: true }),
      /** What the prediction panel needs (everyone who can see the run). */
      predictForm: (id: string, signal?: AbortSignal) =>
        request<PredictionForm>(`/api/training/runs/${id}/predict-form/`, { auth: true, signal }),
      /** Predict up to `max_rows` samples; HTTP 400 answers with PredictionErrorBody. */
      predict: (id: string, body: PredictionRequest, signal?: AbortSignal) =>
        request<PredictionResult>(`/api/training/runs/${id}/predict/`, {
          method: 'POST',
          body,
          auth: true,
          signal,
        }),
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
      /** Import a data file; options left out are detected. `upload` can cancel and follow it. */
      upload: (
        file: File,
        options: Partial<ImportOptions> & { name?: string } = {},
        upload: UploadOptions = {}
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
          ...upload,
        });
      },
      /** Let the server download and import the file at `url` (http/https). */
      importUrl: (
        url: string,
        options: Partial<ImportOptions> & { name?: string } = {},
        signal?: AbortSignal
      ) =>
        request<DataSource>('/api/data/sources/', {
          method: 'POST',
          body: { url, ...options },
          auth: true,
          signal,
        }),
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
      /** Own preparations and the ones shared in the user's projects (`role`, `owner`). */
      list: (signal?: AbortSignal) =>
        request<DataPrep[]>('/api/data/preps/', { auth: true, signal }),
      get: (id: number, signal?: AbortSignal) =>
        request<DataPrep>(`/api/data/preps/${id}/`, { auth: true, signal }),
      create: (data: DataPrepInput) =>
        request<DataPrep>('/api/data/preps/', { method: 'POST', body: data, auth: true }),
      /** Owner and editors (who cannot change the source). Viewers get HTTP 403. */
      update: (id: number, data: Partial<DataPrepInput>) =>
        request<DataPrep>(`/api/data/preps/${id}/`, { method: 'PATCH', body: data, auth: true }),
      /** Owner only (others get HTTP 404). */
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
