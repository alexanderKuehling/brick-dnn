import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import {
  api,
  ApiError,
  getTokens,
  isTwoFactorChallenge,
  onTokensChange,
  setTokens,
  type User,
} from '@/api';

/** After the password: logged in, or the code of the authenticator app is needed. */
export type LoginStep = { done: true } | { done: false; challenge: string };

interface AuthContextValue {
  user: User | null;
  /** False until the stored session (if any) has been checked. */
  ready: boolean;
  login: (username: string, password: string) => Promise<LoginStep>;
  /** `acceptTerms`: the user agreed to the server's terms of use (if it has some). */
  register: (
    username: string,
    email: string,
    password: string,
    acceptTerms?: boolean
  ) => Promise<void>;
  /** `endSession: false` only forgets the login here (e.g. after deleting the account). */
  logout: (endSession?: boolean) => void;
}

/** What the provider holds besides: only the login page needs it (useLoginWithCode). */
interface AuthState extends AuthContextValue {
  loginWithCode: (challenge: string, code: string) => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!getTokens()) {
      setReady(true);
      return;
    }
    api
      .me()
      .then(setUser)
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) {
          setTokens(null);
        }
      })
      .finally(() => setReady(true));
  }, []);

  // If a token refresh fails somewhere, the session is over.
  useEffect(
    () =>
      onTokensChange(() => {
        if (!getTokens()) {
          setUser(null);
        }
      }),
    []
  );

  const login = useCallback(async (username: string, password: string): Promise<LoginStep> => {
    const result = await api.login(username, password);
    if (isTwoFactorChallenge(result)) {
      return { done: false, challenge: result.challenge };
    }
    setTokens({ access: result.access, refresh: result.refresh });
    setUser(await api.me());
    return { done: true };
  }, []);

  const loginWithCode = useCallback(async (challenge: string, code: string) => {
    const { access, refresh } = await api.loginWithCode(challenge, code);
    setTokens({ access, refresh });
    setUser(await api.me());
  }, []);

  const register = useCallback(
    async (username: string, email: string, password: string, acceptTerms = false) => {
      const {
        user: created,
        access,
        refresh,
      } = acceptTerms
        ? await api.register(username, email, password, true)
        : await api.register(username, email, password);
      setTokens({ access, refresh });
      setUser(created);
    },
    []
  );

  const logout = useCallback((endSession = true) => {
    const current = getTokens();
    setTokens(null);
    setUser(null);
    if (current && endSession) {
      api.logout(current.refresh).catch(() => undefined);
    }
  }, []);

  const value = useMemo(
    () => ({ user, ready, login, loginWithCode, register, logout }),
    [user, ready, login, loginWithCode, register, logout]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

function useAuthState(): AuthState {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside <AuthProvider>');
  }
  return context;
}

export function useAuth(): AuthContextValue {
  return useAuthState();
}

/** The second step of a login with two-factor authentication: the challenge from `login`
 * and a code from the authenticator app (or a recovery code). */
export function useLoginWithCode(): AuthState['loginWithCode'] {
  return useAuthState().loginWithCode;
}
