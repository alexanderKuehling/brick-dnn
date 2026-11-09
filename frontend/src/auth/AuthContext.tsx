import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { api, ApiError, getTokens, onTokensChange, setTokens, type User } from '@/api';

interface AuthContextValue {
  user: User | null;
  /** False until the stored session (if any) has been checked. */
  ready: boolean;
  login: (username: string, password: string) => Promise<void>;
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

const AuthContext = createContext<AuthContextValue | null>(null);

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

  const login = useCallback(async (username: string, password: string) => {
    setTokens(await api.login(username, password));
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
    () => ({ user, ready, login, register, logout }),
    [user, ready, login, register, logout]
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used inside <AuthProvider>');
  }
  return context;
}
