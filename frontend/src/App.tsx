import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import '@xyflow/react/dist/style.css';
import './global.css';

import { BrowserRouter, HashRouter, Route } from 'react-router-dom';
import { AppShell, MantineProvider } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { Notifications } from '@mantine/notifications';
import { AccountPage } from './account/AccountPage';
import { AgentActivityPage } from './activity/AgentActivityPage';
import { AuthProvider } from './auth/AuthContext';
import { PricingPage } from './billing/PricingPage';
import { BuilderPage } from './builder/BuilderPage';
import { AppHeader, HEADER_HEIGHT } from './components/AppHeader';
import { RequireAuth } from './components/RequireAuth';
import { DataPage } from './data/DataPage';
import { DEMO } from './demo';
import { LegalPage } from './legal/LegalPage';
import { MonitoredRoutes } from './monitoring';
import { AuthorizePage } from './oauth/AuthorizePage';
import { ForgotPasswordPage } from './pages/ForgotPasswordPage';
import { LoginPage } from './pages/LoginPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { RegisterPage } from './pages/RegisterPage';
import { ResetPasswordPage } from './pages/ResetPasswordPage';
import { VerifyEmailPage } from './pages/VerifyEmailPage';
import { SharedNetworkPage } from './sharing/SharedNetworkPage';
import { theme } from './theme';
import { ApiTokensPage } from './tokens/ApiTokensPage';

// The online demo can be hosted under any path (e.g. GitHub Pages), so it keeps the route in
// the URL hash. It only has the builder: there are no accounts (the Data tab explains that it
// needs a local installation).
const Router = DEMO ? HashRouter : BrowserRouter;

export default function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <ModalsProvider>
        <Notifications position="top-right" />
        <AuthProvider>
          <Router>
            <AppShell header={{ height: HEADER_HEIGHT }} padding={0}>
              <AppShell.Header>
                <AppHeader />
              </AppShell.Header>
              <AppShell.Main>
                <MonitoredRoutes>
                  <Route path="/" element={<BuilderPage />} />
                  {/* One route, so switching preparations keeps the page (and its state). */}
                  <Route path="/data/:prepId?" element={<DataPage />} />
                  {!DEMO && (
                    <>
                      <Route
                        path="/projects"
                        element={
                          <RequireAuth>
                            <ProjectsPage />
                          </RequireAuth>
                        }
                      />
                      <Route
                        path="/tokens"
                        element={
                          <RequireAuth>
                            <ApiTokensPage />
                          </RequireAuth>
                        }
                      />
                      {/* What AI agents did with the user's API tokens and apps. */}
                      <Route
                        path="/activity"
                        element={
                          <RequireAuth>
                            <AgentActivityPage />
                          </RequireAuth>
                        }
                      />
                      <Route
                        path="/account"
                        element={
                          <RequireAuth>
                            <AccountPage />
                          </RequireAuth>
                        }
                      />
                      {/* Plans and prices, on a server that sells some (docs/billing.md). */}
                      <Route path="/pricing" element={<PricingPage />} />
                      {/* Markdown files the operator puts on the server (docs/privacy.md). */}
                      <Route path="/imprint" element={<LegalPage page="imprint" />} />
                      <Route path="/privacy" element={<LegalPage page="privacy" />} />
                      <Route path="/terms" element={<LegalPage page="terms" />} />
                      <Route path="/refund" element={<LegalPage page="refund" />} />
                      {/* Apps (e.g. Claude) send the browser here to connect to the MCP server. */}
                      <Route
                        path="/oauth/authorize"
                        element={
                          <RequireAuth>
                            <AuthorizePage />
                          </RequireAuth>
                        }
                      />
                      {/* A network shared with a link: viewing needs no account. */}
                      <Route path="/share/:token" element={<SharedNetworkPage />} />
                      <Route path="/login" element={<LoginPage />} />
                      <Route path="/register" element={<RegisterPage />} />
                      {/* Links from e-mails: a forgotten password, confirming the address. */}
                      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
                      <Route path="/reset-password" element={<ResetPasswordPage />} />
                      <Route
                        path="/verify-email"
                        element={
                          <RequireAuth>
                            <VerifyEmailPage />
                          </RequireAuth>
                        }
                      />
                    </>
                  )}
                  <Route path="*" element={<NotFoundPage />} />
                </MonitoredRoutes>
              </AppShell.Main>
            </AppShell>
          </Router>
        </AuthProvider>
      </ModalsProvider>
    </MantineProvider>
  );
}
