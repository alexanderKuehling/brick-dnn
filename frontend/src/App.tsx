import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import '@xyflow/react/dist/style.css';
import './global.css';

import { BrowserRouter, HashRouter, Route, Routes } from 'react-router-dom';
import { AppShell, MantineProvider } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { Notifications } from '@mantine/notifications';
import { AuthProvider } from './auth/AuthContext';
import { BuilderPage } from './builder/BuilderPage';
import { AppHeader, HEADER_HEIGHT } from './components/AppHeader';
import { RequireAuth } from './components/RequireAuth';
import { DEMO } from './demo';
import { LoginPage } from './pages/LoginPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { RegisterPage } from './pages/RegisterPage';
import { theme } from './theme';

// The online demo can be hosted under any path (e.g. GitHub Pages), so it keeps the route in
// the URL hash. It only has the builder: there are no accounts.
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
                <Routes>
                  <Route path="/" element={<BuilderPage />} />
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
                      <Route path="/login" element={<LoginPage />} />
                      <Route path="/register" element={<RegisterPage />} />
                    </>
                  )}
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </AppShell.Main>
            </AppShell>
          </Router>
        </AuthProvider>
      </ModalsProvider>
    </MantineProvider>
  );
}
