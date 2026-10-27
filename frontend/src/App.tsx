import '@mantine/core/styles.css';
import '@mantine/notifications/styles.css';
import '@xyflow/react/dist/style.css';
import './global.css';

import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { AppShell, MantineProvider } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { Notifications } from '@mantine/notifications';
import { AuthProvider } from './auth/AuthContext';
import { BuilderPage } from './builder/BuilderPage';
import { AppHeader, HEADER_HEIGHT } from './components/AppHeader';
import { RequireAuth } from './components/RequireAuth';
import { LoginPage } from './pages/LoginPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { RegisterPage } from './pages/RegisterPage';
import { theme } from './theme';

export default function App() {
  return (
    <MantineProvider theme={theme} defaultColorScheme="auto">
      <ModalsProvider>
        <Notifications position="top-right" />
        <AuthProvider>
          <BrowserRouter>
            <AppShell header={{ height: HEADER_HEIGHT }} padding={0}>
              <AppShell.Header>
                <AppHeader />
              </AppShell.Header>
              <AppShell.Main>
                <Routes>
                  <Route path="/" element={<BuilderPage />} />
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
                  <Route path="*" element={<NotFoundPage />} />
                </Routes>
              </AppShell.Main>
            </AppShell>
          </BrowserRouter>
        </AuthProvider>
      </ModalsProvider>
    </MantineProvider>
  );
}
