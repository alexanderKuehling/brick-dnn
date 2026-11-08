import { Navigate, useLocation } from 'react-router-dom';
import { Center, Loader } from '@mantine/core';
import { useAuth } from '@/auth/AuthContext';

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, ready } = useAuth();
  const location = useLocation();

  if (!ready) {
    return (
      <Center h={300}>
        <Loader />
      </Center>
    );
  }
  if (!user) {
    // With the query string: the OAuth sign-in page (/oauth/authorize) needs all of it.
    const from = `${location.pathname}${location.search}${location.hash}`;
    return <Navigate to="/login" replace state={{ from }} />;
  }
  return <>{children}</>;
}
