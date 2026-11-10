import { useLocation } from 'react-router-dom';

/**
 * The parameters of a link from an e-mail (password reset, confirming the address). They are
 * in the fragment (`#token=…`), which browsers never send to the server, so tokens stay out of
 * its access logs; the query string works too.
 */
export function useLinkParams(): URLSearchParams {
  const { hash, search } = useLocation();
  const params = new URLSearchParams(search);
  new URLSearchParams(hash.replace(/^#/, '')).forEach((value, key) => params.set(key, value));
  return params;
}
