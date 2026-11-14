// Before everything else, Sentry included: functions that older browsers lack (polyfills.ts).
import './polyfills';

// Errors while the app loads are reported too (docs/monitoring.md).
import ReactDOM from 'react-dom/client';
import App from './App';
import { rootErrorHandlers } from './monitoring';

ReactDOM.createRoot(document.getElementById('root')!, rootErrorHandlers).render(<App />);
