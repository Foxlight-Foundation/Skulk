import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import { TolgeeProvider } from '@tolgee/react';
import { App } from './App';
import { tolgee } from './i18n/tolgee';
import { store } from './store';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <TolgeeProvider tolgee={tolgee} fallback={null}>
      <Provider store={store}>
        <App />
      </Provider>
    </TolgeeProvider>
  </StrictMode>,
);
