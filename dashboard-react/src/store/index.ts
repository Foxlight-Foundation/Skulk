import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import { apiSlice } from './api';
import { uiSliceReducer, subscribeUIPersistence } from './slices/uiSlice';
import { chatSliceReducer, subscribeChatPersistence } from './slices/chatSlice';

/**
 * Root Redux store for the Skulk dashboard.
 *
 * Two reducer namespaces:
 *  - **Feature slices** (theme, panels, chat, …) — domain state owned by the
 *    dashboard. Added under their own keys as each Zustand store ports across.
 *  - **`apiSlice`** — RTK Query reducer that owns server-cached data
 *    (cluster state, traces, diagnostics, etc.). Endpoints are injected from
 *    feature modules; the slice is the single source of truth for all
 *    network-cached state.
 *
 * `setupListeners` enables RTK Query's `refetchOnFocus` / `refetchOnReconnect`
 * options when individual queries opt in.
 */
export const store = configureStore({
  reducer: {
    ui: uiSliceReducer,
    chat: chatSliceReducer,
    [apiSlice.reducerPath]: apiSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiSlice.middleware),
});

setupListeners(store.dispatch);
subscribeUIPersistence(store);
subscribeChatPersistence(store);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
