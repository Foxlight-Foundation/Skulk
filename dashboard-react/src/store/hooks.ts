import { useDispatch, useSelector } from 'react-redux';
import type { TypedUseSelectorHook } from 'react-redux';
import type { AppDispatch, RootState } from './index';

/**
 * Typed wrappers around react-redux's hooks. Components import these instead
 * of the raw hooks so Redux state is always type-safe without per-call
 * generics.
 */
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
