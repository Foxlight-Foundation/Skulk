import 'styled-components';
import type { Theme } from './theme';

declare module 'styled-components' {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type -- Styled-components requires interface augmentation to inherit the typed theme.
  export interface DefaultTheme extends Theme {}
}
