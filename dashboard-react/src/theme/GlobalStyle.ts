import { createGlobalStyle } from 'styled-components';

export const GlobalStyle = createGlobalStyle`
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html, body, #root {
    height: 100%;
    width: 100%;
  }

  body {
    font-family: ${({ theme }) => theme.fonts.body};
    font-size: ${({ theme }) => theme.fontSizes.md};
    background: ${({ theme }) => theme.colors.bgGradient};
    color: ${({ theme }) => theme.colors.text};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    transition: background 0.2s ease, color 0.2s ease;
  }

  /*
   * Dark mode gets the Foxlight-style fixed sky image plus a subtle veil.
   * These sit behind the app shell so the dashboard content keeps its own
   * surface tokens and scrolling behavior.
   */
  html[data-theme='dark'] body::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-image: url('/starry_bg.webp');
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
  }

  html[data-theme='dark'] body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: linear-gradient(
      to bottom,
      rgba(4, 6, 16, 0.74) 0%,
      rgba(4, 6, 16, 0.16) 26%,
      rgba(4, 6, 16, 0.08) 50%,
      rgba(4, 6, 16, 0.24) 74%,
      rgba(4, 6, 16, 0.74) 100%
    );
  }

  a {
    color: inherit;
    text-decoration: none;
  }

  button {
    cursor: pointer;
    border: none;
    background: none;
    font: inherit;
    color: inherit;
  }
`;
