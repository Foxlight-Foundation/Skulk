// Copyright 2025 Foxlight Foundation

import { Redirect } from "@docusaurus/router";

/**
 * Redirect the site root to the docs introduction page so the home URL is
 * immediately useful rather than a separate marketing landing page.
 */
export default function Home(): JSX.Element {
  return <Redirect to="/intro" />;
}
