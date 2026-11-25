import React, { type ReactNode } from "react";
import Head from "@docusaurus/Head";
import { useLocation } from "@docusaurus/router";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import SiteMetadata from "@theme-original/SiteMetadata";
import { preferredCanonical } from "@site/docs-canonical.cjs";

/** Keep canonical metadata correct in initial HTML and client navigation. */
export default function PreferredSiteMetadata(): ReactNode {
  const { pathname } = useLocation();
  const { siteConfig } = useDocusaurusContext();
  const canonical = preferredCanonical(pathname, siteConfig.url);
  return (
    <>
      <SiteMetadata />
      {canonical && (
        <Head>
          <link rel="canonical" href={canonical} />
          <meta property="og:url" content={canonical} />
        </Head>
      )}
    </>
  );
}
