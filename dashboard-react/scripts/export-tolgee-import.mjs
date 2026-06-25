#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const dashboardRoot = path.resolve(scriptDirectory, '..');
const sourceRoot = path.join(dashboardRoot, 'src');
const outputPath = path.join(sourceRoot, 'i18n', 'en', 'skulk.json');

const sourceFiles = [];

function collectSourceFiles(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      collectSourceFiles(entryPath);
      continue;
    }

    if (!/\.(ts|tsx)$/.test(entry.name) || /\.stories\.(ts|tsx)$/.test(entry.name)) {
      continue;
    }

    sourceFiles.push(entryPath);
  }
}

function readLiteralText(node) {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
    return node.text;
  }

  return null;
}

function formatLocation(sourceFile, node) {
  const { line, character } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
  const relativePath = path.relative(dashboardRoot, sourceFile.fileName);
  return `${relativePath}:${line + 1}:${character + 1}`;
}

collectSourceFiles(sourceRoot);

const translations = new Map();
const conflicts = [];
const invalidCalls = [];

for (const filePath of sourceFiles) {
  const contents = fs.readFileSync(filePath, 'utf8');
  const sourceFile = ts.createSourceFile(filePath, contents, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

  function visit(node) {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 't') {
      const [keyNode, fallbackNode] = node.arguments;
      const key = keyNode ? readLiteralText(keyNode) : null;
      const fallback = fallbackNode ? readLiteralText(fallbackNode) : null;
      const location = formatLocation(sourceFile, node);

      if (!key || !fallback) {
        invalidCalls.push(location);
        ts.forEachChild(node, visit);
        return;
      }

      const previous = translations.get(key);
      if (previous && previous.fallback !== fallback) {
        conflicts.push({ key, previous, current: { fallback, location } });
      } else if (!previous) {
        translations.set(key, { fallback, location });
      }
    }

    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
}

if (invalidCalls.length > 0) {
  console.error('Each Tolgee t() call must use literal key and English fallback arguments:');
  for (const location of invalidCalls) {
    console.error(`  - ${location}`);
  }
  process.exitCode = 1;
}

if (conflicts.length > 0) {
  console.error('Conflicting English fallbacks found for Tolgee keys:');
  for (const conflict of conflicts) {
    console.error(`  - ${conflict.key}`);
    console.error(`    ${conflict.previous.location}: ${JSON.stringify(conflict.previous.fallback)}`);
    console.error(`    ${conflict.current.location}: ${JSON.stringify(conflict.current.fallback)}`);
  }
  process.exitCode = 1;
}

if (process.exitCode) {
  process.exit();
}

const sortedTranslations = Object.fromEntries(
  [...translations.entries()]
    .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
    .map(([key, value]) => [key, value.fallback]),
);

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(sortedTranslations, null, 2)}\n`);

console.log(
  `Wrote ${translations.size} English Tolgee keys to ${path.relative(dashboardRoot, outputPath)}.`,
);
