# Contributing to OpenRead

Thanks for helping. OpenRead is a research prototype, so the bar is "works, tested, honest about limits".

1. Fork, then `npm install` and `cp .env.example .env`. The default `ENGINE=mock` needs no key.
2. `npm run dev` for the app, `npm test` for unit tests, `npm run smoke` for the browser smoke test (needs Chromium; set `CHROME=` to your binary).
3. Keep pull requests small and plain JavaScript. No TypeScript build step, no new framework.
4. Adding an engine: one file in `server/engines/` that exports `makeEngine(name, generate)` where `generate` yields `{ type: 'delta', text }`. Register it in `server/engines/index.js` and add a line to `.env.example` and the README.
5. Changing the prompt or the JSON contract: update `server/lib/prompt.js`, the README section, and the tests in `test/`.
6. Sample images must be public domain or CC0 with a line in `CREDITS.md`. No patient data, ever.
7. By contributing you agree your work is released under the MIT license.

Not for clinical use. Do not add features that imply otherwise.
