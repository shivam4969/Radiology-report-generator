// Engine registry. Selected by ENGINE= and MODEL= in .env.

const loaders = {
  'openai-compatible': () => import('./openai-compatible.js'),
  openai: () => import('./openai-compatible.js'),
  anthropic: () => import('./anthropic.js'),
  bedrock: () => import('./bedrock.js'),
  mock: () => import('./mock.js'),
};

export const ENGINE_NAMES = Object.keys(loaders).filter((k) => k !== 'openai');

export function engineConfig() {
  const engine = (process.env.ENGINE || 'mock').toLowerCase();
  const model = process.env.MODEL || (engine === 'mock' ? 'mock-cxr-v1' : '');
  return { engine, model };
}

export async function loadEngine(name = engineConfig().engine) {
  const load = loaders[name];
  if (!load) {
    throw new Error(`Unknown ENGINE "${name}". Use one of: ${ENGINE_NAMES.join(', ')}`);
  }
  return (await load()).default;
}
