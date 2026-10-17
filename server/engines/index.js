// Engine registry. Selected by ENGINE= and MODEL= in .env.

const loaders = {
  'openai-compatible': () => import('./openai-compatible.js'),
  openai: () => import('./openai-compatible.js'),
  anthropic: () => import('./anthropic.js'),
  bedrock: () => import('./bedrock.js'),
  mock: () => import('./mock.js'),
  local: () => import('./local.js'),
};

export const ENGINE_NAMES = Object.keys(loaders).filter((k) => k !== 'openai');

export function engineConfig() {
  const engine = (process.env.ENGINE || 'mock').toLowerCase();
  let model = process.env.MODEL || (engine === 'mock' ? 'mock-cxr-v1' : '');
  if (!model && engine === 'local') {
    // findings model + report writer, so bench ratings stay per pipeline
    model = `densenet121-res224-all+${(process.env.LOCAL_LLM || 'template').toLowerCase()}`;
  }
  return { engine, model };
}

export async function loadEngine(name = engineConfig().engine) {
  const load = loaders[name];
  if (!load) {
    throw new Error(`Unknown ENGINE "${name}". Use one of: ${ENGINE_NAMES.join(', ')}`);
  }
  return (await load()).default;
}
