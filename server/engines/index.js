// Engine registry. Selected by ENGINE= and MODEL= in .env.

const loaders = {
  'openai-compatible': () => import('./openai-compatible.js'),
  openai: () => import('./openai-compatible.js'),
  anthropic: () => import('./anthropic.js'),
  bedrock: () => import('./bedrock.js'),
  maira2: () => import('./maira2.js'),
  chexagent: () => import('./chexagent.js'),
  mock: () => import('./mock.js'),
  local: () => import('./local.js'),
};

export const ENGINE_NAMES = Object.keys(loaders).filter((k) => k !== 'openai');

export function engineConfig() {
  const engine = (process.env.ENGINE || 'mock').toLowerCase();
  let model = process.env.MODEL || (engine === 'mock' ? 'mock-cxr-v1' : '');
  if (!model && engine === 'local') {
    // findings model + report writer, so bench ratings stay per pipeline
    const reader = (process.env.LOCAL_READER || 'densenet').toLowerCase();
    const readerName = reader === 'densenet' ? 'densenet121-res224-all' : reader === 'internvl3' ? 'internvl3-1b' : reader === 'smolvlm' ? 'smolvlm-500m' : reader;
    model = `${readerName}+${(process.env.LOCAL_LLM || 'template').toLowerCase()}`;
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
