// Amazon Bedrock via the Converse API (streamed). Works with any Bedrock
// vision model that supports Converse: Anthropic Claude, Amazon Nova, Llama
// vision, Pixtral and others.
//
//   AWS_REGION (or AWS_DEFAULT_REGION), plus the usual AWS credential chain
//   MODEL   e.g. us.anthropic.claude-sonnet-4-5-20250929-v1:0 or amazon.nova-pro-v1:0

import { BedrockRuntimeClient, ConverseStreamCommand, ConverseCommand } from '@aws-sdk/client-bedrock-runtime';
import { makeEngine } from './base.js';

let client;
function getClient() {
  if (!client) {
    client = new BedrockRuntimeClient({ region: process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || 'us-east-1' });
  }
  return client;
}

function imageFormat(mime) {
  if (/png/.test(mime)) return 'png';
  if (/webp/.test(mime)) return 'webp';
  if (/gif/.test(mime)) return 'gif';
  return 'jpeg';
}

async function* generate({ system, text, image, model }) {
  const content = [{ text }];
  if (image) {
    content.unshift({ image: { format: imageFormat(image.mime), source: { bytes: Buffer.from(image.base64, 'base64') } } });
  }
  const input = {
    modelId: model,
    system: [{ text: system }],
    messages: [{ role: 'user', content }],
    inferenceConfig: { maxTokens: Number(process.env.MAX_TOKENS || 2048), temperature: 0.2 },
  };
  try {
    if (process.env.BEDROCK_NO_STREAM === '1') {
      const out = await getClient().send(new ConverseCommand(input));
      const txt = out?.output?.message?.content?.map((c) => c.text || '').join('') ?? '';
      yield { type: 'delta', text: txt };
      return;
    }
    const out = await getClient().send(new ConverseStreamCommand(input));
    for await (const ev of out.stream) {
      const t = ev?.contentBlockDelta?.delta?.text;
      if (t) yield { type: 'delta', text: t };
      if (ev?.internalServerException || ev?.modelStreamErrorException || ev?.throttlingException || ev?.validationException) {
        const e = ev.internalServerException || ev.modelStreamErrorException || ev.throttlingException || ev.validationException;
        yield { type: 'error', message: `bedrock: ${e.message || 'stream error'}` };
        return;
      }
    }
  } catch (err) {
    yield { type: 'error', message: `bedrock: ${err?.message || err}` };
  }
}

export default makeEngine('bedrock', generate);
