# Credits

- `samples/cxr-pneumonia-cc0.jpg`: chest radiograph showing lobar pneumonia. Mikael Häggström, released under CC0 1.0 (public domain) via Wikimedia Commons. Used as the demo and benchmark sample.
- Built with Express, React, Vite, better-sqlite3, the AWS SDK for JavaScript and playwright-core. See `package.json` for versions and licenses.
- Local engine: the findings model is [TorchXRayVision](https://github.com/mlmed/torchxrayvision) DenseNet-121 (`densenet121-res224-all`, Cohen et al., Apache 2.0), exported with [ExecuTorch](https://github.com/pytorch/executorch) (BSD). Weights are downloaded at export time and are not part of this repo.
