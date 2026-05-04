# Icon Assets

This folder contains the local icon assets used by the dynamic infographic renderer.

Current starter set:

- Source pack: Tabler Icons
- Version: `3.39.0`
- Style: outline SVG
- License: MIT
- Manifest: `assets/icons/icon_manifest.json`
- License file: `assets/icons/LICENSES/tabler-LICENSE.txt`

Rendering policy:

- Video rendering must not call the network.
- Gemini should request icons by semantic keys from `icon_manifest.json`, not by filenames.
- Unknown icon keys should log a warning and fall back to a deterministic placeholder.
- Runtime rendering should use local SVGs or pre-rasterized PNG cache files.

Refresh the starter assets with:

```powershell
python tools\fetch_tabler_starter_icons.py
```

The fetcher records source URLs and SHA-256 hashes in the manifest.
