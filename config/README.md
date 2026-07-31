# Configuration convention

- Store project configuration as UTF-8 TOML files named `*.toml`.
- Use `snake_case` for file names, sections, and keys.
- Keep paths relative to the project root so the project remains portable.
- Do not store passwords, tokens, or other secrets here; read those from
  environment variables instead.

Configuration discovery uses this order:

1. `TRAINFOUNDRY_CONFIG_DIR`
2. `./config/paths.toml` in the current working directory
3. the configuration packaged with the CLI

Set `TRAINFOUNDRY_PROJECT_ROOT` to control the base directory used to resolve
relative paths. An installed CLI without overrides defaults to
`~/.local/share/trainfoundry`.

Read configuration through the shared helpers:

```python
from config import get_by_key, get_path, load_config

paths = load_config("paths")
text_path_value = get_by_key("paths.text_path")
text_path = get_path("paths.text_path")
metadata_db_path = get_path("paths.metadata_db_path")
```

Python 3.11+ includes TOML support. On Python 3.10 and earlier, install the
compatible parser with `pip install tomli`.
