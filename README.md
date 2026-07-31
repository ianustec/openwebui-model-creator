# Model Creator — Workspace Models for Open WebUI

An [Open WebUI](https://github.com/open-webui/open-webui) **Tool** that interviews
you, drafts a Workspace **Model** (agent preset on top of a base LLM), and
**persists** it via `POST /api/v1/models/create` — private by default.

Pair with **[Skill Creator](https://github.com/ianustec/openwebui-skill-creator)**
to build Skills, then bind them onto Models.

> License: MIT · Author: [IANUSTEC](https://ianustec.com) · Requires Open WebUI `>= 0.10.0`

## Features

- **Saves for real** — does not only dump a draft in chat; calls the Models API
  with the user’s session token.
- **Native function calling** by default (`params.function_calling = "native"`).
- **Auto-bind from context**: knowledge, tools, skills, actions, filters,
  capabilities, default features, and builtin tools.
- **Ask when vague**: if the draft has little context, lists everything visible
  to the user and asks what to enable (`list_model_options` /
  `bindings_confirmed`).
- **Private by default** — empty `access_grants` (owner-only).
- **MODEL.md or JSON** input with YAML frontmatter.
- **Single-file** — paste into Workspace → Tools.

## Requirements

- Open WebUI `>= 0.10.0` (Workspace Models + Skills APIs)
- Permission: `workspace.models` (or admin)
- Python deps (auto-installed from frontmatter): `httpx`, `pydantic`, `PyYAML`

## Installation

### Option A — Open WebUI community
1. Open the tool page on the Open WebUI community site.
2. Click **Get** / **Import**.

### Option B — manual
1. **Workspace → Tools → +**
2. Paste [`model_creator.py`](model_creator.py)
3. Save and enable the tool on a model/chat

## Usage

The model should:

1. Interview: purpose, base LLM, system prompt, domain signals (RAG, web, code…).
2. Draft `MODEL.md` (or JSON).
3. Call `create_model(content, context=...)`.
4. If the tool asks for bindings, choose ids (or call again with
   `bindings_confirmed=true`).

### Example MODEL.md

See [`examples/python-tutor.md`](examples/python-tutor.md).

```markdown
---
id: python-tutor
name: Python Tutor
base_model_id: gpt-4o
description: Patient Python tutor for beginners. Use for learning Python,
  debugging student code, or explaining concepts.
temperature: 0.4
function_calling: native
system: |
  You are a patient Python tutor for {{ USER_NAME }}.
  Prefer short examples and check understanding.
---
```

### Tools exposed

| Function | Purpose |
|----------|---------|
| `create_model` | Parse, auto-bind / ask, persist |
| `update_model` | Update an existing model by id |
| `validate_model` | Validate without saving |
| `list_model_options` | List knowledge/tools/skills/actions visible to you |

### Valves

| Valve | Default | Meaning |
|-------|---------|---------|
| `default_private` | `true` | Empty `access_grants` |
| `default_active` | `true` | `is_active=true` |
| `default_base_model_id` | `""` | Fallback base LLM id |
| `auto_bind_from_context` | `true` | Infer bindings from text |
| `ask_when_vague` | `true` | Ask before creating vague agents |
| `owui_api_base` | `""` | Override API base (else `request.base_url`) |

## Related

- [Skill Creator](https://github.com/ianustec/openwebui-skill-creator)
- [Generate Documents](https://github.com/ianustec/openwebui-generate-documents)
- [Generate Slides](https://github.com/ianustec/openwebui-generate-slides)
- [Generate Spreadsheets](https://github.com/ianustec/openwebui-generate-spreadsheets)

## License

MIT — see [LICENSE](LICENSE).
