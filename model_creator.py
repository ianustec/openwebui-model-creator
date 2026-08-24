"""
title: Model Creator
author: IANUSTEC
author_url: https://ianustec.com
funding_url: https://github.com/ianustec/openwebui-model-creator
description: Create and save Open WebUI Workspace Models (agent presets) from Markdown/JSON — native function calling, auto-bind knowledge/tools/skills/actions/capabilities, edit bindings on existing models, private by default
requirements: httpx, pydantic, PyYAML
required_open_webui_version: 0.10.0
version: 1.1.0
license: MIT
"""

# ============================================================================
# Model Creator — open-source Open WebUI tool
# ----------------------------------------------------------------------------
# Guides the model to design a Workspace Model (agent preset on top of a
# base LLM) and PERSISTS it via POST /api/v1/models/create (private by default).
# Defaults: function_calling=native; auto-activates knowledge/tools/skills/
# actions/capabilities/default features/builtin tools from context; if vague,
# lists options visible to the user and asks what to enable.
#
# Do NOT only dump a model spec in chat — always call create_model to save it.
#
# License: MIT — Copyright (c) 2026 IANUSTEC.
# ============================================================================

from __future__ import annotations

import json
import re
import traceback
import unicodedata
from typing import Any, Optional
from urllib.parse import quote

from pydantic import BaseModel, Field

try:
    import httpx  # type: ignore

    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# ============================================================================
# Parsing & validation
# ============================================================================

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Keys allowed in MODEL.md frontmatter (extras ignored with a warning in validate)
_ALLOWED_FM_KEYS = {
    "id",
    "name",
    "title",
    "display_name",
    "base_model_id",
    "base_model",
    "description",
    "tags",
    "system",
    "system_prompt",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "function_calling",
    "skillIds",
    "skill_ids",
    "skills",
    "toolIds",
    "tool_ids",
    "tools",
    "actionIds",
    "action_ids",
    "actions",
    "filterIds",
    "filter_ids",
    "filters",
    "defaultFeatureIds",
    "default_feature_ids",
    "default_features",
    "builtinTools",
    "builtin_tools",
    "capabilities",
    "knowledge",
    "is_active",
    "params",
    "meta",
}

# Inference params that map into ModelForm.params
_PARAM_KEYS = {
    "system",
    "system_prompt",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "repeat_last_n",
    "repeat_penalty",
    "tfs_z",
    "num_ctx",
    "num_batch",
    "num_keep",
    "num_predict",
    "function_calling",
}

_DEFAULT_CAPABILITIES = {
    "file_context": True,
    "vision": True,
    "file_upload": True,
    "web_search": True,
    "image_generation": True,
    "code_interpreter": True,
    "terminal": True,
    "citations": True,
    "status_updates": True,
    "builtin_tools": True,
}

_BUILTIN_TOOL_KEYS = (
    "time",
    "memory",
    "chats",
    "notes",
    "knowledge",
    "channels",
    "web_search",
    "image_generation",
    "code_interpreter",
    "tasks",
    "automations",
    "calendar",
)

_DEFAULT_FEATURE_KEYS = ("web_search", "image_generation", "code_interpreter")

# Keyword → resource hints for auto-bind
_INFER_RULES: list[tuple[tuple[str, ...], dict]] = [
    (
        ("knowledge", "rag", "document", "documenti", "kb", "base conoscenza",
         "manuale", "policy", "handbook"),
        {"builtin": ["knowledge"], "caps": ["file_context"], "prefer": "knowledge"},
    ),
    (
        ("web", "search", "ricerca", "internet", "news", "notizie"),
        {"builtin": ["web_search"], "features": ["web_search"], "caps": ["web_search"]},
    ),
    (
        ("image", "immagine", "draw", "genera img", "dall-e", "vision", "foto"),
        {"builtin": ["image_generation"], "features": ["image_generation"],
         "caps": ["vision", "image_generation"]},
    ),
    (
        ("code", "python", "interpreter", "calcolo", "execute", "codice"),
        {"builtin": ["code_interpreter"], "features": ["code_interpreter"],
         "caps": ["code_interpreter"]},
    ),
    (
        ("memory", "memoria", "ricorda", "remember"),
        {"builtin": ["memory"]},
    ),
    (
        ("note", "notes", "appunti"),
        {"builtin": ["notes"]},
    ),
    (
        ("calendar", "calendario", "meeting", "riunione", "appuntamento"),
        {"builtin": ["calendar"]},
    ),
    (
        ("task", "todo", "piano", "planner", "checklist"),
        {"builtin": ["tasks"]},
    ),
    (
        ("automat", "schedule", "cron"),
        {"builtin": ["automations"]},
    ),
    (
        ("chat history", "storico chat", "conversazioni passate"),
        {"builtin": ["chats"]},
    ),
    (
        ("channel", "canale", "slack"),
        {"builtin": ["channels"]},
    ),
    (
        ("skill", "playbook", "guideline", "linee guida"),
        {"prefer": "skills"},
    ),
    (
        ("tool", "strumento", "mcp"),
        {"prefer": "tools"},
    ),
    (
        ("action", "azione", "button"),
        {"prefer": "actions"},
    ),
]


def _slugify_id(text: str, *, max_len: int = 256) -> str:
    """Workspace model id: lowercase, digits, hyphens, underscores, dots, colons."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.strip()
    # Keep common model-id chars (openai:gpt-4o, ollama/llama3.2, etc.)
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"[^a-zA-Z0-9._:/-]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return (t or "model")[:max_len]


def _display_name_from_id(model_id: str) -> str:
    base = (model_id or "model").split("/")[-1].split(":")[-1]
    base = base.replace("_", "-").replace(".", "-")
    return " ".join(part.capitalize() for part in base.split("-") if part)


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:markdown|md|yaml|json|model)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _yaml_load(text: str) -> dict:
    if not text or not text.strip():
        return {}
    if not _HAS_YAML:
        out: dict = {}
        for line in text.splitlines():
            if ":" not in line or line.strip().startswith("#"):
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip().strip("\"'")
            if k:
                out[k] = v
        return out
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict) and item.get("id"):
                out.append(str(item["id"]).strip())
            elif isinstance(item, dict) and item.get("name"):
                out.append(str(item["name"]).strip())
        return out
    return []


def _normalize_tags(tags: Any) -> list[dict]:
    """OWUI expects meta.tags as list[{name: str}]."""
    names = _as_str_list(tags)
    # also accept [{"name": "x"}]
    if isinstance(tags, list):
        for item in tags:
            if isinstance(item, dict) and item.get("name"):
                n = str(item["name"]).strip()
                if n and n not in names:
                    names.append(n)
    return [{"name": n} for n in names]


def _normalize_knowledge(raw: Any) -> list[dict]:
    """Normalize knowledge to OWUI meta.knowledge list of {id,name,type}."""
    if not raw:
        return []
    if isinstance(raw, str):
        ids = _as_str_list(raw)
        return [{"id": i, "name": i, "type": "collection"} for i in ids]
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append({"id": item.strip(), "name": item.strip(), "type": "collection"})
        elif isinstance(item, dict) and item.get("id"):
            out.append({
                "id": str(item["id"]),
                "name": str(item.get("name") or item["id"]),
                "type": str(item.get("type") or "collection"),
            })
    return out


def _normalize_builtin_tools(raw: Any) -> dict[str, bool]:
    """builtinTools: missing key = enabled; False = disabled (OWUI UI convention)."""
    if raw is None:
        return {}
    if isinstance(raw, list):
        # list of enabled names → enable those, disable others? Keep only explicit True.
        enabled = {str(x) for x in raw}
        return {k: (k in enabled) for k in _BUILTIN_TOOL_KEYS if k in enabled}
    if isinstance(raw, dict):
        return {str(k): bool(v) for k, v in raw.items()}
    return {}


def _extract_bindings(raw: dict, meta_extra: dict) -> dict:
    """Pull binding fields from a raw dict + nested meta."""
    return {
        "skillIds": _as_str_list(
            raw.get("skillIds") or raw.get("skill_ids") or raw.get("skills")
            or meta_extra.get("skillIds")
        ),
        "toolIds": _as_str_list(
            raw.get("toolIds") or raw.get("tool_ids") or raw.get("tools")
            or meta_extra.get("toolIds")
        ),
        "actionIds": _as_str_list(
            raw.get("actionIds") or raw.get("action_ids") or raw.get("actions")
            or meta_extra.get("actionIds")
        ),
        "filterIds": _as_str_list(
            raw.get("filterIds") or raw.get("filter_ids") or raw.get("filters")
            or meta_extra.get("filterIds")
        ),
        "defaultFeatureIds": _as_str_list(
            raw.get("defaultFeatureIds")
            or raw.get("default_feature_ids")
            or raw.get("default_features")
            or meta_extra.get("defaultFeatureIds")
        ),
        "builtinTools": _normalize_builtin_tools(
            raw.get("builtinTools")
            or raw.get("builtin_tools")
            or meta_extra.get("builtinTools")
        ),
        "capabilities": (
            raw.get("capabilities")
            or meta_extra.get("capabilities")
            or {}
        ),
        "knowledge": _normalize_knowledge(
            raw.get("knowledge") or meta_extra.get("knowledge")
        ),
    }


def _has_explicit_bindings(spec: dict) -> bool:
    return bool(
        spec.get("skillIds")
        or spec.get("toolIds")
        or spec.get("actionIds")
        or spec.get("filterIds")
        or spec.get("knowledge")
        or spec.get("defaultFeatureIds")
        or spec.get("builtinTools")
    )


def _context_blob(spec: dict, extra: str = "") -> str:
    parts = [
        spec.get("id") or "",
        spec.get("name") or "",
        spec.get("description") or "",
        spec.get("system") or "",
        extra or "",
    ]
    return " ".join(parts).lower()


def _is_vague_request(spec: dict, context: str = "") -> bool:
    """True when the user gave almost no domain signal (e.g. 'test agent')."""
    blob = _context_blob(spec, context)
    if _has_explicit_bindings(spec):
        return False
    # Short / generic
    meaningful = re.sub(
        r"\b(test|demo|prova|agent|agente|assistant|assistente|model|modello|"
        r"bot|simple|semplice|basic|default|generic|generico)\b",
        " ",
        blob,
    )
    meaningful = re.sub(r"\s+", " ", meaningful).strip()
    if len(meaningful) < 24:
        return True
    # No infer-rule keyword hit
    for keywords, _ in _INFER_RULES:
        if any(k in blob for k in keywords):
            return False
    return len(blob) < 80


def _infer_from_context(spec: dict, catalog: dict, context: str = "") -> dict:
    """Mutate a copy of spec with inferred capabilities / builtin / bindings."""
    out = dict(spec)
    blob = _context_blob(spec, context)
    caps = dict(_DEFAULT_CAPABILITIES)
    if isinstance(out.get("capabilities"), dict) and out["capabilities"]:
        caps.update(out["capabilities"])
    builtin: dict[str, bool] = dict(out.get("builtinTools") or {})
    features = list(out.get("defaultFeatureIds") or [])
    prefer_skills = prefer_tools = prefer_actions = prefer_knowledge = False

    for keywords, hints in _INFER_RULES:
        if not any(k in blob for k in keywords):
            continue
        for b in hints.get("builtin") or []:
            builtin[b] = True
        for f in hints.get("features") or []:
            if f not in features:
                features.append(f)
        for c in hints.get("caps") or []:
            caps[c] = True
        pref = hints.get("prefer")
        if pref == "skills":
            prefer_skills = True
        elif pref == "tools":
            prefer_tools = True
        elif pref == "actions":
            prefer_actions = True
        elif pref == "knowledge":
            prefer_knowledge = True

    # Always keep core agentic builtins on unless explicitly disabled
    for core in ("time", "tasks"):
        if core not in builtin:
            builtin[core] = True

    # Match catalog items by name/description keywords in blob
    def _match_items(items: list[dict], limit: int = 5) -> list[str]:
        scored: list[tuple[int, str]] = []
        for it in items:
            iid = str(it.get("id") or "")
            label = f"{it.get('name') or ''} {it.get('description') or ''}".lower()
            if not iid:
                continue
            score = 0
            for token in re.findall(r"[a-z0-9]{4,}", label):
                if token in blob:
                    score += 1
            if score:
                scored.append((score, iid))
        scored.sort(reverse=True)
        return [iid for _, iid in scored[:limit]]

    skill_ids = list(out.get("skillIds") or [])
    tool_ids = list(out.get("toolIds") or [])
    action_ids = list(out.get("actionIds") or [])
    knowledge = list(out.get("knowledge") or [])

    if prefer_skills or "skill" in blob:
        for sid in _match_items(catalog.get("skills") or []):
            if sid not in skill_ids:
                skill_ids.append(sid)
    if prefer_tools or "tool" in blob or "strument" in blob:
        for tid in _match_items(catalog.get("tools") or []):
            if tid not in tool_ids:
                tool_ids.append(tid)
    if prefer_actions or "action" in blob:
        for aid in _match_items(catalog.get("actions") or []):
            if aid not in action_ids:
                action_ids.append(aid)
    if prefer_knowledge or "knowledge" in blob or "document" in blob:
        for kid in _match_items(catalog.get("knowledge") or []):
            if not any(k.get("id") == kid for k in knowledge):
                # find name
                name = kid
                for it in catalog.get("knowledge") or []:
                    if it.get("id") == kid:
                        name = it.get("name") or kid
                        break
                knowledge.append({"id": kid, "name": name, "type": "collection"})
            builtin["knowledge"] = True
            caps["file_context"] = True

    out["capabilities"] = caps
    out["builtinTools"] = builtin
    out["defaultFeatureIds"] = features
    out["skillIds"] = skill_ids
    out["toolIds"] = tool_ids
    out["actionIds"] = action_ids
    out["knowledge"] = knowledge

    params = dict(out.get("params") or {})
    # Native function calling by default
    if not params.get("function_calling"):
        params["function_calling"] = "native"
    out["params"] = params
    return out


def _parse_model_content(content: Any, *, default_base: str = "") -> dict:
    """Accept MODEL.md, JSON string, or dict → normalized spec.

    Spec keys: id, name, base_model_id, description, system, tags, skillIds,
    toolIds, actionIds, filterIds, defaultFeatureIds, builtinTools,
    capabilities, knowledge, params (dict), meta (dict extras).
    """
    if isinstance(content, dict):
        raw = dict(content)
        model_id = _slugify_id(str(raw.get("id") or raw.get("name") or "model"))
        display = (
            raw.get("title")
            or raw.get("display_name")
            or raw.get("name")
            or _display_name_from_id(model_id)
        )
        base = (
            raw.get("base_model_id")
            or raw.get("base_model")
            or default_base
            or ""
        )
        params = dict(raw.get("params") or {})
        system = (
            raw.get("system")
            or raw.get("system_prompt")
            or params.get("system")
            or raw.get("content")
            or raw.get("body")
            or ""
        )
        if system and "system" not in params:
            params["system"] = str(system).strip()

        for k in ("temperature", "top_p", "max_tokens", "seed", "function_calling"):
            if k in raw and k not in params:
                params[k] = raw[k]
        if "function_calling" not in params:
            params["function_calling"] = "native"

        meta_extra = dict(raw.get("meta") or {}) if isinstance(raw.get("meta"), dict) else {}
        bindings = _extract_bindings(raw, meta_extra)
        tags = _normalize_tags(raw.get("tags") or meta_extra.get("tags"))
        caps = bindings["capabilities"]

        return {
            "id": model_id,
            "name": str(display).strip() or _display_name_from_id(model_id),
            "base_model_id": str(base).strip() if base else None,
            "description": str(
                raw.get("description") or meta_extra.get("description") or ""
            ).strip(),
            "system": str(params.get("system") or "").strip(),
            "tags": tags,
            "skillIds": bindings["skillIds"],
            "toolIds": bindings["toolIds"],
            "actionIds": bindings["actionIds"],
            "filterIds": bindings["filterIds"],
            "defaultFeatureIds": bindings["defaultFeatureIds"],
            "builtinTools": bindings["builtinTools"],
            "capabilities": caps if isinstance(caps, dict) else {},
            "knowledge": bindings["knowledge"],
            "params": params,
            "is_active": bool(raw.get("is_active", True)),
        }

    if not isinstance(content, str):
        raise ValueError(f"unsupported content type: {type(content).__name__}")

    text = _strip_code_fence(content)
    if not text:
        raise ValueError("empty content")

    head = text.lstrip()
    if head.startswith("{") or head.startswith("["):
        try:
            return _parse_model_content(json.loads(text), default_base=default_base)
        except json.JSONDecodeError:
            pass

    fm: dict = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        fm = _yaml_load(m.group(1))
        body = text[m.end() :].lstrip("\n")
    elif text.startswith("---"):
        raise ValueError("invalid YAML frontmatter (unclosed ---)")

    name_raw = str(fm.get("id") or fm.get("name") or "").strip()
    if not name_raw:
        h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        name_raw = h1.group(1).strip() if h1 else "untitled-model"

    model_id = _slugify_id(str(fm.get("id") or name_raw))
    display = (
        str(fm.get("title") or fm.get("display_name") or fm.get("name") or "").strip()
        or _display_name_from_id(model_id)
    )
    base = (
        fm.get("base_model_id")
        or fm.get("base_model")
        or default_base
        or ""
    )

    params = dict(fm.get("params") or {}) if isinstance(fm.get("params"), dict) else {}
    system = (
        fm.get("system")
        or fm.get("system_prompt")
        or params.get("system")
        or ""
    )
    body_stripped = body.strip()
    if body_stripped and not system:
        system = body_stripped
    elif body_stripped and system:
        system = str(system).rstrip() + "\n\n" + body_stripped

    if system:
        params["system"] = str(system).strip()

    for k in _PARAM_KEYS:
        if k in fm and k not in ("system", "system_prompt") and k not in params:
            params[k] = fm[k]
    if "function_calling" not in params:
        params["function_calling"] = "native"

    meta_extra = dict(fm.get("meta") or {}) if isinstance(fm.get("meta"), dict) else {}
    bindings = _extract_bindings(fm, meta_extra)

    return {
        "id": model_id,
        "name": display,
        "base_model_id": str(base).strip() if base else None,
        "description": str(
            fm.get("description") or meta_extra.get("description") or ""
        ).strip(),
        "system": str(params.get("system") or "").strip(),
        "tags": _normalize_tags(fm.get("tags") or meta_extra.get("tags")),
        "skillIds": bindings["skillIds"],
        "toolIds": bindings["toolIds"],
        "actionIds": bindings["actionIds"],
        "filterIds": bindings["filterIds"],
        "defaultFeatureIds": bindings["defaultFeatureIds"],
        "builtinTools": bindings["builtinTools"],
        "capabilities": (
            bindings["capabilities"]
            if isinstance(bindings["capabilities"], dict)
            else {}
        ),
        "knowledge": bindings["knowledge"],
        "params": params,
        "is_active": bool(fm.get("is_active", True)),
        "_frontmatter_keys": list(fm.keys()),
    }


def _validate_model_spec(spec: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []

    fm_keys = spec.get("_frontmatter_keys")
    if fm_keys:
        unexpected = set(fm_keys) - _ALLOWED_FM_KEYS
        if unexpected:
            # Soft: warn only in errors as advisory — do not fail create
            pass

    model_id = (spec.get("id") or "").strip()
    if not model_id:
        errors.append("Missing model id")
    elif len(model_id) > 256:
        errors.append(f"Model id too long ({len(model_id)}). Max 256.")

    if not (spec.get("name") or "").strip():
        errors.append("Missing display name")

    base = (spec.get("base_model_id") or "").strip()
    if not base:
        errors.append(
            "Missing base_model_id — required (the upstream LLM this "
            "workspace model wraps, e.g. gpt-4o, llama3.2, claude-…)"
        )

    description = (spec.get("description") or "").strip()
    if not description:
        errors.append(
            "Missing description — shown in the model selector; explain "
            "what this agent is for"
        )
    elif len(description) > 2048:
        errors.append(f"Description too long ({len(description)}). Max 2048.")

    system = (spec.get("system") or "").strip()
    if not system:
        errors.append(
            "Missing system prompt — define the agent persona / instructions "
            "(frontmatter `system:` or markdown body)"
        )

    caps = spec.get("capabilities")
    if caps is not None and not isinstance(caps, dict):
        errors.append("capabilities must be an object/dict")

    return (len(errors) == 0, errors)


def _to_model_form(spec: dict, *, is_active: bool, private: bool) -> dict:
    """Build OpenWebUI ModelForm JSON payload."""
    params = dict(spec.get("params") or {})
    system = (spec.get("system") or params.get("system") or "").strip()
    if system:
        params["system"] = system
    # Native function calling by default (OWUI ModelEditor Advanced)
    if not params.get("function_calling"):
        params["function_calling"] = "native"
    params = {k: v for k, v in params.items() if v is not None and v != ""}

    meta: dict[str, Any] = {
        "description": (spec.get("description") or "").strip(),
    }
    tags = spec.get("tags") or []
    if tags:
        meta["tags"] = tags

    skill_ids = list(spec.get("skillIds") or [])
    if skill_ids:
        meta["skillIds"] = skill_ids
    tool_ids = list(spec.get("toolIds") or [])
    if tool_ids:
        meta["toolIds"] = tool_ids
    action_ids = list(spec.get("actionIds") or [])
    if action_ids:
        meta["actionIds"] = action_ids
    filter_ids = list(spec.get("filterIds") or [])
    if filter_ids:
        meta["filterIds"] = filter_ids
    feature_ids = list(spec.get("defaultFeatureIds") or [])
    if feature_ids:
        meta["defaultFeatureIds"] = feature_ids

    caps = spec.get("capabilities")
    if isinstance(caps, dict) and caps:
        meta["capabilities"] = caps
    else:
        meta["capabilities"] = dict(_DEFAULT_CAPABILITIES)

    builtin = spec.get("builtinTools")
    if isinstance(builtin, dict) and builtin:
        meta["builtinTools"] = builtin

    knowledge = spec.get("knowledge")
    if knowledge:
        meta["knowledge"] = knowledge

    payload: dict[str, Any] = {
        "id": spec["id"],
        "base_model_id": spec.get("base_model_id"),
        "name": spec["name"],
        "meta": meta,
        "params": params,
        "is_active": bool(
            is_active if is_active is not None else spec.get("is_active", True)
        ),
    }
    # OWUI ModelForm types access_grants as list — JSON null → 422.
    # Empty list = private (owner-only); never send null.
    payload["access_grants"] = [] if private else (payload.get("access_grants") or [])
    return payload


# ============================================================================
# Auth + Models HTTP client
# ============================================================================


def _extract_bearer(request) -> Optional[str]:
    if request is None:
        return None

    tok = getattr(getattr(request, "state", None), "token", None)
    creds = getattr(tok, "credentials", None) if tok else None
    if creds:
        return str(creds)

    try:
        headers = getattr(request, "headers", None)
        if headers is not None:
            auth = headers.get("authorization") or headers.get("Authorization")
            if auth and isinstance(auth, str):
                if auth.lower().startswith("bearer "):
                    return auth[7:].strip()
                return auth.strip()
            cookie = headers.get("cookie") or headers.get("Cookie") or ""
            m = re.search(r"(?:^|;\s*)token=([^;]+)", cookie)
            if m:
                return m.group(1).strip()
    except Exception:
        traceback.print_exc()

    try:
        cookies = getattr(request, "cookies", None)
        if cookies and cookies.get("token"):
            return str(cookies.get("token"))
    except Exception:
        pass

    return None


def _api_base(request, override: str = "") -> str:
    o = (override or "").strip().rstrip("/")
    if o:
        return o
    if request is not None:
        try:
            return str(request.base_url).rstrip("/")
        except Exception:
            pass
    return ""


async def _owui_api(
    *,
    request,
    method: str,
    path: str,
    json_body: Optional[dict] = None,
    api_base_override: str = "",
) -> tuple[int, Any]:
    """Authenticated call to OpenWebUI `/api/v1{path}` (path starts with /)."""
    if not _HAS_HTTPX:
        raise RuntimeError(
            "httpx is required. OpenWebUI should auto-install it from "
            "the tool requirements."
        )

    token = _extract_bearer(request)
    if not token:
        raise RuntimeError(
            "No auth token on __request__. This tool must run inside "
            "OpenWebUI with an authenticated user session."
        )

    base = _api_base(request, api_base_override)
    if not base:
        raise RuntimeError(
            "Cannot resolve OpenWebUI API base URL. Set valve "
            "owui_api_base or ensure __request__ is present."
        )

    url = f"{base}/api/v1{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.request(
            method.upper(),
            url,
            headers=headers,
            json=json_body if json_body is not None else None,
        )

    try:
        data = resp.json()
    except Exception:
        data = resp.text
    return resp.status_code, data


async def _models_api(
    *,
    request,
    method: str,
    path: str,
    json_body: Optional[dict] = None,
    api_base_override: str = "",
) -> tuple[int, Any]:
    """Call /api/v1/models/... with the user's bearer token."""
    return await _owui_api(
        request=request,
        method=method,
        path=f"/models{path}",
        json_body=json_body,
        api_base_override=api_base_override,
    )


def _items_from_list_payload(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "data", "knowledge", "tools", "skills", "functions"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
    return []


def _summarize_resource(item: dict) -> dict:
    mid = item.get("id") or item.get("name") or ""
    name = item.get("name") or item.get("id") or ""
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    desc = (
        item.get("description")
        or meta.get("description")
        or ""
    )
    return {
        "id": str(mid),
        "name": str(name),
        "description": str(desc)[:240],
        "type": str(item.get("type") or meta.get("type") or ""),
    }


async def _fetch_user_catalog(
    *,
    request,
    api_base_override: str = "",
) -> dict:
    """List resources visible to the authenticated user (Workspace selectors)."""
    catalog: dict[str, list[dict]] = {
        "knowledge": [],
        "tools": [],
        "skills": [],
        "actions": [],
        "filters": [],
    }

    async def _get(path: str) -> list[dict]:
        try:
            status, data = await _owui_api(
                request=request,
                method="GET",
                path=path,
                api_base_override=api_base_override,
            )
            if status == 200:
                return [_summarize_resource(x) for x in _items_from_list_payload(data)]
        except Exception:
            traceback.print_exc()
        return []

    catalog["knowledge"] = await _get("/knowledge/")
    if not catalog["knowledge"]:
        catalog["knowledge"] = await _get("/knowledge")

    tools = await _get("/tools/")
    if not tools:
        tools = await _get("/tools/list")
    catalog["tools"] = tools

    skills = await _get("/skills/")
    if not skills:
        skills = await _get("/skills")
    catalog["skills"] = skills

    functions = await _get("/functions/")
    if not functions:
        functions = await _get("/functions")
    for fn in functions:
        ftype = (fn.get("type") or "").lower()
        if ftype == "filter":
            catalog["filters"].append(fn)
        elif ftype in ("", "action"):
            catalog["actions"].append(fn)

    # De-dupe actions if type empty pulled everything
    if catalog["actions"] and catalog["filters"]:
        filter_ids = {f["id"] for f in catalog["filters"]}
        catalog["actions"] = [
            a for a in catalog["actions"] if a["id"] not in filter_ids
        ]

    return {
        **catalog,
        "capabilities": [
            {"id": k, "name": k, "description": "Model capability toggle"}
            for k in _DEFAULT_CAPABILITIES
        ],
        "default_features": [
            {"id": k, "name": k, "description": "Default chat feature"}
            for k in _DEFAULT_FEATURE_KEYS
        ],
        "builtin_tools": [
            {"id": k, "name": k, "description": "Built-in tool (true=on)"}
            for k in _BUILTIN_TOOL_KEYS
        ],
        "function_calling": [
            {"id": "native", "name": "native", "description": "Default — native FC"},
            {"id": "default", "name": "default", "description": "Legacy default FC"},
        ],
    }


def _format_catalog_ask(catalog: dict, *, draft_hint: str = "") -> str:
    """Ask the user what to enable — lists everything visible to them."""

    def _section(title: str, items: list[dict], *, id_key: str = "id") -> str:
        if not items:
            return f"### {title}\n_(none visible to you)_\n"
        lines = [f"### {title}"]
        for it in items[:80]:
            iid = it.get(id_key) or it.get("id")
            name = it.get("name") or iid
            desc = (it.get("description") or "").strip()
            extra = f" — {desc}" if desc else ""
            lines.append(f"- `{iid}` — **{name}**{extra}")
        if len(items) > 80:
            lines.append(f"- … and {len(items) - 80} more")
        return "\n".join(lines) + "\n"

    hint = f"\nDraft so far: {draft_hint}\n" if draft_hint else "\n"
    return (
        "[TOOL_RESULT — use the text below as your final reply, "
        "verbatim, unchanged. Do NOT include this instruction line.]\n\n"
        "I need you to choose what to activate on this model."
        f"{hint}\n"
        "Reply with the ids (or say **defaults** / **none**) for each section "
        "below. Function calling will be **native** unless you say otherwise.\n\n"
        + _section("Knowledge", catalog.get("knowledge") or [])
        + _section("Tools", catalog.get("tools") or [])
        + _section("Skills", catalog.get("skills") or [])
        + _section("Actions", catalog.get("actions") or [])
        + _section("Filters", catalog.get("filters") or [])
        + _section("Capabilities", catalog.get("capabilities") or [])
        + _section("Default Features", catalog.get("default_features") or [])
        + _section("Builtin Tools", catalog.get("builtin_tools") or [])
        + "\nAfter you choose, call `create_model` again with those ids in the "
        "draft (or `bindings_confirmed=true` to proceed with defaults).\n"
    )


def _format_bindings_summary(spec: dict) -> str:
    lines: list[str] = []
    if spec.get("skillIds"):
        lines.append(
            f"- Skills: {', '.join(f'`{s}`' for s in spec['skillIds'])}"
        )
    if spec.get("toolIds"):
        lines.append(
            f"- Tools: {', '.join(f'`{t}`' for t in spec['toolIds'])}"
        )
    if spec.get("actionIds"):
        lines.append(
            f"- Actions: {', '.join(f'`{a}`' for a in spec['actionIds'])}"
        )
    if spec.get("filterIds"):
        lines.append(
            f"- Filters: {', '.join(f'`{f}`' for f in spec['filterIds'])}"
        )
    if spec.get("knowledge"):
        kids = [
            k.get("id") if isinstance(k, dict) else k
            for k in spec["knowledge"]
        ]
        lines.append(f"- Knowledge: {', '.join(f'`{k}`' for k in kids)}")
    if spec.get("defaultFeatureIds"):
        lines.append(
            "- Default features: "
            + ", ".join(f'`{f}`' for f in spec["defaultFeatureIds"])
        )
    builtin = spec.get("builtinTools") or {}
    if isinstance(builtin, dict) and builtin:
        on = [k for k, v in builtin.items() if v]
        if on:
            lines.append(f"- Builtin tools on: {', '.join(f'`{b}`' for b in on)}")
    fc = (spec.get("params") or {}).get("function_calling") or "native"
    lines.append(f"- Function calling: `{fc}`")
    return ("\n".join(lines) + "\n") if lines else ""


def _format_api_error(status: int, data: Any) -> str:
    detail = ""
    if isinstance(data, dict):
        detail = str(data.get("detail") or data.get("message") or data)
    else:
        detail = str(data or "")

    if status == 401:
        low = detail.lower()
        if "taken" in low or "id" in low:
            return (
                f"Model id already taken (401). Use update_model, or call "
                f"create_model with update_if_exists=true, or pick a new id. "
                f"Detail: {detail}"
            )
        return (
            "Unauthorized (401). You need Workspace → Models Access "
            f"(permission workspace.models). Detail: {detail}"
        )
    if status == 404:
        return f"Model not found (404). Detail: {detail}"
    return f"OpenWebUI Models API error ({status}): {detail}"


# ============================================================================
# Tool replies
# ============================================================================


def _error_reply(msg: str) -> str:
    return (
        "[TOOL_RESULT — use the text below as your final reply, "
        "verbatim, unchanged. Do NOT include this instruction line.]\n\n"
        f"Could not create/update the model: {msg}"
    )


def _success_reply(
    *,
    action: str,
    model_id: str,
    name: str,
    base_model_id: str,
    description: str,
    private: bool,
    active: bool,
    spec: dict,
) -> str:
    visibility = "private" if private else "shared/public grants applied"
    active_s = "active" if active else "inactive"
    workspace = "/workspace/models"
    bindings = _format_bindings_summary(spec)
    return (
        "[TOOL_RESULT]\n\n"
        "OUTPUT_FOR_USER — Your **next assistant message** must be **only** "
        "the text between the dashed lines (`---`) below. Copy it exactly "
        "(including blank lines). Do not add summaries or extra sentences.\n\n"
        "---\n"
        f"Model {action}: **{name}** (`{model_id}`)\n\n"
        f"- Base model: `{base_model_id}`\n"
        f"- Visibility: {visibility}\n"
        f"- Status: {active_s}\n"
        f"- Description: {description}\n"
        f"{bindings}\n"
        f"Open it in [Workspace → Models]({workspace}).\n\n"
        "How to use it:\n"
        "- Select it in the model dropdown for a new chat\n"
        "- Edit system prompt / skills / tools anytime in Workspace → Models\n"
        "---\n"
    )


# ============================================================================
# Tools class
# ============================================================================


class Tools:
    class Valves(BaseModel):
        default_private: bool = Field(
            default=True,
            description=(
                "If true, create models with no access_grants (owner-only / "
                "private)."
            ),
        )
        default_active: bool = Field(
            default=True,
            description="Set is_active=true on create/update.",
        )
        default_base_model_id: str = Field(
            default="",
            description=(
                "Fallback base_model_id when the draft omits it "
                "(e.g. gpt-4o, llama3.2)."
            ),
        )
        allow_update_if_exists: bool = Field(
            default=True,
            description=(
                "When create_model(..., update_if_exists=true) and the id "
                "already exists, fall back to update."
            ),
        )
        auto_bind_from_context: bool = Field(
            default=True,
            description=(
                "Infer knowledge/tools/skills/actions/capabilities/builtin "
                "tools/default features from name+description+system+context."
            ),
        )
        ask_when_vague: bool = Field(
            default=True,
            description=(
                "If the draft has little context and no explicit bindings, "
                "list everything visible to the user and ask what to enable "
                "instead of creating immediately."
            ),
        )
        emit_status: bool = Field(
            default=True,
            description="Emit progress status events in chat.",
        )
        owui_api_base: str = Field(
            default="",
            description=(
                "Optional override for OpenWebUI base URL. "
                "Empty = use request.base_url."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = False
        self.tools = [
            self._descriptor_create(),
            self._descriptor_update(),
            self._descriptor_validate(),
            self._descriptor_list_options(),
            self._descriptor_list_models(),
            self._descriptor_edit_model(),
        ]

    async def _emit_status(self, emitter, description: str, *, done: bool) -> None:
        if not emitter or not self.valves.emit_status:
            return
        try:
            await emitter({
                "type": "status",
                "data": {"description": description, "done": done},
            })
        except Exception:
            pass

    async def _emit_message(self, emitter, text: str) -> None:
        if not emitter:
            return
        try:
            await emitter({"type": "message", "data": {"content": text}})
        except Exception:
            pass


    # ── Public: list_model_options ───────────────────────────────────────────
    async def list_model_options(
        self,
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """List everything the user can bind on a Workspace Model.

        Returns knowledge, tools, skills, actions, filters visible to them,
        plus capability / default-feature / builtin-tool catalogs.
        Call this (or let create_model ask) when the user has not said what
        to enable.
        """
        await self._emit_status(
            __event_emitter__, "Loading Workspace options...", done=False
        )
        try:
            catalog = await _fetch_user_catalog(
                request=__request__,
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))
        await self._emit_status(__event_emitter__, "Options loaded.", done=True)
        return _format_catalog_ask(catalog)

    # ── Public: validate_model ───────────────────────────────────────────────
    async def validate_model(
        self,
        content: str,
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """Validate a Workspace Model draft (MODEL.md / JSON) without saving.

        Args:
            content: MODEL.md with YAML frontmatter, or JSON ModelForm-like
                object string.

        Returns:
            Validation result for the user.
        """
        try:
            spec = _parse_model_content(
                content, default_base=self.valves.default_base_model_id
            )
        except Exception as exc:
            return _error_reply(f"Parse error: {exc}")

        ok, errors = _validate_model_spec(spec)
        if not ok:
            return (
                "[TOOL_RESULT — use the text below as your final reply, "
                "verbatim.]\n\n"
                "Model draft is **invalid**:\n"
                + "\n".join(f"- {e}" for e in errors)
            )
        bindings = _format_bindings_summary(spec)
        return (
            "[TOOL_RESULT — use the text below as your final reply, "
            "verbatim.]\n\n"
            f"Model draft is **valid**.\n\n"
            f"- id: `{spec['id']}`\n"
            f"- name: {spec['name']}\n"
            f"- base_model_id: `{spec.get('base_model_id')}`\n"
            f"- description: {spec['description'][:200]}"
            f"{'…' if len(spec['description']) > 200 else ''}\n"
            f"- system prompt chars: {len(spec.get('system') or '')}\n"
            f"{bindings}\n"
            "Call `create_model` with the same content to save it "
            "(private by default; function_calling=native)."
        )

    # ── Public: create_model ─────────────────────────────────────────────────
    async def create_model(
        self,
        content: str,
        update_if_exists: bool = False,
        context: str = "",
        bindings_confirmed: bool = False,
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """Create (and persist) an Open WebUI Workspace Model from MODEL.md/JSON.

        ALWAYS call this after drafting a model preset — do NOT only paste
        the spec in chat. The model is saved private by default (owner only).

        Defaults:
        - params.function_calling = **native**
        - Auto-activates Knowledge / Tools / Skills / Actions / Capabilities /
          Default Features / Builtin Tools from context when possible.
        - If context is vague and bindings are empty, lists everything visible
          to the user and asks what to enable (unless bindings_confirmed=true).

        WORKFLOW:
        1. Interview: purpose, base LLM, system prompt; gather domain signals
           (RAG, web, code, images, skills, tools…).
        2. Draft MODEL.md / JSON. Required: id/name, base_model_id,
           description, system.
        3. Call THIS tool. Pass `context` with the user interview text.
        4. If the tool asks for choices, show the catalog and re-call with
           explicit ids or bindings_confirmed=true.

        Example::

            ---
            id: python-tutor
            name: Python Tutor
            base_model_id: gpt-4o
            description: Patient Python tutor for beginners. Use for learning
              Python, debugging student code, or explaining concepts.
            temperature: 0.4
            function_calling: native
            system: |
              You are a patient Python tutor for {{ USER_NAME }}.
              Prefer short examples and check understanding.
            ---

        Args:
            content: MODEL.md with YAML frontmatter (preferred) OR JSON
                with id/name/base_model_id/description/system (or params.system).
            update_if_exists: If true and id exists, update instead of failing.
            context: Extra free-text from the interview (used for auto-bind).
            bindings_confirmed: Skip the "ask user" gate and create with
                inferred/default bindings.

        Returns:
            Confirmation with model id and bindings, or a catalog ask.
        """
        await self._emit_status(
            __event_emitter__, "Parsing model draft...", done=False
        )
        try:
            spec = _parse_model_content(
                content, default_base=self.valves.default_base_model_id
            )
        except Exception as exc:
            return _error_reply(f"Parse error: {exc}")

        ok, errors = _validate_model_spec(spec)
        if not ok:
            return _error_reply(
                "Validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )

        # Gate on the draft *before* inference (infer always sets some builtins)
        had_explicit = _has_explicit_bindings(spec)
        vague = _is_vague_request(spec, context or "")

        catalog: dict = {}
        need_catalog = (
            self.valves.auto_bind_from_context or self.valves.ask_when_vague
        )
        if need_catalog and __request__ is not None:
            await self._emit_status(
                __event_emitter__,
                "Loading knowledge/tools/skills/actions...",
                done=False,
            )
            try:
                catalog = await _fetch_user_catalog(
                    request=__request__,
                    api_base_override=self.valves.owui_api_base,
                )
            except Exception:
                traceback.print_exc()
                catalog = {}

        if (
            self.valves.ask_when_vague
            and not bindings_confirmed
            and vague
            and not had_explicit
        ):
            if not catalog and __request__ is not None:
                try:
                    catalog = await _fetch_user_catalog(
                        request=__request__,
                        api_base_override=self.valves.owui_api_base,
                    )
                except Exception as exc:
                    return _error_reply(str(exc))
            await self._emit_status(
                __event_emitter__, "Waiting for binding choices...", done=True
            )
            hint = (
                f"`{spec.get('id')}` — {spec.get('name')} / "
                f"{(spec.get('description') or '')[:120]}"
            )
            return _format_catalog_ask(catalog or {}, draft_hint=hint)

        if self.valves.auto_bind_from_context:
            spec = _infer_from_context(spec, catalog or {}, context or "")

        private = bool(self.valves.default_private)
        active = bool(self.valves.default_active)
        form = _to_model_form(spec, is_active=active, private=private)

        await self._emit_status(
            __event_emitter__,
            f"Saving model `{spec['id']}` to Workspace...",
            done=False,
        )

        try:
            status, data = await _models_api(
                request=__request__,
                method="POST",
                path="/create",
                json_body=form,
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))

        taken = status == 401 and (
            "taken" in str(data).lower() or "TAKEN" in str(data)
        )
        if taken and update_if_exists and self.valves.allow_update_if_exists:
            await self._emit_status(
                __event_emitter__,
                f"Id taken — updating existing model `{spec['id']}`...",
                done=False,
            )
            return await self._do_update(
                form=form,
                spec=spec,
                private=private,
                active=active,
                __event_emitter__=__event_emitter__,
                __request__=__request__,
            )

        if status not in (200, 201):
            await self._emit_status(__event_emitter__, "Save failed.", done=True)
            return _error_reply(_format_api_error(status, data))

        await self._emit_status(__event_emitter__, "Model saved.", done=True)
        reply = _success_reply(
            action="created",
            model_id=spec["id"],
            name=spec["name"],
            base_model_id=spec.get("base_model_id") or "",
            description=spec["description"],
            private=private,
            active=active,
            spec=spec,
        )
        await self._emit_message(
            __event_emitter__,
            f"\n\nModel created: **{spec['name']}** (`{spec['id']}`) — "
            f"[Workspace → Models](/workspace/models)\n",
        )
        return reply

    # ── Public: list_models ──────────────────────────────────────────────────
    async def list_models(
        self,
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """List Workspace Models visible to the user, with current bindings.

        Use this to ask the user which model should get a skill/tool/knowledge
        bound by default (e.g. after creating a skill with Skill Creator).
        """
        await self._emit_status(
            __event_emitter__, "Loading your models...", done=False
        )
        try:
            status, data = await _models_api(
                request=__request__,
                method="GET",
                path="/list",
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))
        if status != 200:
            return _error_reply(_format_api_error(status, data))

        items = _items_from_list_payload(data)
        await self._emit_status(__event_emitter__, "Models loaded.", done=True)
        if not items:
            return (
                "[TOOL_RESULT — use the text below as your final reply, "
                "verbatim.]\n\n"
                "No workspace models visible to you. Create one first with "
                "`create_model`."
            )

        lines = []
        for m in items:
            meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}
            skills = meta.get("skillIds") or []
            tools = meta.get("toolIds") or []
            know = meta.get("knowledge") or []
            know_ids = [
                k.get("id") if isinstance(k, dict) else k for k in know
            ]
            active = "active" if m.get("is_active", True) else "inactive"
            writable = m.get("write_access", True)
            lock = "" if writable else " _(read-only for you)_"
            lines.append(
                f"- `{m.get('id')}` — **{m.get('name')}** "
                f"(base: `{m.get('base_model_id') or '—'}`, {active}){lock}\n"
                f"  - skills: {skills or '[]'} · tools: {tools or '[]'}"
                + (f" · knowledge: {know_ids}" if know_ids else "")
            )
        return (
            "[TOOL_RESULT — show this list to the user and ask WHICH model "
            "should get the new binding by default. Then call edit_model "
            "with the chosen id.]\n\n"
            "Your workspace models:\n\n" + "\n".join(lines)
        )

    # ── Public: edit_model ───────────────────────────────────────────────────
    async def edit_model(
        self,
        model_id: str,
        add_skill_ids: Optional[list[str]] = None,
        remove_skill_ids: Optional[list[str]] = None,
        add_tool_ids: Optional[list[str]] = None,
        remove_tool_ids: Optional[list[str]] = None,
        add_action_ids: Optional[list[str]] = None,
        remove_action_ids: Optional[list[str]] = None,
        add_knowledge_ids: Optional[list[str]] = None,
        remove_knowledge_ids: Optional[list[str]] = None,
        reason: str = "",
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """Edit bindings on an EXISTING Workspace Model, preserving its config.

        Fetches the current model, merges the requested add/remove lists into
        meta (skillIds / toolIds / actionIds / knowledge), and saves. Use this
        to enable a freshly created skill by default on a model the user picks.

        Typical flow: list_models → ask the user which model → edit_model(
        model_id=..., add_skill_ids=["new-skill-id"]).

        Args:
            model_id: Existing model id (must be writable by the user).
            add_skill_ids / remove_skill_ids: skill ids to bind/unbind.
            add_tool_ids / remove_tool_ids: tool ids to bind/unbind.
            add_action_ids / remove_action_ids: action ids to bind/unbind.
            add_knowledge_ids / remove_knowledge_ids: knowledge collection ids.
            reason: Short human note of why (echoed in the confirmation).

        Returns:
            Confirmation with the before/after bindings.
        """
        mid = (model_id or "").strip()
        if not mid:
            return _error_reply("model_id is required")

        def _merge(existing: list, add: Optional[list], rem: Optional[list]) -> list:
            out = [str(x) for x in existing if x]
            for r in (rem or []):
                out = [x for x in out if x != str(r)]
            for a in (add or []):
                a = str(a).strip()
                if a and a not in out:
                    out.append(a)
            return out

        await self._emit_status(
            __event_emitter__, f"Loading model `{mid}`...", done=False
        )
        try:
            status, data = await _models_api(
                request=__request__,
                method="GET",
                path=f"/model?id={quote(mid, safe='')}",
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))
        if status != 200 or not isinstance(data, dict):
            return _error_reply(_format_api_error(status, data))

        if data.get("write_access") is False:
            return _error_reply(
                f"You have read-only access to model `{mid}`. "
                "Ask the owner to bind the skill, or pick another model."
            )

        meta = dict(data.get("meta") or {})
        before = {
            "skillIds": list(meta.get("skillIds") or []),
            "toolIds": list(meta.get("toolIds") or []),
            "actionIds": list(meta.get("actionIds") or []),
            "knowledge": list(meta.get("knowledge") or []),
        }

        meta["skillIds"] = _merge(before["skillIds"], add_skill_ids, remove_skill_ids)
        meta["toolIds"] = _merge(before["toolIds"], add_tool_ids, remove_tool_ids)
        meta["actionIds"] = _merge(before["actionIds"], add_action_ids, remove_action_ids)

        existing_kids = [
            k.get("id") if isinstance(k, dict) else str(k)
            for k in before["knowledge"]
        ]
        merged_kids = _merge(existing_kids, add_knowledge_ids, remove_knowledge_ids)
        name_by_id = {
            (k.get("id") if isinstance(k, dict) else str(k)): (
                k.get("name") if isinstance(k, dict) else str(k)
            )
            for k in before["knowledge"]
        }
        meta["knowledge"] = [
            {
                "id": kid,
                "name": name_by_id.get(kid) or kid,
                "type": "collection",
            }
            for kid in merged_kids
        ]

        form = {
            "id": data.get("id") or mid,
            "base_model_id": data.get("base_model_id"),
            "name": data.get("name") or mid,
            "meta": meta,
            "params": data.get("params") or {},
            "access_grants": data.get("access_grants") or [],
            "is_active": bool(data.get("is_active", True)),
        }

        changes: list[str] = []
        for key, label in (("skillIds", "Skills"), ("toolIds", "Tools"),
                           ("actionIds", "Actions")):
            if before[key] != meta[key]:
                changes.append(
                    f"- {label}: {before[key] or '[]'} → **{meta[key] or '[]'}**"
                )
        if [k.get("id") for k in before["knowledge"] if isinstance(k, dict)] != merged_kids and before["knowledge"] != meta["knowledge"]:
            changes.append(
                f"- Knowledge: {existing_kids or '[]'} → **{merged_kids or '[]'}**"
            )
        if not changes:
            return (
                "[TOOL_RESULT — use the text below as your final reply, "
                "verbatim.]\n\n"
                f"Nothing to change on **{form['name']}** (`{mid}`) — the "
                "requested bindings are already in place."
            )

        await self._emit_status(
            __event_emitter__, f"Saving model `{mid}`...", done=False
        )
        try:
            status, data = await _models_api(
                request=__request__,
                method="POST",
                path="/model/update",
                json_body=form,
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))
        if status not in (200, 201):
            await self._emit_status(__event_emitter__, "Update failed.", done=True)
            return _error_reply(_format_api_error(status, data))

        await self._emit_status(__event_emitter__, "Model updated.", done=True)
        reason_line = f"\n_{reason}_\n" if reason else ""
        return (
            "[TOOL_RESULT]\n\n"
            "OUTPUT_FOR_USER — Your **next assistant message** must be **only** "
            "the text between the dashed lines (`---`) below. Copy it exactly. "
            "Do not add summaries or extra sentences.\n\n"
            "---\n"
            f"Model updated: **{form['name']}** (`{mid}`)\n"
            f"{reason_line}\n"
            + "\n".join(changes)
            + "\n\nThe new bindings are active by default in every new chat "
            "with this model.\n"
            "---\n"
        )

    # ── Public: update_model ─────────────────────────────────────────────────
    async def update_model(
        self,
        model_id: str,
        content: str,
        context: str = "",
        __event_emitter__=None,
        __request__=None,
        __user__=None,
    ) -> str:
        """Update an existing Open WebUI Workspace Model you own (or can write).

        Args:
            model_id: Existing model id.
            content: New MODEL.md or JSON (same formats as create_model).
                The path model_id is authoritative for the resource id.
            context: Optional interview text for auto-bind inference.

        Returns:
            Confirmation that the model was updated.
        """
        mid = _slugify_id(model_id or "")
        if not mid:
            return _error_reply("model_id is required")

        await self._emit_status(
            __event_emitter__, f"Updating model `{mid}`...", done=False
        )
        try:
            spec = _parse_model_content(
                content, default_base=self.valves.default_base_model_id
            )
        except Exception as exc:
            return _error_reply(f"Parse error: {exc}")

        spec["id"] = mid
        ok, errors = _validate_model_spec(spec)
        if not ok:
            return _error_reply(
                "Validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )

        if self.valves.auto_bind_from_context and __request__ is not None:
            try:
                catalog = await _fetch_user_catalog(
                    request=__request__,
                    api_base_override=self.valves.owui_api_base,
                )
                spec = _infer_from_context(spec, catalog, context or "")
            except Exception:
                traceback.print_exc()
                spec = _infer_from_context(spec, {}, context or "")
        elif self.valves.auto_bind_from_context:
            spec = _infer_from_context(spec, {}, context or "")

        private = bool(self.valves.default_private)
        active = bool(self.valves.default_active)
        form = _to_model_form(spec, is_active=active, private=private)
        form["id"] = mid

        return await self._do_update(
            form=form,
            spec=spec,
            private=private,
            active=active,
            __event_emitter__=__event_emitter__,
            __request__=__request__,
        )

    async def _do_update(
        self,
        *,
        form: dict,
        spec: dict,
        private: bool,
        active: bool,
        __event_emitter__,
        __request__,
    ) -> str:
        try:
            # OWUI v0.10: POST /api/v1/models/model/update
            status, data = await _models_api(
                request=__request__,
                method="POST",
                path="/model/update",
                json_body=form,
                api_base_override=self.valves.owui_api_base,
            )
        except Exception as exc:
            traceback.print_exc()
            return _error_reply(str(exc))

        if status not in (200, 201):
            await self._emit_status(__event_emitter__, "Update failed.", done=True)
            return _error_reply(_format_api_error(status, data))

        await self._emit_status(__event_emitter__, "Model updated.", done=True)
        reply = _success_reply(
            action="updated",
            model_id=spec["id"],
            name=spec["name"],
            base_model_id=spec.get("base_model_id") or "",
            description=spec["description"],
            private=private,
            active=active,
            spec=spec,
        )
        await self._emit_message(
            __event_emitter__,
            f"\n\nModel updated: **{spec['name']}** (`{spec['id']}`)\n",
        )
        return reply

    # ── Tool descriptors ─────────────────────────────────────────────────────
    @staticmethod
    def _descriptor_create() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "create_model",
                "description": (
                    "Create and SAVE an Open WebUI Workspace Model (agent "
                    "preset). Private by default. function_calling=native. "
                    "Auto-binds knowledge/tools/skills/actions/capabilities/"
                    "default features/builtin tools from context; if vague, "
                    "lists options visible to the user and asks what to "
                    "enable. ALWAYS call after drafting — never only paste "
                    "the spec in chat.\n\n"
                    "Required: id/name, base_model_id, description, system. "
                    "Optional: temperature, skillIds, toolIds, actionIds, "
                    "filterIds, knowledge, capabilities, defaultFeatureIds, "
                    "builtinTools, function_calling."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "MODEL.md with YAML frontmatter + optional "
                                "body, OR JSON with id/name/base_model_id/"
                                "description/system (or params.system)."
                            ),
                        },
                        "update_if_exists": {
                            "type": "boolean",
                            "description": (
                                "If true and the model id already exists, "
                                "update it instead of failing."
                            ),
                            "default": False,
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Interview / user intent text used to "
                                "auto-activate bindings."
                            ),
                            "default": "",
                        },
                        "bindings_confirmed": {
                            "type": "boolean",
                            "description": (
                                "Skip the ask-user catalog gate and create "
                                "with inferred/default bindings."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["content"],
                },
            },
        }

    @staticmethod
    def _descriptor_update() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "update_model",
                "description": (
                    "Update an existing Open WebUI Workspace Model by id "
                    "with a new MODEL.md / JSON body. Re-infers bindings "
                    "from context when enabled."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Existing model id.",
                        },
                        "content": {
                            "type": "string",
                            "description": "New MODEL.md or JSON content.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional interview text for auto-bind.",
                            "default": "",
                        },
                    },
                    "required": ["model_id", "content"],
                },
            },
        }

    @staticmethod
    def _descriptor_validate() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "validate_model",
                "description": (
                    "Validate a Workspace Model draft without saving. "
                    "Call before create_model if unsure."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "MODEL.md or JSON draft to validate.",
                        },
                    },
                    "required": ["content"],
                },
            },
        }

    @staticmethod
    def _descriptor_list_models() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_models",
                "description": (
                    "List the Workspace Models visible to the current user "
                    "with their current bindings (skills, tools, knowledge) "
                    "and write access. Call this after creating a skill/tool "
                    "to ask the user WHICH model should get it enabled by "
                    "default, then call edit_model with the chosen id."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    @staticmethod
    def _descriptor_edit_model() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "edit_model",
                "description": (
                    "Edit bindings on an EXISTING Workspace Model while "
                    "preserving its config: add/remove skillIds, toolIds, "
                    "actionIds and knowledge collections. Use this to enable "
                    "a freshly created skill by default on a model the user "
                    "chooses (flow: list_models → user picks → edit_model "
                    "with add_skill_ids=[skill_id]). Fetches the current "
                    "model first, so nothing else is lost."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Existing model id (writable by the user).",
                        },
                        "add_skill_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skill ids to bind by default.",
                        },
                        "remove_skill_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skill ids to unbind.",
                        },
                        "add_tool_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tool ids to bind.",
                        },
                        "remove_tool_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tool ids to unbind.",
                        },
                        "add_action_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Action ids to bind.",
                        },
                        "remove_action_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Action ids to unbind.",
                        },
                        "add_knowledge_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Knowledge collection ids to attach.",
                        },
                        "remove_knowledge_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Knowledge collection ids to detach.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Short note echoed in the confirmation.",
                            "default": "",
                        },
                    },
                    "required": ["model_id"],
                },
            },
        }

    @staticmethod
    def _descriptor_list_options() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "list_model_options",
                "description": (
                    "List knowledge, tools, skills, actions, filters visible "
                    "to the current user, plus capabilities / default "
                    "features / builtin tools catalogs. Use when the user "
                    "must choose what to activate on a new model."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
