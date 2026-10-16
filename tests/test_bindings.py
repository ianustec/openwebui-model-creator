"""Binding regressions; run with stdlib unittest, without Open WebUI."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

MODULE = Path(__file__).resolve().parents[1] / "model_creator.py"
loader = importlib.util.spec_from_file_location("model_creator", MODULE)
assert loader and loader.loader
mc = importlib.util.module_from_spec(loader)
loader.loader.exec_module(mc)

CATALOG = {
    "skills": [{"id": "s", "name": "chantier"}],
    "tools": [{"id": "t", "name": "chantier"}],
    "actions": [{"id": "a", "name": "chantier"}],
    "knowledge": [{"id": "k", "name": "chantier"}],
}
ALIASES = {
    "skillIds": ("skillIds", "skill_ids", "skills"),
    "toolIds": ("toolIds", "tool_ids", "tools"),
    "actionIds": ("actionIds", "action_ids", "actions"),
    "filterIds": ("filterIds", "filter_ids", "filters"),
    "knowledge": ("knowledge",),
    "defaultFeatureIds": ("defaultFeatureIds", "default_feature_ids", "default_features"),
    "builtinTools": ("builtinTools", "builtin_tools"),
    "capabilities": ("capabilities",),
}


class ExplicitBindingsTests(unittest.TestCase):
    def parse(self, **bindings):
        return mc._parse_model_content({"id": "test", **bindings})

    def test_absent_bindings_remain_implicit(self):
        spec = self.parse()
        self.assertFalse(mc._has_explicit_bindings(spec))
        self.assertTrue(mc._is_vague_request(spec))
        out = mc._infer_from_context(spec, {})
        self.assertEqual(out["builtinTools"], {"time": True, "tasks": True})
        self.assertEqual(out["capabilities"], mc._DEFAULT_CAPABILITIES)
        self.assertEqual(out["params"]["function_calling"], "native")

    def test_empty_aliases_are_explicit_and_not_vague(self):
        for key, aliases in ALIASES.items():
            empty = {} if key in ("builtinTools", "capabilities") else []
            for alias in aliases:
                with self.subTest(alias=alias):
                    spec = self.parse(**{alias: empty})
                    self.assertEqual(spec[key], empty)
                    self.assertTrue(mc._has_explicit_bindings(spec))
                    self.assertFalse(mc._is_vague_request(spec))

    def test_empty_top_level_wins_over_nested_meta(self):
        for key, aliases in ALIASES.items():
            empty = {} if key in ("builtinTools", "capabilities") else []
            filled = {"vision": True} if key == "capabilities" else (
                {"memory": True} if key == "builtinTools" else ["nested"])
            for alias in aliases:
                with self.subTest(alias=alias):
                    bindings = mc._extract_bindings({alias: empty}, {key: filled})
                    self.assertEqual(bindings[key], empty)

    def test_nested_empty_bindings_are_explicit(self):
        for key in ALIASES:
            empty = {} if key in ("builtinTools", "capabilities") else []
            with self.subTest(key=key):
                spec = self.parse(meta={key: empty})
                self.assertTrue(mc._has_explicit_bindings(spec))
                self.assertFalse(mc._is_vague_request(spec))

    def test_filled_bindings_normalize_as_before(self):
        spec = self.parse(skills=[" s "], tools=" t, other ", actions=[{"id": "a"}],
                          filters=["f"], knowledge=["k"], default_features=["web_search"],
                          builtin_tools={"memory": False})
        self.assertEqual(spec["skillIds"], ["s"])
        self.assertEqual(spec["toolIds"], ["t", "other"])
        self.assertEqual(spec["actionIds"], ["a"])
        self.assertEqual(spec["filterIds"], ["f"])
        self.assertEqual(spec["knowledge"], [{"id": "k", "name": "k", "type": "collection"}])
        self.assertTrue(mc._has_explicit_bindings(spec))

    def test_empty_categories_are_not_inferred(self):
        for key in ("skills", "tools", "actions", "knowledge"):
            canonical = next(k for k, aliases in ALIASES.items() if key in aliases)
            with self.subTest(key=key):
                spec = self.parse(**{key: []})
                out = mc._infer_from_context(spec, CATALOG, "skill tool action knowledge chantier")
                self.assertEqual(out[canonical], [])

    def test_empty_category_does_not_disable_omitted_category(self):
        out = mc._infer_from_context(self.parse(skills=[]), CATALOG, "skill tool chantier")
        self.assertEqual(out["skillIds"], [])
        self.assertEqual(out["toolIds"], ["t"])

    def test_absent_categories_still_infer(self):
        out = mc._infer_from_context(self.parse(), CATALOG, "skill tool action knowledge chantier python")
        for key, value in (("skillIds", ["s"]), ("toolIds", ["t"]), ("actionIds", ["a"])):
            self.assertEqual(out[key], value)
        self.assertEqual(out["knowledge"][0]["id"], "k")
        self.assertIn("code_interpreter", out["defaultFeatureIds"])

    def test_explicit_empty_features_builtins_capabilities_survive_form(self):
        spec = self.parse(default_features=[], builtin_tools={}, capabilities={})
        out = mc._infer_from_context(spec, {}, "python image knowledge web")
        form = mc._to_model_form(out, is_active=True, private=True)
        for key in ("defaultFeatureIds", "builtinTools", "capabilities"):
            empty = [] if key == "defaultFeatureIds" else {}
            self.assertEqual(out[key], empty)
            self.assertEqual(form["meta"][key], empty)
        self.assertNotIn("_explicit_bindings", form["meta"])

    def test_explicit_builtins_do_not_force_core_defaults(self):
        spec = self.parse(builtinTools={"memory": True})
        out = mc._infer_from_context(spec, {})
        self.assertEqual(out["builtinTools"], {"memory": True})

    def test_explicit_core_false_is_preserved(self):
        out = mc._infer_from_context(self.parse(builtinTools={"time": False, "tasks": False}), {})
        self.assertEqual(out["builtinTools"], {"time": False, "tasks": False})

    def test_explicit_empty_lists_are_serialized_in_form(self):
        spec = self.parse(skills=[], tools=[], actions=[], filters=[], knowledge=[])
        form = mc._to_model_form(spec, is_active=True, private=True)
        for key in ("skillIds", "toolIds", "actionIds", "filterIds", "knowledge"):
            with self.subTest(key=key):
                self.assertEqual(form["meta"][key], [])

    def test_complete_yaml_model_and_json_preserve_empty_choices(self):
        raw = {"id": "assistant-facture", "name": "Assistant Facture", "base_model_id": "local",
               "description": "Factures de chantier", "skills": [], "tools": [], "actions": [],
               "knowledge": [], "default_features": [], "builtin_tools": {}, "capabilities": {}}
        import yaml
        md = "---\n" + yaml.safe_dump(raw, allow_unicode=True) + "---\n# Mission\nRédiger une facture de chantier."
        for content in (md, {**raw, "system": "Rédiger une facture de chantier."},
                        json.dumps({**raw, "system": "Rédiger une facture de chantier."})):
            with self.subTest(format=type(content).__name__):
                spec = mc._parse_model_content(content)
                self.assertEqual(spec["id"], "assistant-facture")
                self.assertIn("facture de chantier", spec["system"])
                self.assertFalse(mc._is_vague_request(spec))
                out = mc._infer_from_context(spec, CATALOG, "skill tool action knowledge image python chantier")
                for key in ("skillIds", "toolIds", "actionIds", "knowledge", "defaultFeatureIds"):
                    self.assertEqual(out[key], [])
                self.assertEqual(out["builtinTools"], {})
                self.assertEqual(out["capabilities"], {})


class KeywordInferenceTests(unittest.TestCase):
    def infer(self, context, catalog=None):
        return mc._infer_from_context(mc._parse_model_content({"id": "test"}),
                                      catalog or CATALOG, context)

    def test_substrings_do_not_activate_rules_or_catalog(self):
        for context in ("rédaction chantier", "annoter chantier", "actionnaire chantier",
                        "photosynthèse chantier", "encode chantier", "skillful chantier",
                        "toolbox chantier", "documentaire chantier"):
            with self.subTest(context=context):
                out = self.infer(context)
                self.assertEqual(out["builtinTools"], {"time": True, "tasks": True})
                self.assertEqual(out["defaultFeatureIds"], [])
                for key in ("skillIds", "toolIds", "actionIds", "knowledge"):
                    self.assertEqual(out[key], [])

    def test_postal_code_is_not_programming(self):
        for context in ("code postal du client", "CODE POSTAL : 75001", "codes postaux"):
            with self.subTest(context=context):
                self.assertNotIn("code_interpreter", self.infer(context)["defaultFeatureIds"])
        self.assertIn("code_interpreter", self.infer("code postal et python")["defaultFeatureIds"])
        self.assertIn("code_interpreter", self.infer("code postal et code à écrire")["defaultFeatureIds"])

    def test_french_builtin_keywords(self):
        cases = {
            "knowledge": ("connaissance", "connaissances", "base documentaire", "documents"),
            "web_search": ("recherche web", "internet", "actualités"),
            "image_generation": ("image", "images", "dessin", "photo"),
            "code_interpreter": ("code", "python", "calcul"),
            "memory": ("mémoire", "souviens-toi"),
            "notes": ("note", "notes"),
            "calendar": ("calendrier", "réunion", "rendez-vous"),
            "automations": ("automatisation", "planifier"),
            "chats": ("historique des conversations",),
            "channels": ("canal",),
        }
        for builtin, keywords in cases.items():
            for keyword in keywords:
                with self.subTest(keyword=keyword):
                    self.assertTrue(self.infer(keyword.upper())["builtinTools"].get(builtin))

    def test_french_task_keywords_trigger_rule_not_just_core_default(self):
        # A long context bypasses the generic/short check; the keyword is the only signal.
        for keyword in ("tâche", "tâches", "liste de tâches", "planning"):
            with self.subTest(keyword=keyword):
                spec = {"description": "Un accompagnement professionnel adapté au besoin. " + keyword}
                self.assertFalse(mc._is_vague_request(spec))

    def test_french_preferences_match_catalog(self):
        for keyword, key, expected in (("compétence", "skillIds", "s"),
                                       ("compétences", "skillIds", "s"),
                                       ("procédure", "skillIds", "s"),
                                       ("outil", "toolIds", "t"), ("outils", "toolIds", "t"),
                                       ("action", "actionIds", "a"), ("actions", "actionIds", "a"),
                                       ("bouton", "actionIds", "a")):
            with self.subTest(keyword=keyword):
                self.assertEqual(self.infer(keyword + " chantier")[key], [expected])
        self.assertEqual(self.infer("base documentaire chantier")["knowledge"][0]["id"], "k")

    def test_catalog_matching_is_whole_word_and_accent_insensitive(self):
        catalog = {"skills": [{"id": "s", "name": "RÉUNION"}]}
        self.assertEqual(self.infer("compétence reunion", catalog)["skillIds"], ["s"])
        self.assertEqual(self.infer("skill reunions", catalog)["skillIds"], [])
        self.assertEqual(self.infer("skill superreunion", catalog)["skillIds"], [])

    def test_english_and_italian_keywords_still_work(self):
        for keyword, builtin in (("knowledge", "knowledge"), ("documenti", "knowledge"),
                                 ("ricerca", "web_search"), ("immagine", "image_generation"),
                                 ("calcolo", "code_interpreter"), ("ricorda", "memory"),
                                 ("appunti", "notes"), ("riunione", "calendar"),
                                 ("schedule", "automations"), ("chat history", "chats"),
                                 ("canale", "channels")):
            with self.subTest(keyword=keyword):
                self.assertTrue(self.infer(keyword)["builtinTools"].get(builtin))

    def test_vagueness_uses_whole_keyword_not_substring(self):
        self.assertTrue(mc._is_vague_request({"description": "Un assistant pour la rédaction et pour annoter."}))
        self.assertFalse(mc._is_vague_request({"description": "Un assistant pour la rédaction de notes."}))


class CreateModelGateTests(unittest.IsolatedAsyncioTestCase):
    def draft(self, **bindings):
        return {"id": "test", "name": "Test", "base_model_id": "local",
                "description": "test", "system": "test", **bindings}

    async def test_vague_omitted_bindings_still_ask_instead_of_saving(self):
        with patch.object(mc, "_models_api", new_callable=AsyncMock) as api:
            await mc.Tools().create_model(self.draft())
            api.assert_not_awaited()

    async def test_vague_explicit_empty_saves_without_catalog_question(self):
        for key in ("skills", "tools", "knowledge"):
            with self.subTest(key=key):
                with patch.object(mc, "_models_api", new_callable=AsyncMock,
                                  return_value=(200, {})) as api:
                    reply = await mc.Tools().create_model(self.draft(**{key: []}))
                    api.assert_awaited_once()
                    self.assertEqual(api.call_args.kwargs["path"], "/create")
                    self.assertIn("Model created:", reply)

    async def test_update_keeps_empty_bindings_in_actual_api_payload(self):
        with patch.object(mc, "_models_api", new_callable=AsyncMock,
                          return_value=(200, {})) as api:
            await mc.Tools().update_model("test", self.draft(skills=[], tools=[], knowledge=[],
                actions=[], default_features=[], builtin_tools={}, capabilities={}),
                context="skill tool action knowledge python image chantier")
            api.assert_awaited_once()
            form = api.call_args.kwargs["json_body"]
            for key in ("skillIds", "toolIds", "actionIds", "knowledge", "defaultFeatureIds"):
                self.assertEqual(form["meta"][key], [])
            self.assertEqual(form["meta"]["builtinTools"], {})
            self.assertEqual(form["meta"]["capabilities"], {})


if __name__ == "__main__":
    unittest.main()
