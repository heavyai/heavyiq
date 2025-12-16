"""
Unit tests for heavyiq.lcel.chains.utils module.

Tests the get_value_from_runnable_binding function to ensure it correctly
extracts underlying LLM and prompt instances from configurable runnables.
"""

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

# Import directly from utils module to avoid importing chains at module level
from heavyiq.lcel.chains.utils import configure_step, get_value_from_runnable_binding, merge_dicts


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test_merge_flat_dicts(self):
        """Test merging two flat dictionaries."""
        d1 = {"a": 1, "b": 2}
        d2 = {"c": 3, "d": 4}
        result = merge_dicts(d1, d2)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_overlapping_keys(self):
        """Test that d2 values override d1 values for overlapping keys."""
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 3, "c": 4}
        result = merge_dicts(d1, d2)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries recursively."""
        d1 = {"a": {"x": 1, "y": 2}, "b": 3}
        d2 = {"a": {"y": 20, "z": 30}, "c": 4}
        result = merge_dicts(d1, d2)
        assert result == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}

    def test_original_dicts_unchanged(self):
        """Test that original dictionaries are not modified."""
        d1 = {"a": 1}
        d2 = {"b": 2}
        merge_dicts(d1, d2)
        assert d1 == {"a": 1}
        assert d2 == {"b": 2}


class TestGetValueFromRunnableBindingWithLLM:
    """Tests for get_value_from_runnable_binding with LLM runnables."""

    def test_get_llm_from_configured_runnable(self, heavyiq_config):
        """Test extracting LLM from a configured runnable with alternatives."""
        from heavyiq.langchain.llms import OverrideVLLMOpenAI
        from heavyiq.lcel.llms import llm_runnable

        # Configure for nl_to_tables LLM type
        nl_to_tables_llm_rbl = llm_runnable.with_config(
            configurable={"llm": "nl_to_tables", "llm_temperature": 0},
            config={"tags": ["nl_to_tables_llm"]},
        )

        # Get the underlying LLM
        llm = get_value_from_runnable_binding(nl_to_tables_llm_rbl)

        # Should return OverrideVLLMOpenAI when config uses API_VLLM
        assert llm is not None
        assert isinstance(llm, OverrideVLLMOpenAI), f"Expected OverrideVLLMOpenAI, got {type(llm).__name__}"
        assert llm.openai_api_base == heavyiq_config.custom_llm_api_base

    def test_get_llm_from_nl_to_sql_runnable(self, heavyiq_config):
        """Test extracting LLM from nl_to_sql configured runnable."""
        from heavyiq.langchain.llms import OverrideVLLMOpenAI
        from heavyiq.lcel.llms import llm_runnable

        nl_to_sql_llm_rbl = llm_runnable.with_config(
            configurable={"llm": "nl_to_sql", "llm_temperature": 0},
            config={"tags": ["nl_to_sql_llm"]},
        )

        llm = get_value_from_runnable_binding(nl_to_sql_llm_rbl)

        # Should return OverrideVLLMOpenAI when config uses API_VLLM
        assert llm is not None
        assert isinstance(llm, OverrideVLLMOpenAI), f"Expected OverrideVLLMOpenAI, got {type(llm).__name__}"
        assert llm.temperature == 0

    def test_get_llm_preserves_config_after_configure_step(self, heavyiq_config):
        """Test that configure_step preserves the configurable settings."""
        from heavyiq.langchain.llms import OverrideVLLMOpenAI
        from heavyiq.lcel.llms import llm_runnable

        # Create configured LLM runnable
        nl_to_tables_llm_rbl = llm_runnable.with_config(
            configurable={"llm": "nl_to_tables", "llm_temperature": 0},
            config={"tags": ["nl_to_tables_llm"]},
        )

        # Wrap with configure_step (like in table_chain.py)
        model = configure_step(nl_to_tables_llm_rbl, run_name="Call LLM", step="Calling LLM.")

        # Verify configurable settings are preserved
        assert model.config.get("configurable", {}).get("llm") == "nl_to_tables"
        assert model.config.get("configurable", {}).get("llm_temperature") == 0

        # Get the underlying LLM - should still be OverrideVLLMOpenAI after configure_step
        llm = get_value_from_runnable_binding(model)

        assert llm is not None
        assert isinstance(llm, OverrideVLLMOpenAI), f"Expected OverrideVLLMOpenAI, got {type(llm).__name__}"
        assert llm.openai_api_base == heavyiq_config.custom_llm_api_base


class TestGetValueFromRunnableBindingWithPrompt:
    """Tests for get_value_from_runnable_binding with prompt runnables."""

    def test_get_prompt_from_configured_runnable(self):
        """Test extracting PromptTemplate from a configured runnable."""
        # Create a simple configurable prompt
        default_template = "Default: {input}"
        custom_template = "Custom: {input}"

        prompt_runnable = PromptTemplate.from_template(default_template).configurable_alternatives(
            ConfigurableField(id="prompt"),
            default_key="openai",
            custom=PromptTemplate.from_template(custom_template),
        )

        # Get the default prompt
        default_prompt = get_value_from_runnable_binding(prompt_runnable)
        assert isinstance(default_prompt, PromptTemplate)
        assert "Default" in default_prompt.template

    def test_get_custom_prompt_from_configured_runnable(self):
        """Test extracting custom PromptTemplate from a configured runnable."""
        default_template = "Default: {input}"
        custom_template = "Custom: {input}"

        prompt_runnable = PromptTemplate.from_template(default_template).configurable_alternatives(
            ConfigurableField(id="prompt"),
            default_key="openai",
            custom=PromptTemplate.from_template(custom_template),
        )

        # Configure for custom prompt
        custom_prompt_rbl = prompt_runnable.with_config(configurable={"prompt": "custom"})

        custom_prompt = get_value_from_runnable_binding(custom_prompt_rbl)
        assert isinstance(custom_prompt, PromptTemplate)
        assert "Custom" in custom_prompt.template

    def test_get_prompt_from_to_tables_runnable(self):
        """Test extracting prompt from to_tables_prompt_runnable."""
        from heavyiq.lcel.prompts import to_tables_prompt_runnable

        # Get the default prompt
        prompt = get_value_from_runnable_binding(to_tables_prompt_runnable)
        assert isinstance(prompt, PromptTemplate)

    def test_get_prompt_from_to_sql_runnable(self):
        """Test extracting prompt from to_sql_prompt_runnable."""
        from heavyiq.lcel.prompts import to_sql_prompt_runnable

        # Get the default prompt
        prompt = get_value_from_runnable_binding(to_sql_prompt_runnable)
        assert isinstance(prompt, PromptTemplate)

    def test_get_custom_prompt_from_to_sql_runnable(self):
        """Test extracting custom prompt from to_sql_prompt_runnable."""
        from heavyiq.lcel.prompts import to_sql_prompt_runnable

        custom_prompt_rbl = to_sql_prompt_runnable.with_config(configurable={"prompt": "custom"})

        prompt = get_value_from_runnable_binding(custom_prompt_rbl)
        assert isinstance(prompt, PromptTemplate)


class TestConfigureStep:
    """Tests for configure_step function."""

    def test_configure_step_preserves_configurable(self):
        """Test that configure_step preserves existing configurable settings."""
        # Create a simple configurable runnable
        template = "Test: {input}"
        prompt_runnable = PromptTemplate.from_template(template).configurable_alternatives(
            ConfigurableField(id="prompt"),
            default_key="default",
        )

        # Add configurable settings
        configured = prompt_runnable.with_config(
            configurable={"prompt": "default", "extra_key": "extra_value"},
            config={"tags": ["original"]},
        )

        # Apply configure_step
        step_configured = configure_step(configured, run_name="Test Step", step="Testing")

        # Check that configurable settings are preserved
        assert step_configured.config.get("configurable", {}).get("prompt") == "default"
        assert step_configured.config.get("configurable", {}).get("extra_key") == "extra_value"

        # Check that step metadata is added
        assert step_configured.config.get("run_name") == "Test Step"
        assert "intermediate-step" in step_configured.config.get("tags", [])
        assert step_configured.config.get("metadata", {}).get("step") == "Testing"

    def test_configure_step_with_empty_config(self):
        """Test configure_step with a runnable that has no existing config."""
        template = "Test: {input}"
        prompt = PromptTemplate.from_template(template)

        step_configured = configure_step(prompt, run_name="Test Step", step="Testing")

        # Should still work and add the step metadata
        assert step_configured.config.get("run_name") == "Test Step"
        assert "intermediate-step" in step_configured.config.get("tags", [])

