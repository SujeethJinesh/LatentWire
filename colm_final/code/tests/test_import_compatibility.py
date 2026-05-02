from __future__ import annotations


def test_public_package_imports_and_aliases() -> None:
    import latent_bridge
    import rotalign

    assert rotalign is latent_bridge
    assert "RotAlignKVTranslator" in latent_bridge.__all__
    assert latent_bridge.RotAlignKVTranslator is rotalign.RotAlignKVTranslator
    assert latent_bridge.TranslatorConfig is rotalign.TranslatorConfig


def test_script_modules_import_through_rotalign_alias() -> None:
    import latent_bridge.ablation_sweep as ablation_sweep
    import latent_bridge.calibrate as calibrate
    import latent_bridge.evaluate as evaluate

    assert callable(ablation_sweep.parse_accuracies)
    assert callable(calibrate.collect_kvs)
    assert callable(evaluate.eval_target_alone)
