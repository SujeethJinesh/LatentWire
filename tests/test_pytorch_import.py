"""Tests for PyTorch import error handling."""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pytorch_import_available():
    """Test that PyTorch can be imported (or skipped if not available)."""
    try:
        import torch
        # If we get here, PyTorch is available
        assert True
        print("PyTorch is available and importable")
    except ImportError as e:
        # PyTorch not available in test environment - that's OK
        pytest.skip(f"PyTorch not available in test environment: {e}")


def test_eval_module_pytorch_check():
    """Test that eval module has PyTorch availability check."""
    try:
        # Import the eval module
        import latentwire.eval as eval_module

        # Check that the module has the PYTORCH_AVAILABLE flag
        assert hasattr(eval_module, 'PYTORCH_AVAILABLE')

        # If PyTorch is not available, there should be an error message
        if not eval_module.PYTORCH_AVAILABLE:
            assert hasattr(eval_module, 'PYTORCH_IMPORT_ERROR')
            assert isinstance(eval_module.PYTORCH_IMPORT_ERROR, str)
    except ImportError as e:
        # If transformers or other dependencies are missing, skip
        if "transformers" in str(e) or "torch" in str(e):
            pytest.skip(f"Required dependencies not available: {e}")
        raise


def test_train_module_pytorch_check():
    """Test that train module has PyTorch availability check."""
    try:
        # Import the train module
        import latentwire.train as train_module

        # Check that the module has the PYTORCH_AVAILABLE flag
        assert hasattr(train_module, 'PYTORCH_AVAILABLE')

        # If PyTorch is not available, there should be an error message
        if not train_module.PYTORCH_AVAILABLE:
            assert hasattr(train_module, 'PYTORCH_IMPORT_ERROR')
            assert isinstance(train_module.PYTORCH_IMPORT_ERROR, str)
    except ImportError as e:
        # If transformers or other dependencies are missing, skip
        if "transformers" in str(e) or "torch" in str(e):
            pytest.skip(f"Required dependencies not available: {e}")
        raise


def test_cli_eval_pytorch_import_check(capsys):
    """Test that CLI eval checks for PyTorch before running."""
    from latentwire.cli import eval as cli_eval

    # Mock torch to be unavailable
    with patch.dict('sys.modules'):
        # Remove torch from modules temporarily
        torch_module = sys.modules.pop('torch', None)

        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("Test: PyTorch not available")
            # Use the real import for everything else
            return __import__(name, *args, **kwargs)

        try:
            with patch('builtins.__import__', side_effect=mock_import):
                # Try to run eval - should exit with error message
                with pytest.raises(SystemExit) as exc_info:
                    cli_eval.run_eval(['--ckpt', 'dummy'])

                assert exc_info.value.code == 1

                # Check error message was printed
                captured = capsys.readouterr()
                assert "PyTorch is not properly installed" in captured.out
                assert "pip install torch" in captured.out
        finally:
            # Restore torch module if it existed
            if torch_module is not None:
                sys.modules['torch'] = torch_module


def test_cli_train_pytorch_import_check(capsys):
    """Test that CLI train checks for PyTorch before running."""
    from latentwire.cli import train as cli_train

    # Mock torch to be unavailable
    with patch.dict('sys.modules'):
        # Remove torch from modules temporarily
        torch_module = sys.modules.pop('torch', None)

        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("Test: PyTorch not available")
            # Use the real import for everything else
            return __import__(name, *args, **kwargs)

        try:
            with patch('builtins.__import__', side_effect=mock_import):
                # Try to run train - should exit with error message
                with pytest.raises(SystemExit) as exc_info:
                    cli_train.run_train(['--models', 'llama'])

                assert exc_info.value.code == 1

                # Check error message was printed
                captured = capsys.readouterr()
                assert "PyTorch is not properly installed" in captured.out
                assert "pip install torch" in captured.out
        finally:
            # Restore torch module if it existed
            if torch_module is not None:
                sys.modules['torch'] = torch_module


def test_error_message_formatting():
    """Test that error messages are properly formatted."""
    test_error = "dlopen(/path/to/torch/_C.cpython-39-darwin.so, 0x0002): Library not loaded: @loader_path/libtorch_cpu.dylib"

    try:
        # Import modules to check their error handling
        import latentwire.eval as eval_module
        import latentwire.train as train_module

        # If PyTorch is available, we can't test the error path
        if eval_module.PYTORCH_AVAILABLE:
            pytest.skip("PyTorch is available, cannot test error handling")

        # Check that error message is stored
        assert eval_module.PYTORCH_IMPORT_ERROR
        assert train_module.PYTORCH_IMPORT_ERROR

        # The stored error should be a string
        assert isinstance(eval_module.PYTORCH_IMPORT_ERROR, str)
        assert isinstance(train_module.PYTORCH_IMPORT_ERROR, str)
    except ImportError as e:
        # If transformers or other dependencies are missing, skip
        if "transformers" in str(e) or "torch" in str(e):
            pytest.skip(f"Required dependencies not available: {e}")
        raise