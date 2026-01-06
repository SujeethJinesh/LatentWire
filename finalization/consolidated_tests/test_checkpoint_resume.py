#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test checkpoint saving and resuming functionality.

This script verifies that:
1. Checkpoints can be saved correctly with all necessary state
2. Checkpoints can be loaded and training resumed from exact step
3. Atomic saving prevents corruption
4. Pruning works correctly
5. Resume maintains exact training state
"""

import os
import sys
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch not available. Cannot run checkpoint tests.")
    sys.exit(1)

from checkpoint_manager import CheckpointManager, ExperimentCheckpointer
from latentwire.checkpointing import save_latest_checkpoint, prune_save_dir


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.register_buffer('step_count', torch.tensor(0))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def increment_step(self):
        self.step_count += 1


class TestCheckpointResume:
    """Test suite for checkpoint saving and resuming."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_dir = None
        self.passed = 0
        self.failed = 0
        self.tests_run = []

    def setup(self):
        """Setup test directory."""
        self.test_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
        if self.verbose:
            print(f"📁 Test directory: {self.test_dir}")

    def teardown(self):
        """Clean up test directory."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            if self.verbose:
                print("🧹 Cleaned up test directory")

    def log_result(self, test_name: str, passed: bool, msg: str = ""):
        """Log test result."""
        self.tests_run.append(test_name)
        if passed:
            self.passed += 1
            print(f"✅ {test_name}: PASSED {msg}")
        else:
            self.failed += 1
            print(f"❌ {test_name}: FAILED {msg}")

    def test_basic_save_load(self) -> bool:
        """Test basic checkpoint save and load."""
        test_name = "Basic Save/Load"
        try:
            # Create model and optimizer
            model = DummyModel()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Set some state
            model.increment_step()
            model.increment_step()

            # Do a forward pass to create optimizer state
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

            # Save checkpoint
            manager = CheckpointManager(
                save_dir=os.path.join(self.test_dir, "basic"),
                verbose=False
            )

            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': 100,
                'epoch': 5,
                'loss': 0.5,
            }

            saved_path = manager.save_checkpoint(
                state=state,
                step=100,
                epoch=5
            )

            # Verify saved files exist
            assert os.path.exists(saved_path), f"Checkpoint dir not created: {saved_path}"
            assert os.path.exists(os.path.join(saved_path, "state.pt")), "state.pt not found"
            assert os.path.exists(os.path.join(saved_path, "metadata.json")), "metadata.json not found"

            # Load checkpoint
            loaded_state = manager.load_checkpoint(saved_path)
            assert loaded_state is not None, "Failed to load checkpoint"
            assert loaded_state['step'] == 100, f"Step mismatch: {loaded_state['step']} != 100"
            assert loaded_state['epoch'] == 5, f"Epoch mismatch: {loaded_state['epoch']} != 5"
            assert abs(loaded_state['loss'] - 0.5) < 1e-6, f"Loss mismatch: {loaded_state['loss']} != 0.5"

            # Create new model and load state
            model2 = DummyModel()
            model2.load_state_dict(loaded_state['model'])

            # Verify model state
            assert model2.step_count.item() == 2, f"Model step count mismatch: {model2.step_count.item()} != 2"

            # Verify parameters match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                assert n1 == n2, f"Parameter name mismatch: {n1} != {n2}"
                assert torch.allclose(p1, p2), f"Parameter value mismatch for {n1}"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_resume_exact_state(self) -> bool:
        """Test that resuming maintains exact training state."""
        test_name = "Resume Exact State"
        try:
            # Create initial training state
            model = DummyModel()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

            # Train for a few steps
            torch.manual_seed(42)
            for step in range(5):
                x = torch.randn(4, 10)
                y = model(x)
                loss = y.sum()
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.increment_step()

            original_loss = loss.item()
            original_lr = scheduler.get_last_lr()[0]
            original_step = model.step_count.item()

            # Save checkpoint
            checkpointer = ExperimentCheckpointer({
                'output_dir': os.path.join(self.test_dir, "resume"),
                'save_interval': 1,
                'verbose': False,
            })

            metrics = {'loss': original_loss, 'lr': original_lr}
            saved_path = checkpointer.save_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                step=5,
                metrics=metrics,
                best_metric=original_loss
            )

            # Create new model and optimizer
            model2 = DummyModel()
            optimizer2 = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
            scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10)

            # Resume from checkpoint
            epoch, step, best_metric, loaded_metrics = checkpointer.resume_training(
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2
            )

            # Verify state matches
            assert epoch == 1, f"Epoch mismatch: {epoch} != 1"
            assert step == 5, f"Step mismatch: {step} != 5"
            assert abs(best_metric - original_loss) < 1e-6, f"Best metric mismatch"
            assert abs(loaded_metrics['loss'] - original_loss) < 1e-6, "Loss metric mismatch"
            assert model2.step_count.item() == original_step, f"Model step mismatch"

            # Verify optimizer state
            assert len(optimizer2.state) == len(optimizer.state), "Optimizer state length mismatch"

            # Verify scheduler state
            assert scheduler2.get_last_lr()[0] == original_lr, f"LR mismatch: {scheduler2.get_last_lr()[0]} != {original_lr}"

            # Continue training from same random seed position
            torch.manual_seed(42)
            for _ in range(5):  # Skip first 5 steps
                torch.randn(4, 10)

            # Next batch should be identical
            x_orig = torch.randn(4, 10)
            x_resumed = torch.randn(4, 10)
            assert torch.allclose(x_orig, x_resumed), "Random state not preserved"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_atomic_saving(self) -> bool:
        """Test atomic saving prevents corruption."""
        test_name = "Atomic Saving"
        try:
            save_dir = os.path.join(self.test_dir, "atomic")
            os.makedirs(save_dir, exist_ok=True)

            # Test atomic torch save
            model = DummyModel()
            artifacts = {
                'encoder.pt': model.state_dict(),
                'config.json': {'test': 'config', 'value': 123},
                'training_stats.json': {'step': 100, 'loss': 0.5},
            }

            # Save with atomic operations
            freed_pre, freed_post = save_latest_checkpoint(
                save_dir=save_dir,
                artifacts=artifacts,
                pre_prune=True,
                post_prune=True,
                verbose=False
            )

            # Verify files exist and are valid
            assert os.path.exists(os.path.join(save_dir, 'encoder.pt')), "encoder.pt not saved"
            assert os.path.exists(os.path.join(save_dir, 'config.json')), "config.json not saved"
            assert os.path.exists(os.path.join(save_dir, 'training_stats.json')), "training_stats.json not saved"

            # Verify no temp files remain
            for f in os.listdir(save_dir):
                assert not f.endswith('.tmp'), f"Temp file remained: {f}"
                assert not f.endswith('.partial'), f"Partial file remained: {f}"

            # Load and verify content
            loaded_state = torch.load(os.path.join(save_dir, 'encoder.pt'))
            assert 'fc1.weight' in loaded_state, "Model state corrupted"

            with open(os.path.join(save_dir, 'config.json'), 'r') as f:
                config = json.load(f)
                assert config['test'] == 'config', "Config corrupted"
                assert config['value'] == 123, "Config value corrupted"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_pruning(self) -> bool:
        """Test checkpoint pruning removes old files."""
        test_name = "Checkpoint Pruning"
        try:
            save_dir = os.path.join(self.test_dir, "pruning")
            os.makedirs(save_dir, exist_ok=True)

            # Create some old files that should be removed
            old_files = [
                'step_100/state.pt',
                'step_200/state.pt',
                'encoder_step300.pt',
                'state_step400.pt',
                'temp.tmp',
                'backup.bak',
                'old_model.pt.old',
                'random_file.txt',
            ]

            for f in old_files:
                fpath = os.path.join(save_dir, f)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, 'w') as fp:
                    fp.write("old data")

            # Files that should be kept
            keep_files = ['encoder.pt', 'config.json', 'training_stats.json']

            # Prune directory
            freed = prune_save_dir(save_dir, keep_only=keep_files)

            # Verify old files removed
            remaining = set(os.listdir(save_dir))
            for old in ['step_100', 'step_200', 'temp.tmp', 'backup.bak']:
                assert old not in remaining, f"Failed to remove: {old}"

            # Save new checkpoint
            model = DummyModel()
            artifacts = {
                'encoder.pt': model.state_dict(),
                'config.json': {'test': 'new'},
            }

            save_latest_checkpoint(
                save_dir=save_dir,
                artifacts=artifacts,
                pre_prune=True,
                post_prune=True,
                verbose=False
            )

            # Verify only canonical files remain
            final_files = set(os.listdir(save_dir))
            assert 'encoder.pt' in final_files, "encoder.pt missing"
            assert 'config.json' in final_files, "config.json missing"
            assert 'random_file.txt' not in final_files, "Non-canonical file not removed"

            self.log_result(test_name, True, f"- Freed {freed} bytes")
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_checkpoint_manager_intervals(self) -> bool:
        """Test checkpoint manager save intervals."""
        test_name = "Save Intervals"
        try:
            manager = CheckpointManager(
                save_dir=os.path.join(self.test_dir, "intervals"),
                save_interval=100,
                keep_last_n=2,
                verbose=False
            )

            # Test should_save logic
            assert not manager.should_save(50), "Should not save at step 50"
            assert manager.should_save(100), "Should save at step 100"
            assert not manager.should_save(150), "Should not save at step 150"
            assert manager.should_save(200), "Should save at step 200"
            assert manager.should_save(50, force=True), "Should save when forced"

            # Test keep_last_n
            for step in [100, 200, 300, 400]:
                state = {'step': step}
                manager.save_checkpoint(state, step=step)

            # Should only have 2 latest checkpoints
            checkpoints = list(Path(manager.save_dir).glob("step_*"))
            assert len(checkpoints) == 2, f"Expected 2 checkpoints, found {len(checkpoints)}"

            # Verify they are the latest ones
            names = [c.name for c in checkpoints]
            assert 'step_300' in names, "step_300 should be kept"
            assert 'step_400' in names, "step_400 should be kept"
            assert 'step_100' not in names, "step_100 should be removed"
            assert 'step_200' not in names, "step_200 should be removed"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_symlinks(self) -> bool:
        """Test latest and best symlinks."""
        test_name = "Symlinks"
        try:
            manager = CheckpointManager(
                save_dir=os.path.join(self.test_dir, "symlinks"),
                verbose=False
            )

            # Save regular checkpoint
            state1 = {'step': 100, 'metric': 0.8}
            path1 = manager.save_checkpoint(state1, step=100)

            # Check latest symlink
            latest_link = Path(manager.save_dir) / "latest"
            assert latest_link.exists(), "Latest symlink not created"
            assert latest_link.is_symlink(), "Latest is not a symlink"

            # Save better checkpoint
            state2 = {'step': 200, 'metric': 0.9}
            path2 = manager.save_checkpoint(state2, step=200, is_best=True)

            # Check both symlinks
            best_link = Path(manager.save_dir) / "best"
            assert best_link.exists(), "Best symlink not created"
            assert best_link.is_symlink(), "Best is not a symlink"
            assert latest_link.readlink() == Path("step_200"), "Latest symlink not updated"
            assert best_link.readlink() == Path("step_200"), "Best symlink incorrect"

            # Save worse checkpoint - best shouldn't change
            state3 = {'step': 300, 'metric': 0.7}
            path3 = manager.save_checkpoint(state3, step=300, is_best=False)

            assert latest_link.readlink() == Path("step_300"), "Latest should update"
            assert best_link.readlink() == Path("step_200"), "Best should not change"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def test_find_latest_checkpoint(self) -> bool:
        """Test finding latest checkpoint."""
        test_name = "Find Latest"
        try:
            save_dir = os.path.join(self.test_dir, "find_latest")
            manager = CheckpointManager(save_dir=save_dir, verbose=False)

            # No checkpoints initially
            assert manager.find_latest_checkpoint() is None, "Should find no checkpoints"

            # Create some checkpoints
            for step in [100, 50, 200, 150]:
                state = {'step': step}
                manager.save_checkpoint(state, step=step)

            # Should find step_200
            latest = manager.find_latest_checkpoint()
            assert latest is not None, "Should find checkpoint"
            assert "step_200" in latest, f"Should find step_200, got {latest}"

            # Test with epoch checkpoints
            epoch_dir = os.path.join(save_dir, "epoch5")
            os.makedirs(epoch_dir, exist_ok=True)
            torch.save({'epoch': 5}, os.path.join(epoch_dir, 'state.pt'))

            # Step 200 should still be latest (200 > 5*10000 would be false, but we have actual step 200)
            latest = manager.find_latest_checkpoint()
            assert "step_200" in latest, "Step checkpoint should be preferred"

            self.log_result(test_name, True)
            return True

        except Exception as e:
            self.log_result(test_name, False, f"- {str(e)}")
            if self.verbose:
                traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*60)
        print("🧪 CHECKPOINT RESUME TESTS")
        print("="*60 + "\n")

        self.setup()

        try:
            # Run tests in order
            self.test_basic_save_load()
            self.test_resume_exact_state()
            self.test_atomic_saving()
            self.test_pruning()
            self.test_checkpoint_manager_intervals()
            self.test_symlinks()
            self.test_find_latest_checkpoint()

        finally:
            self.teardown()

        # Print summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {self.passed}/{self.passed + self.failed}")
        print(f"❌ Failed: {self.failed}/{self.passed + self.failed}")

        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n⚠️  {self.failed} tests failed:")
            for test in self.tests_run:
                if test.startswith("❌"):
                    print(f"  - {test}")
            return False


def main():
    """Main test runner."""
    tester = TestCheckpointResume(verbose=True)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()