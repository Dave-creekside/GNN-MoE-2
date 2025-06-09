import torch
import torch.optim as optim
import unittest
from dataclasses import dataclass

# Import the components to be tested from the ghost architecture
from ..gnn_moe_config import GhostMoEConfig
from ..gnn_moe_architecture import (
    GhostAwareExpertBlock,
    ExpertSaturationMonitor,
    GhostActivationController,
    PrimaryGhostLRScheduler,
    GhostMoEModel,
    create_dynamic_optimizer
)

# --- Minimal Configuration for Testing ---
@dataclass
class MinimalGhostConfig(GhostMoEConfig):
    # Architecture
    embed_dim: int = 8
    num_experts: int = 2
    num_ghost_experts: int = 2
    num_layers: int = 1
    num_heads: int = 2
    max_seq_length: int = 16
    vocab_size: int = 32

    # Ghost specific
    ghost_activation_threshold: float = 0.5
    ghost_learning_rate: float = 1e-4
    max_steps: int = 100 # For LR scheduler test

# --- Test Cases ---

class TestGhostComponents(unittest.TestCase):

    def setUp(self):
        """Set up a minimal config for all tests."""
        self.config = MinimalGhostConfig()

    def test_ghost_aware_expert_block_activation(self):
        """Tests that a ghost expert's output is scaled by its activation level."""
        print("\nRunning test: test_ghost_aware_expert_block_activation")
        ghost_expert = GhostAwareExpertBlock(self.config, is_ghost=True)
        ghost_expert.eval() # Disable dropout for deterministic test

        # Set activation level to 50%
        ghost_expert.activation_level = 0.5
        
        dummy_input = torch.randn(1, self.config.max_seq_length, self.config.embed_dim)
        
        # Get the output without activation scaling first
        with torch.no_grad():
            # Temporarily set activation to 1.0 to get the full output
            ghost_expert.activation_level = 1.0
            full_output = ghost_expert(dummy_input)

            # Set activation back to 0.5 for the actual test
            ghost_expert.activation_level = 0.5
            scaled_output = ghost_expert(dummy_input)

        # The scaled output should be half of the full output
        self.assertTrue(torch.allclose(scaled_output, full_output * 0.5))
        print("✅ GhostAwareExpertBlock correctly scales output.")

    def test_saturation_detection(self):
        """Tests the ExpertSaturationMonitor logic."""
        print("\nRunning test: test_saturation_detection")
        monitor = ExpertSaturationMonitor(self.config)
        
        # Case 1: Low orthogonality, low variance (no saturation)
        primary_outputs_redundant = [
            torch.randn(1, 4, self.config.embed_dim),
            torch.randn(1, 4, self.config.embed_dim) * 0.1 # Second expert is similar
        ]
        input_features_low_var = torch.stack(primary_outputs_redundant, dim=2).mean(dim=2)
        metrics_case1 = monitor.compute_saturation_metrics(primary_outputs_redundant, input_features_low_var)
        self.assertFalse(metrics_case1['needs_ghost_activation'])
        print(f"✅ Correctly detects no saturation (low ortho, low var). Saturation: {metrics_case1['saturation_level']:.4f}")

        # Case 2: High orthogonality, but low unexplained variance (no saturation)
        primary_outputs_ortho = [
            torch.randn(1, 4, self.config.embed_dim),
            torch.randn(1, 4, self.config.embed_dim)
        ]
        # Make them orthogonal by flattening them into vectors for the dot product
        vec0 = primary_outputs_ortho[0].view(-1)
        vec1 = primary_outputs_ortho[1].view(-1)
        proj = (torch.dot(vec1, vec0) / torch.dot(vec0, vec0)) * vec0
        vec1_ortho = vec1 - proj
        primary_outputs_ortho[1] = vec1_ortho.view(1, 4, self.config.embed_dim)

        input_features_low_var_2 = torch.stack(primary_outputs_ortho, dim=2).mean(dim=2)
        metrics_case2 = monitor.compute_saturation_metrics(primary_outputs_ortho, input_features_low_var_2)
        self.assertFalse(metrics_case2['needs_ghost_activation'])
        print(f"✅ Correctly detects no saturation (high ortho, low var). Saturation: {metrics_case2['saturation_level']:.4f}")

        # Case 3: High orthogonality and high unexplained variance (saturation)
        input_features_high_var = input_features_low_var_2 + torch.randn_like(input_features_low_var_2) * 5
        metrics_case3 = monitor.compute_saturation_metrics(primary_outputs_ortho, input_features_high_var)
        self.assertTrue(metrics_case3['needs_ghost_activation'])
        print(f"✅ Correctly detects saturation (high ortho, high var). Saturation: {metrics_case3['saturation_level']:.4f}")

    def test_activation_controller(self):
        """Tests the GhostActivationController state transitions."""
        print("\nRunning test: test_activation_controller")
        controller = GhostActivationController(self.config)
        
        # Initially, all ghosts are dormant
        self.assertEqual(controller.ghost_states, ["dormant", "dormant"])
        self.assertTrue(torch.all(controller.activation_rates == 0.0))

        # Step 1: No saturation signal
        metrics_no_sat = {'needs_ghost_activation': False}
        controller.update_ghost_activations(metrics_no_sat, step=1)
        self.assertEqual(controller.ghost_states, ["dormant", "dormant"])
        
        # Step 2: Saturation signal received, first ghost should start activating
        metrics_sat = {'needs_ghost_activation': True}
        rates = controller.update_ghost_activations(metrics_sat, step=2)
        self.assertEqual(controller.ghost_states, ["activating", "dormant"])
        self.assertAlmostEqual(rates[0].item(), 0.01, msg="First ghost should be initialized to 0.01")
        self.assertEqual(rates[1].item(), 0.0, msg="Second ghost should remain dormant")
        print("✅ Correctly activates first ghost on saturation signal.")

        # Step 3: Continue activating first ghost (no new saturation signal)
        rates = controller.update_ghost_activations(metrics_no_sat, step=3) # Signal doesn't need to persist
        self.assertEqual(controller.ghost_states, ["activating", "dormant"])
        self.assertAlmostEqual(rates[0].item(), 0.02, msg="First ghost activation should ramp up")
        print("✅ Correctly ramps up activation for first ghost.")

        # Step 4: Saturation signal again, second ghost should now activate
        rates = controller.update_ghost_activations(metrics_sat, step=4)
        self.assertEqual(controller.ghost_states, ["activating", "activating"], msg="Second ghost should now be activating")
        # The first ghost's rate is NOT ramped up in this step, because the function returned early
        # after activating the second ghost. This is the correct, desired behavior.
        self.assertAlmostEqual(rates[0].item(), 0.02, msg="First ghost's rate should not change when a new ghost is activated")
        self.assertAlmostEqual(rates[1].item(), 0.01, msg="Second ghost should be initialized to 0.01")
        print("✅ Correctly activates second ghost on subsequent signal.")

    def test_lr_scheduler(self):
        """Tests the inverse learning rate dynamics of the scheduler."""
        print("\nRunning test: test_lr_scheduler")
        model = GhostMoEModel(self.config)
        optimizer = create_dynamic_optimizer(model, self.config)
        scheduler = PrimaryGhostLRScheduler(self.config, optimizer)

        # Before the first step, the LR should be the initial value.
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], self.config.learning_rate)
        print("✅ Correctly sets initial LRs before first step.")

        # At step 0, after one step
        primary_lr_start, ghost_lrs_start = scheduler.step(torch.tensor([0.0, 0.0]))
        self.assertLess(primary_lr_start, self.config.learning_rate)
        # Ghosts are inactive, so their LR should be 0
        self.assertTrue(all(lr == 0.0 for lr in ghost_lrs_start))
        print("✅ Correctly decays LR on first step (ghosts inactive).")

        # Simulate some steps for the primary scheduler to decay
        for _ in range(self.config.max_steps // 2):
            scheduler.primary_scheduler.step()

        # At mid-training, with ghosts now active
        mid_primary_lr = scheduler.primary_scheduler.get_last_lr()[0]
        ghost_activations = torch.tensor([1.0, 0.5]) # One fully active, one half active
        primary_lr_mid, ghost_lrs_mid = scheduler.step(ghost_activations)
        
        self.assertLess(primary_lr_mid, primary_lr_start)
        
        # Check inverse relationship: as primary LR decreased, ghost LR should increase
        # The effective ghost LR is scaled by both the inverse factor and its activation level
        lr_decay_factor = primary_lr_mid / self.config.learning_rate
        ghost_lr_factor = 1.0 - lr_decay_factor
        
        expected_ghost_lr_0 = self.config.ghost_learning_rate * ghost_lr_factor * ghost_activations[0]
        expected_ghost_lr_1 = self.config.ghost_learning_rate * ghost_lr_factor * ghost_activations[1]

        self.assertAlmostEqual(ghost_lrs_mid[0], expected_ghost_lr_0)
        self.assertAlmostEqual(ghost_lrs_mid[1], expected_ghost_lr_1)
        print("✅ Correctly computes inverse LRs at mid-training (ghosts active).")

    def test_model_forward_pass(self):
        """Ensures the full GhostMoEModel can perform a forward pass without errors."""
        print("\nRunning test: test_model_forward_pass")
        model = GhostMoEModel(self.config)
        model.eval()

        dummy_input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.max_seq_length))
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        
        try:
            with torch.no_grad():
                outputs = model(
                    dummy_input_ids,
                    step=1,
                    attention_mask=dummy_attention_mask,
                    return_loss=False
                )
            # Check output shape
            self.assertEqual(
                outputs['logits'].shape,
                (2, self.config.max_seq_length, self.config.vocab_size)
            )
            print("✅ Model forward pass completed successfully with correct output shape.")
        except Exception as e:
            self.fail(f"Model forward pass failed with an exception: {e}")


if __name__ == '__main__':
    unittest.main()
