#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_experiments.py

Comprehensive automated experiment suite for Geometric Constrained Learning paper.
Systematically evaluates all key hyperparameters and design choices to generate
robust, reproducible results for publication.

Usage:
    python paper_experiments.py [--dry-run] [--experiments EXPERIMENT_NAMES]
    
Examples:
    python paper_experiments.py                    # Run all experiments
    python paper_experiments.py --dry-run          # Show what would run
    python paper_experiments.py --experiments seed_study,lr_ratios  # Run specific experiments
"""

import os
import sys
import json
import time
import subprocess
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Suppress common warnings that clutter logs
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*MPS.*")

class PaperExperimentSuite:
    """Automated experiment suite for Geometric Constrained Learning paper data collection."""
    
    def __init__(self, base_output_dir="paper_experiments", dry_run=False):
        self.base_output_dir = Path(base_output_dir)
        self.dry_run = dry_run
        self.results_log = []
        self.start_time = datetime.now()
        
        # Base configuration for all experiments
        self.base_config = {
            "dataset_name": "Creekside/GRPO-Lambda-ParsedForUnsloth",
            "dataset_source": "huggingface", 
            "dataset_config_name": "default",
            "epochs": 3,
            "max_batches_per_epoch": 50,
            "batch_size": 2,
            "eval_every": 25,
            "embed_dim": 128,
            "num_experts": 2,
            "ghost_num_ghost_experts": 2,
            "training_mode": "geometric",
            "geometric_enabled": True,
            "geometric_rotation_dimensions": 4,
            "geometric_learning_rate": 0.001,
            "geometric_expert_learning_rate": 0.0001,
            "geometric_lambda_cognitive_rotations": True
        }
        
        # Create output directory structure
        if not dry_run:
            self.setup_output_directory()
    
    def setup_output_directory(self):
        """Create organized directory structure for experiment results."""
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each experiment type
        subdirs = [
            "seed_study", "rotation_dims", "lr_ratios", "expert_scales", 
            "loss_weights", "baselines", "datasets", "summary"
        ]
        for subdir in subdirs:
            (self.base_output_dir / subdir).mkdir(exist_ok=True)
        
        # Create experiment log
        self.log_file = self.base_output_dir / "experiment_log.json"
        print(f"📁 Experiment results will be saved to: {self.base_output_dir}")
    
    def run_experiment(self, exp_name, config_overrides, output_subdir, exp_index=1, total_exps=1):
        """Run a single experiment with given configuration."""
        # Merge base config with overrides
        config = {**self.base_config, **config_overrides}
        
        # Build command
        cmd = ["python", "run.py"]
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        # Set output directory
        output_dir = self.base_output_dir / output_subdir / exp_name
        cmd.extend(["--checkpoint_dir", str(output_dir)])
        
        print(f"\n{'='*60}")
        print(f"🚀 EXPERIMENT {exp_index}/{total_exps}: {exp_name}")
        print(f"📁 Output: {output_dir}")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print(f"   Command: {' '.join(cmd)}")
            return {"status": "dry_run", "duration": 0}
        
        # Run experiment with live streaming
        start_time = time.time()
        last_heartbeat = start_time
        heartbeat_interval = 30  # seconds
        
        try:
            # Create clean environment to suppress warnings
            clean_env = os.environ.copy()
            clean_env.update({
                "TOKENIZERS_PARALLELISM": "false",  # Suppress tokenizers warnings
                "PYTHONWARNINGS": "ignore::UserWarning",  # Suppress PyTorch warnings
            })
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=clean_env
            )
            
            output_lines = []
            print("📊 Live Training Progress:")
            print("-" * 40)
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                current_time = time.time()
                
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    # Display the line
                    print(output.strip())
                    output_lines.append(output.strip())
                    last_heartbeat = current_time
                else:
                    # Send heartbeat if no output for a while (for Colab stability)
                    if current_time - last_heartbeat > heartbeat_interval:
                        elapsed = current_time - start_time
                        print(f"💓 Heartbeat: {elapsed:.0f}s elapsed, experiment still running...")
                        last_heartbeat = current_time
                    time.sleep(0.1)  # Small delay to prevent busy waiting
            
            # Wait for process to complete
            return_code = process.poll()
            duration = time.time() - start_time
            
            print("-" * 40)
            if return_code == 0:
                print(f"✅ SUCCESS! Experiment completed in {duration:.1f}s ({duration/60:.1f} minutes)")
                status = "success"
            else:
                print(f"❌ FAILED! Return code: {return_code}")
                if output_lines:
                    print("Last few lines of output:")
                    for line in output_lines[-5:]:
                        print(f"   {line}")
                status = "failed"
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT after 1 hour")
            duration = 3600
            status = "timeout"
            process.kill()
        except KeyboardInterrupt:
            print(f"⚠️  INTERRUPTED by user")
            duration = time.time() - start_time
            status = "interrupted"
            process.kill()
            raise
        except Exception as e:
            print(f"💥 EXCEPTION: {e}")
            duration = time.time() - start_time
            status = "error"
        
        # Log result
        log_entry = {
            "experiment": exp_name,
            "config": config,
            "status": status,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir)
        }
        self.results_log.append(log_entry)
        
        # Print summary
        print(f"\n📋 Experiment Summary:")
        print(f"   Name: {exp_name}")
        print(f"   Status: {status}")
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        print(f"   Completed: {datetime.now().strftime('%H:%M:%S')}")
        
        return log_entry
    
    def seed_study(self):
        """Multi-seed consistency study (5 seeds)."""
        print("\n" + "="*60)
        print("🎲 SEED CONSISTENCY STUDY")
        print("Testing reproducibility across multiple random seeds")
        print("="*60)
        
        seeds = [42, 123, 456, 789, 999]
        results = []
        
        for i, seed in enumerate(seeds, 1):
            config = {"run_name": f"seed_{seed}", "seed": seed}
            result = self.run_experiment(f"seed_{seed}", config, "seed_study", i, len(seeds))
            results.append(result)
        
        return results
    
    def rotation_dimension_ablation(self):
        """Test different numbers of rotation dimensions."""
        print("\n" + "="*60)
        print("🔄 ROTATION DIMENSION ABLATION")
        print("Testing optimal number of rotation parameters")
        print("="*60)
        
        dimensions = [2, 4, 6, 8, 12]
        results = []
        
        for i, dims in enumerate(dimensions, 1):
            config = {
                "run_name": f"rotdims_{dims}",
                "geometric_rotation_dimensions": dims
            }
            result = self.run_experiment(f"dims_{dims}", config, "rotation_dims", i, len(dimensions))
            results.append(result)
        
        return results
    
    def learning_rate_ratio_study(self):
        """Test different geometric:expert learning rate ratios."""
        print("\n" + "="*60)
        print("📈 LEARNING RATE RATIO STUDY")
        print("Finding optimal ratio between rotation and expert learning rates")
        print("="*60)
        
        # Test ratios: 5:1, 10:1 (baseline), 20:1, 50:1
        lr_configs = [
            {"geo_lr": 0.0005, "exp_lr": 0.0001, "ratio": "5:1"},
            {"geo_lr": 0.001, "exp_lr": 0.0001, "ratio": "10:1"},   # baseline
            {"geo_lr": 0.002, "exp_lr": 0.0001, "ratio": "20:1"},
            {"geo_lr": 0.005, "exp_lr": 0.0001, "ratio": "50:1"}
        ]
        
        results = []
        for i, lr_config in enumerate(lr_configs, 1):
            config = {
                "run_name": f"lr_ratio_{lr_config['ratio'].replace(':', '_')}",
                "geometric_learning_rate": lr_config["geo_lr"],
                "geometric_expert_learning_rate": lr_config["exp_lr"]
            }
            result = self.run_experiment(f"ratio_{lr_config['ratio']}", config, "lr_ratios", i, len(lr_configs))
            results.append(result)
        
        return results
    
    def expert_scale_study(self):
        """Test different numbers of experts."""
        print("\n" + "="*60)
        print("👥 EXPERT SCALE STUDY")
        print("Testing scalability with different numbers of experts")
        print("="*60)
        
        # Test (primary, ghost) expert combinations
        expert_configs = [
            {"primary": 2, "ghost": 1},
            {"primary": 4, "ghost": 2},
            {"primary": 6, "ghost": 3},
            {"primary": 8, "ghost": 4}
        ]
        
        results = []
        for i, exp_config in enumerate(expert_configs, 1):
            config = {
                "run_name": f"experts_{exp_config['primary']}p_{exp_config['ghost']}g",
                "num_experts": exp_config["primary"],
                "ghost_num_ghost_experts": exp_config["ghost"],
                "embed_dim": 128  # Keep embed_dim constant to ensure fair comparison
            }
            result = self.run_experiment(
                f"experts_{exp_config['primary']}_{exp_config['ghost']}", 
                config, 
                "expert_scales",
                i, len(expert_configs)
            )
            results.append(result)
        
        return results
    
    def loss_weight_ablation(self):
        """Test different loss component weights."""
        print("\n" + "="*60)
        print("⚖️  LOSS COMPONENT WEIGHT ABLATION")
        print("Understanding the importance of each loss component")
        print("="*60)
        
        # Test different weight combinations
        weight_configs = [
            {"ortho": 0.5, "eff": 0.2, "spec": 0.3, "name": "default"},
            {"ortho": 1.0, "eff": 0.2, "spec": 0.3, "name": "high_orthogonality"},
            {"ortho": 0.5, "eff": 0.5, "spec": 0.3, "name": "high_efficiency"},
            {"ortho": 0.5, "eff": 0.2, "spec": 0.8, "name": "high_specialization"},
            {"ortho": 0.2, "eff": 0.1, "spec": 0.1, "name": "low_regularization"}
        ]
        
        results = []
        for i, weight_config in enumerate(weight_configs, 1):
            config = {
                "run_name": f"weights_{weight_config['name']}",
                "geometric_orthogonality_weight": weight_config["ortho"],
                "geometric_rotation_efficiency_weight": weight_config["eff"],
                "geometric_specialization_weight": weight_config["spec"]
            }
            result = self.run_experiment(weight_config["name"], config, "loss_weights", i, len(weight_configs))
            results.append(result)
        
        return results
    
    def baseline_comparison(self):
        """Compare geometric vs standard training."""
        print("\n" + "="*60)
        print("🆚 BASELINE COMPARISON")
        print("Geometric Constrained Learning vs Standard Training")
        print("="*60)
        
        configs = [
            {
                "run_name": "baseline_standard",
                "training_mode": "standard",
                "geometric_enabled": False
            },
            {
                "run_name": "baseline_geometric", 
                "training_mode": "geometric",
                "geometric_enabled": True
            }
        ]
        
        results = []
        for config in configs:
            result = self.run_experiment(config["run_name"], config, "baselines")
            results.append(result)
        
        return results
    
    def multi_dataset_study(self):
        """Test on different reasoning datasets."""
        print("\n" + "="*60)
        print("📚 MULTI-DATASET STUDY")
        print("Testing generalization across different reasoning domains")
        print("="*60)
        
        # Note: Some datasets may require different preprocessing
        datasets = [
            {
                "name": "lambda_calculus",
                "dataset_name": "Creekside/GRPO-Lambda-ParsedForUnsloth",
                "config_name": "default"
            },
            # Add more datasets here as they become available
            # {
            #     "name": "math_word_problems", 
            #     "dataset_name": "microsoft/orca-math-word-problems-200k",
            #     "config_name": "default"
            # }
        ]
        
        results = []
        for dataset in datasets:
            config = {
                "run_name": f"dataset_{dataset['name']}",
                "dataset_name": dataset["dataset_name"],
                "dataset_config_name": dataset["config_name"]
            }
            result = self.run_experiment(dataset["name"], config, "datasets")
            results.append(result)
        
        return results
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("📊 GENERATING SUMMARY REPORT")
        print("="*60)
        
        if self.dry_run:
            print("📝 Would generate summary report with all results")
            return
        
        # Calculate statistics
        total_experiments = len(all_results)
        successful = sum(1 for r in all_results if r["status"] == "success")
        failed = sum(1 for r in all_results if r["status"] == "failed")
        total_duration = sum(r["duration"] for r in all_results)
        
        summary = {
            "experiment_suite": "Geometric Constrained Learning Paper Data Collection",
            "timestamp": datetime.now().isoformat(),
            "total_experiments": total_experiments,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_experiments if total_experiments > 0 else 0,
            "total_duration_hours": total_duration / 3600,
            "average_experiment_duration_minutes": (total_duration / total_experiments / 60) if total_experiments > 0 else 0,
            "experiments": all_results
        }
        
        # Save detailed results
        summary_file = self.base_output_dir / "summary" / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate human-readable report
        report_file = self.base_output_dir / "summary" / "experiment_report.md"
        with open(report_file, 'w') as f:
            f.write("# Geometric Constrained Learning - Experiment Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Overall Results\n\n")
            f.write(f"- **Total Experiments:** {total_experiments}\n")
            f.write(f"- **Successful:** {successful}\n")
            f.write(f"- **Failed:** {failed}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1%}\n")
            f.write(f"- **Total Runtime:** {summary['total_duration_hours']:.1f} hours\n")
            f.write(f"- **Average per Experiment:** {summary['average_experiment_duration_minutes']:.1f} minutes\n\n")
            
            # Group results by experiment type
            by_type = {}
            for result in all_results:
                exp_type = result["experiment"].split("_")[0]
                if exp_type not in by_type:
                    by_type[exp_type] = []
                by_type[exp_type].append(result)
            
            f.write("## Results by Experiment Type\n\n")
            for exp_type, results in by_type.items():
                successful_type = sum(1 for r in results if r["status"] == "success")
                f.write(f"### {exp_type.replace('_', ' ').title()}\n")
                f.write(f"- Experiments: {len(results)}\n")
                f.write(f"- Successful: {successful_type}\n")
                f.write(f"- Success Rate: {successful_type/len(results):.1%}\n\n")
        
        print(f"📊 Summary report saved to: {report_file}")
        print(f"📈 Detailed results saved to: {summary_file}")
        
        # Print quick summary
        print(f"\n🎯 EXPERIMENT SUITE COMPLETE!")
        print(f"   Total: {total_experiments} experiments")
        print(f"   Success: {successful}/{total_experiments} ({summary['success_rate']:.1%})")
        print(f"   Duration: {summary['total_duration_hours']:.1f} hours")
    
    def run_all_experiments(self, selected_experiments=None):
        """Run the complete experiment suite."""
        experiments = {
            "seed_study": self.seed_study,
            "rotation_dims": self.rotation_dimension_ablation,
            "lr_ratios": self.learning_rate_ratio_study,
            "expert_scales": self.expert_scale_study,
            "loss_weights": self.loss_weight_ablation,
            "baselines": self.baseline_comparison,
            "datasets": self.multi_dataset_study
        }
        
        if selected_experiments:
            experiments = {k: v for k, v in experiments.items() if k in selected_experiments}
        
        print("🧪 GEOMETRIC CONSTRAINED LEARNING - PAPER EXPERIMENT SUITE")
        print("="*70)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Output Directory: {self.base_output_dir}")
        print(f"🎯 Running {len(experiments)} experiment types")
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No experiments will actually execute")
        
        estimated_total = len(experiments) * 15 * 60  # Rough estimate: 15 min per experiment type
        print(f"⏱️  Estimated completion: {(datetime.now() + timedelta(seconds=estimated_total)).strftime('%H:%M')}")
        print("="*70)
        
        all_results = []
        
        for exp_name, exp_func in experiments.items():
            try:
                results = exp_func()
                all_results.extend(results)
            except KeyboardInterrupt:
                print("\n⚠️  Experiment suite interrupted by user")
                break
            except Exception as e:
                print(f"\n💥 Error in {exp_name}: {e}")
                continue
        
        # Generate summary
        self.generate_summary_report(all_results)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Automated experiment suite for Geometric Constrained Learning paper")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    parser.add_argument("--output-dir", default="paper_experiments", help="Output directory for results")
    parser.add_argument("--experiments", help="Comma-separated list of experiments to run (default: all)")
    
    args = parser.parse_args()
    
    # Parse selected experiments
    selected_experiments = None
    if args.experiments:
        selected_experiments = [exp.strip() for exp in args.experiments.split(",")]
    
    # Create and run experiment suite
    suite = PaperExperimentSuite(args.output_dir, args.dry_run)
    
    try:
        results = suite.run_all_experiments(selected_experiments)
        print(f"\n✅ Experiment suite completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Experiment suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
