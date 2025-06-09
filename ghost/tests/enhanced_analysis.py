import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil

class SweepAnalyzer:
    def __init__(self, sweep_dir):
        self.sweep_dir = sweep_dir
        self.log_files = self._find_log_files()
        if not self.log_files:
            print("No training_log.json files found.")
            self.df = pd.DataFrame()
        else:
            self.df = self._load_log_data()

    def _find_log_files(self):
        log_files = []
        for dirpath, _, filenames in os.walk(self.sweep_dir):
            if "training_log.json" in filenames:
                log_files.append(os.path.join(dirpath, "training_log.json"))
        return log_files

    def _load_log_data(self):
        all_data = []
        for log_file in self.log_files:
            run_name = os.path.basename(os.path.dirname(log_file))
            with open(log_file, 'r') as f:
                data = json.load(f)
            for entry in data:
                entry['run_name'] = run_name
                all_data.append(entry)
        return pd.DataFrame(all_data)

    def analyze(self):
        if self.df.empty:
            return

        print("Generating individual run plots...")
        for run_name in self.df['run_name'].unique():
            self._plot_individual_run(run_name)

        print("Generating comparison plots...")
        self._plot_comparison()

        print("Generating performance summary...")
        summary_df = self._generate_summary()
        
        print("Finding best model and saving checkpoint...")
        self._save_best_model(summary_df)

        print("\nAnalysis Complete.")
        print(f"Results saved in: {self.sweep_dir}")

    def _plot_individual_run(self, run_name):
        run_df = self.df[self.df['run_name'] == run_name].sort_values('step')
        output_dir = os.path.dirname(self.log_files[0]) # Simplified for example
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Analysis for {run_name}', fontsize=16)

        sns.lineplot(data=run_df, x='step', y='eval_loss', ax=axes[0], marker='o', label='Eval Loss')
        axes[0].set_title('Evaluation Loss')
        axes[0].grid(True)

        if 'ghost_activations' in run_df.columns:
            run_df['avg_ghost_activation'] = run_df['ghost_activations'].apply(lambda x: sum(x) / len(x) if x else 0)
            sns.lineplot(data=run_df, x='step', y='avg_ghost_activation', ax=axes[1], label='Avg Ghost Activation')
            axes[1].set_title('Average Ghost Activation')
            axes[1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(os.path.dirname(self.log_files[0]), f'{run_name}_analysis.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def _plot_comparison(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sns.lineplot(data=self.df, x='step', y='eval_loss', hue='run_name', ax=ax, marker='o')
        ax.set_title('Sweep Comparison: Evaluation Loss')
        ax.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sweep_dir, 'comparison_eval_loss.png'))
        plt.close(fig)

    def _generate_summary(self):
        summary_data = []
        for run_name in self.df['run_name'].unique():
            run_df = self.df[self.df['run_name'] == run_name]
            final_metrics = run_df.iloc[-1]
            summary_data.append({
                'run_name': run_name,
                'final_eval_loss': final_metrics['eval_loss'],
                'min_eval_loss': run_df['eval_loss'].min(),
                'final_train_loss': final_metrics['train_loss'],
            })
        summary_df = pd.DataFrame(summary_data).sort_values(by='min_eval_loss', ascending=True)
        summary_df.to_csv(os.path.join(self.sweep_dir, 'sweep_summary.csv'), index=False)
        print("\nPerformance Summary:")
        print(summary_df)
        return summary_df

    def _save_best_model(self, summary_df):
        if summary_df.empty:
            return
        best_run_name = summary_df.iloc[0]['run_name']
        best_run_dir = os.path.join(self.sweep_dir, best_run_name)
        
        best_model_src = os.path.join(best_run_dir, 'best_model.pt')
        best_model_dest_dir = os.path.join(self.sweep_dir, 'best_model')
        
        if os.path.exists(best_model_src):
            os.makedirs(best_model_dest_dir, exist_ok=True)
            shutil.copy(best_model_src, os.path.join(best_model_dest_dir, 'best_model.pt'))
            
            # Save metadata
            best_run_info = summary_df.iloc[0].to_dict()
            with open(os.path.join(best_model_dest_dir, 'best_model_metadata.json'), 'w') as f:
                json.dump(best_run_info, f, indent=4)
            
            print(f"\n🏆 Best model checkpoint from run '{best_run_name}' saved to {best_model_dest_dir}")
        else:
            print(f"\n⚠️ Could not find 'best_model.pt' for the best run '{best_run_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced analysis for hyperparameter sweeps.")
    parser.add_argument("sweep_dir", type=str, help="The root directory of the sweep to analyze.")
    args = parser.parse_args()

    analyzer = SweepAnalyzer(args.sweep_dir)
    analyzer.analyze()

if __name__ == "__main__":
    main()
