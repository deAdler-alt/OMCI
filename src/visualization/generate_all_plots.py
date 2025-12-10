"""
WBAN Optimization - Batch Plot Generator
Author: Kamil Piejko
Date: 2024

Automatyczne generowanie wszystkich wykresów dla pracy dyplomowej.
"""

import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_all_plots(
    main_csv: str = 'results/main_scenarios/all_main_scenarios.csv',
    power_csv: str = 'results/power_analysis/all_power_analysis.csv',
    config_path: str = 'config/wban_params.yaml',
    output_dir: str = 'results/plots',
    scenarios: list = ['S1', 'S2', 'S3'],
    weight_variants: list = ['balanced']
):
    """
    Generuje wszystkie wykresy dla pracy dyplomowej.
    
    Args:
        main_csv: Ścieżka do CSV z głównymi scenariuszami
        power_csv: Ścieżka do CSV z analizą mocy
        config_path: Ścieżka do config
        output_dir: Katalog wynikowy
        scenarios: Lista scenariuszy
        weight_variants: Lista wariantów wag
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.visualization.plot_energy import plot_energy_vs_sensors
    from src.visualization.plot_lifetime import plot_lifetime_comparison
    from src.visualization.plot_convergence import plot_convergence_curves
    from src.visualization.plot_placement import plot_best_placements_from_csv
    from src.visualization.plot_power_sensitivity import (
        plot_power_sensitivity, plot_power_correlation
    )
    from src.visualization.plot_comparison import (
        plot_algorithm_comparison_table, plot_radar_comparison
    )
    
    logger.info("="*60)
    logger.info("GENERATING ALL PLOTS")
    logger.info("="*60)
    
    start_time = datetime.now()
    plot_count = 0
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Energy plots
        logger.info("\n[1/7] Generating energy plots...")
        for weight in weight_variants:
            plot_energy_vs_sensors(
                csv_path=main_csv,
                output_path=f'{output_dir}/energy_vs_sensors_{weight}.png',
                weight_variant=weight
            )
            plot_count += 1
        
        # 2. Lifetime plots
        logger.info("\n[2/7] Generating lifetime plots...")
        plot_lifetime_comparison(
            csv_path=main_csv,
            output_path=f'{output_dir}/lifetime_comparison.png'
        )
        plot_count += 1
        
        # 3. Convergence plots
        logger.info("\n[3/7] Generating convergence plots...")
        for scenario in scenarios:
            for weight in weight_variants:
                plot_convergence_curves(
                    csv_path=main_csv,
                    output_path=f'{output_dir}/convergence_{scenario}_{weight}.png',
                    scenario=scenario,
                    weight_variant=weight
                )
                plot_count += 1
        
        # 4. Placement plots
        logger.info("\n[4/7] Generating placement plots...")
        for weight in weight_variants:
            plot_best_placements_from_csv(
                csv_path=main_csv,
                config_path=config_path,
                output_dir=f'{output_dir}/placements',
                scenarios=scenarios,
                algorithms=['GA', 'PSO'],
                weight_variant=weight
            )
            plot_count += len(scenarios) * 2  # scenarios × algorithms
        
        # 5. Power sensitivity
        logger.info("\n[5/7] Generating power sensitivity plots...")
        plot_power_sensitivity(
            csv_path=power_csv,
            output_path=f'{output_dir}/power_sensitivity.png'
        )
        plot_count += 1
        
        plot_power_correlation(
            csv_path=power_csv,
            output_path=f'{output_dir}/power_correlation.png'
        )
        plot_count += 1
        
        # 6. Comparison tables
        logger.info("\n[6/7] Generating comparison tables...")
        for scenario in scenarios:
            for weight in weight_variants:
                plot_algorithm_comparison_table(
                    csv_path=main_csv,
                    output_path=f'{output_dir}/comparison_table_{scenario}_{weight}.png',
                    scenario=scenario,
                    weight_variant=weight
                )
                plot_count += 1
        
        # 7. Radar plots
        logger.info("\n[7/7] Generating radar plots...")
        for scenario in scenarios:
            for weight in weight_variants:
                plot_radar_comparison(
                    csv_path=main_csv,
                    output_path=f'{output_dir}/comparison_radar_{scenario}_{weight}.png',
                    scenario=scenario,
                    weight_variant=weight
                )
                plot_count += 1
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure to run experiments first!")
        return False
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "="*60)
    logger.info("PLOT GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total plots: {plot_count}")
    logger.info(f"Time elapsed: {elapsed:.2f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    return True


def generate_summary_report(output_dir: str = 'results/plots'):
    """
    Generuje raport HTML z wszystkimi wykresami.
    
    Args:
        output_dir: Katalog z wykresami
    """
    logger.info("Generating HTML summary report...")
    
    plots_dir = Path(output_dir)
    
    # Znajdź wszystkie PNG
    plot_files = sorted(plots_dir.rglob('*.png'))
    
    if not plot_files:
        logger.warning("No plots found!")
        return
    
    # Wygeneruj HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>WBAN Optimization - Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #2E86AB; }
        h2 { color: #333; margin-top: 40px; }
        .plot { margin: 20px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .metadata { color: #666; font-size: 0.9em; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>WBAN Sensor Placement Optimization - Results</h1>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <p>Total plots: """ + str(len(plot_files)) + """</p>
    
    <h2>1. Energy Analysis</h2>
"""
    
    # Grupuj wykresy
    categories = {
        'Energy Analysis': ['energy'],
        'Network Lifetime': ['lifetime'],
        'Convergence': ['convergence'],
        'Sensor Placement': ['placement'],
        'Power Sensitivity': ['power'],
        'Algorithm Comparison': ['comparison']
    }
    
    for category, keywords in categories.items():
        html += f"<h2>{category}</h2>\n"
        
        for plot_file in plot_files:
            if any(kw in plot_file.stem.lower() for kw in keywords):
                rel_path = plot_file.relative_to(plots_dir)
                html += f"""
    <div class="plot">
        <h3>{plot_file.stem.replace('_', ' ').title()}</h3>
        <img src="{rel_path}" alt="{plot_file.stem}">
        <div class="metadata">File: {rel_path}</div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Zapisz HTML
    report_file = plots_dir / 'summary_report.html'
    with open(report_file, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {report_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all plots for WBAN optimization')
    parser.add_argument('--main-csv', type=str,
                       default='results/main_scenarios/all_main_scenarios.csv',
                       help='Path to main scenarios CSV')
    parser.add_argument('--power-csv', type=str,
                       default='results/power_analysis/all_power_analysis.csv',
                       help='Path to power analysis CSV')
    parser.add_argument('--config', type=str,
                       default='config/wban_params.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str,
                       default='results/plots',
                       help='Output directory')
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML summary report')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output) / f'plot_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Generuj wykresy
    success = generate_all_plots(
        main_csv=args.main_csv,
        power_csv=args.power_csv,
        config_path=args.config,
        output_dir=args.output
    )
    
    if success and args.report:
        generate_summary_report(args.output)
    
    if success:
        print("\n✓ All plots generated successfully!")
        print(f"Check: {args.output}/")
    else:
        print("\n✗ Plot generation failed. Check logs.")
