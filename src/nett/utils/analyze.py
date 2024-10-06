import subprocess
from pathlib import Path
from typing import Optional

# TODO: make this callable without needing to specify params 
# and run automatically when nett.run() completes


def analyze(
          config: str,
                run_dir: str | Path,
                output_dir: Optional[str | Path] = None,
                ep_bucket: int = 100,
                num_episodes: int = 1000,
                bar_order: str | list[int] = "default",
                color_bars: bool = True) -> None:
   
        # TODO may need to clean up this file structure
        # set paths
        run_dir = Path(run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

        analysis_dir = Path(__file__).resolve().parent.joinpath("analysis")
        if output_dir is None:
            output_dir = run_dir.joinpath("results")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        chick_data_dir = Path(analysis_dir).joinpath("ChickData", f"{config.lower()}.csv")

        if not chick_data_dir.exists():
            raise ValueError(f"'{config}' is not a valid config.")
        elif not run_dir.exists():
            raise ValueError(f"'{run_dir}' is not a valid run directory.")
        elif not analysis_dir.exists():
            raise ValueError(f"'{analysis_dir}' is not a valid analysis directory. This is likely an error in the package.")

        # translate bar_order for R to read
        bar_order_str = str(bar_order).translate({ord(i): None for i in ' []'}) # remove spaces and brackets from bar_order

        # merge
        print("Running merge")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_merge_csvs.R")),
                        "--logs-dir", str(run_dir),
                        "--results-dir", str(output_dir),
                        "--results-name", "analysis_data",
                        "--csv-train", "train_results.csv",
                        "--csv-test", "test_results.csv"], check=True)

        # train
        print("Running analysis for [train]")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_train_viz.R")),
                        "--data-loc", str(output_dir.joinpath("analysis_data")),
                        "--results-wd", str(output_dir),
                        "--ep-bucket", str(ep_bucket),
                        "--num-episodes", str(num_episodes)], check=True)

        # test
        print("Running analysis for [test]")
        subprocess.run(["Rscript", str(analysis_dir.joinpath("NETT_test_viz.R")),
                        "--data-loc", str(output_dir.joinpath("analysis_data")),
                        "--results-wd", str(output_dir),
                        "--bar-order", bar_order_str,
                        "--color-bars", str(color_bars),
                        "--chick-file", str(chick_data_dir)], check=True)

        print(f"Analysis complete. See results at {output_dir}")
