# PIV Flow Analyzer — CISC108 Final Project

A Drafter website that takes particle image velocimetry (PIV) CSV files and helps you examine fluid flow. It calculates speed for every velocity vector, displays three Matplotlib graphs, stores multiple experiments in a `State` dataclass, compares two experiments, and exports results.

The website is based on the included original `basic_stats.py`. Its original calculations and plotting functions have been corrected and integrated into the site.

## Files

| File | What it is |
| --- | --- |
| `piv_flow_analyzer.py` | **Main website. Run this file.** |
| `test_piv_flow_analyzer.py` | 29 tests covering calculations and every route |
| `sample_straight_flow.csv` | Synthetic 7 × 7 mostly horizontal flow, 49 vectors |
| `sample_vortex_flow.csv` | Synthetic 7 × 7 rotating flow, 49 vectors |
| `original_basic_stats.py` | Your original code, kept for reference |
| `requirements.txt` | Drafter and Matplotlib dependencies |

## How to run

1. Install **Python 3.10 or newer**. If you use Thonny, install packages using **Tools → Manage packages**.
2. Install the dependencies in the Python environment that runs the project:

   ```bash
   python -m pip install -r requirements.txt
   ```

   In Thonny, you can alternatively install `drafter` and `matplotlib` individually through its package manager.

3. From the project folder, run:

   ```bash
   python piv_flow_analyzer.py
   ```

   Or open `piv_flow_analyzer.py` in Thonny and press **Run**. Drafter opens its website in a browser; the typical local address is `http://localhost:8000`.

4. Click **Load two sample experiments** to try the features instantly, or upload `sample_straight_flow.csv` and `sample_vortex_flow.csv` through the Upload page.

5. On the Visualization page, choose Vector Field, Speed Map, or Colored Vectors. On Settings, adjust the speed threshold or normalize arrows. The comparison page uses the saved graph mode and settings.

6. Visit Export to save a statistics summary and processed vectors. The comparison page also provides a comparison CSV.

**Important:** Drafter keeps your uploaded experiments in browser state, so they normally disappear if the page is reloaded or the browser tab is closed. Download the processed CSV files to preserve your work. The synthetic samples are demonstrations, not real laboratory data.

## How to run the tests

```bash
python -m unittest -v test_piv_flow_analyzer.py
```

The tests use Drafter's `assert_has` and `assert_state` helpers for pages and normal Python `unittest` assertions for numerical logic. They use fresh State instances instead of reusing application state. Each route has at least one test (including the form-cancel routes).

**Testing status:** All 29 tests passed in the preparation environment with a **local lightweight Drafter compatibility shim** and actual Matplotlib image rendering. Syntax checks passed. Real Drafter was not installed in that environment, so **browser interactions and real Drafter component integration still need testing on your machine**. The code uses the component signatures from the official Drafter documentation.

## Test the actual website manually before submitting

- [ ] Start the real Drafter website and verify that Home loads.
- [ ] Click **Load two sample experiments**, then see both listed on Datasets.
- [ ] Open Results; confirm the mean/max values appear.
- [ ] Open Visualize and switch among all three charts.
- [ ] Save a threshold and turn normalized arrows on; revisit Visualize and confirm the arrows change.
- [ ] Open Compare, switch the two experiments and update the comparison.
- [ ] Attempt to compare an experiment against itself; verify the feedback message.
- [ ] Upload both provided CSV files, one at a time, giving them different names.
- [ ] Try an invalid CSV file and check that nothing is added.
- [ ] Delete a dataset and confirm the selected experiment still works.
- [ ] Download summary and processed CSV files and verify them in Excel or a text editor.
- [ ] Click **Cancel and return home** from Upload, and the return links on Visualization, Settings, and Compare.
- [ ] Run all tests with the real Drafter library; fix any local compatibility issues before submitting.

## Grading rubric coverage

| Requirement | Where to find it |
| --- | --- |
| 7+ routes | 19 functions decorated with `@route` in `piv_flow_analyzer.py` |
| 5+ pages | Home, Upload, Datasets, Results, Visualizations, Comparison, Settings, Export |
| `State` with 4+ fields | `State` dataclass has 7 fields |
| Second dataclass | `PIVData` stores arrays, speeds, name, description |
| List of dataclasses | `State.datasets: list[PIVData]` |
| 4+ State fields changed in routes | `process_upload`, `load_demo`, `select_dataset`, `delete_dataset`, `compare_datasets`, `set_visualization`, `save_settings` mutate all 7 fields in meaningful ways |
| 3+ approved input types | TextBox + TextArea on Upload, SelectBox on Visualizations/Comparison, CheckBox on Settings |
| 3+ meaningful if statements | CSV validation, empty dataset handling, selection validation, threshold filtering, same-dataset comparison, settings validation |
| Loop over a dataclass list | `datasets_page` loops directly through `state.datasets` and reads names, speeds, descriptions |
| Legitimate purpose | Upload, calculate, visualize, compare and export PIV experiment measurements |
| No global mutable state | Experiment data is in `State`; the only globals are fixed demonstration strings and the immutable `PLOT_MODES` tuple |
| Drafter required | `@route`, `Page`, components and `start_server` in the main source |
| Advanced techniques documented | Comments/docstrings explain Matplotlib axes/subplots, CSV validation, state index rules, and optional uploads |
| Type hints/docstrings/line lengths | Functions annotated and documented; main Python code has no lines longer than 120 characters |
| Route unit tests | `TestDrafterRoutes` contains one or more tests for all 19 routes |

## What the calculations mean

- Each measurement gives position `(x, y)` and velocity components `(u, v)`.
- The velocity magnitude is `sqrt(u**2 + v**2)`.
- Mean U, mean V, mean speed, minimum speed and maximum speed are calculated from the full, original experiment.
- A threshold only changes which measurements are **displayed** on a graph; it does not remove rows or alter exported statistics.
- Normalization changes the arrow components to `(u/speed, v/speed)` for nonzero speed, emphasizing direction rather than relative magnitudes. Stationary vectors stay `(0, 0)`.
- The comparison graphs share a color range when using color-coded modes. Arrows are scaled for readable display, so you should compare the statistics for exact magnitudes.

## Source references

Drafter official docs: https://drafter-edu.github.io/drafter/

- Files: https://drafter-edu.github.io/drafter/add/files/
- Matplotlib plots: https://drafter-edu.github.io/drafter/reference/components/media/matplotlibplot/
- Forms: https://drafter-edu.github.io/drafter/concepts/forms-and-input/
- Testing: https://drafter-edu.github.io/drafter/reference/testing-functions/
