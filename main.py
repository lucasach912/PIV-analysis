"""PIV Flow Analyzer: CISC108 Drafter website for comparing fluid flows.

This project builds on the original basic_stats.py calculations. It uses only
Drafter, Matplotlib, and Python's standard library.
"""
set_site_information(
    author="Lucas Achilles",
    title="PIV Analysis Website",
    description="A website for analyzing and visualizing Particle Image Velocimetry (PIV) data.",
    ai="ChatGPT",
    framework="Drafter"
)
from dataclasses import dataclass
import csv
from io import StringIO
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.tri import Triangulation
from drafter import *


# These are constants; datasets and user preferences live in State, not here.
PLOT_MODES = ("Velocity vectors", "Speed map", "Colored vectors",
              "Speed histogram", "Velocity heatmap")
MAX_CSV_CHARACTERS = 1_500_000
MAX_VECTORS = 15_000
DEMO_STRAIGHT = """x,y,u,v
0,0,1.0,0.15
1,0,1.4,0.15
2,0,1.7,0.15
0,1,1.1,0.05
1,1,1.6,0.05
2,1,1.9,0.05
0,2,0.9,-0.1
1,2,1.3,-0.1
2,2,1.6,-0.1
"""
DEMO_VORTEX = """x,y,u,v
0,0,1.0,-1.0
1,0,1.0,0.0
2,0,1.0,1.0
0,1,0.0,-1.0
1,1,0.0,0.0
2,1,0.0,1.0
0,2,-1.0,-1.0
1,2,-1.0,0.0
2,2,-1.0,1.0
"""


@dataclass
class PIVData:
    """Store one experiment's positions, velocity components, and metadata."""
    x: list[float]
    y: list[float]
    u: list[float]
    v: list[float]
    speeds: list[float]
    name: str = "Untitled experiment"
    description: str = ""


@dataclass
class State:
    """Store the user's experiments and chosen settings between routes."""
    datasets: list[PIVData]
    active_index: int
    compare_index: int
    plot_mode: str
    minimum_speed: float
    normalize_arrows: bool
    status_message: str
    arrow_density: int
    color_max_speed: float  # 0 means automatic color scale.
    show_grid: bool
    position_units: str
    velocity_units: str
    stationary_cutoff: float


def fresh_state() -> State:
    """Create independent initial state, including an empty dataset list."""
    return State([], -1, -1, PLOT_MODES[0], 0.0, False, "", 1, 0.0,
                 True, "mm", "m/s", 0.1)


# ----------------------------- Calculations -----------------------------

def calculate_speeds(u_values: list[float], v_values: list[float]) -> list[float]:
    """Calculate velocity magnitude for each matching U and V component."""
    if len(u_values) != len(v_values):
        raise ValueError("U and V must have the same number of values.")
    speeds = []
    for index in range(len(u_values)):
        try:
            speed = (u_values[index] ** 2 + v_values[index] ** 2) ** 0.5
        except OverflowError:
            raise ValueError("Velocity magnitude is too large to calculate.")
        if not math.isfinite(speed):
            raise ValueError("Velocity magnitude is too large to calculate.")
        speeds.append(speed)
    return speeds


def mean_value(values: list[float]) -> float:
    """Calculate the mean using a basic loop; reject empty lists."""
    if not values:
        raise ValueError("Cannot calculate the mean of an empty list.")
    total = 0.0
    for number in values:
        total += number
    return total / len(values)


def max_speed(speeds: list[float]) -> float:
    """Find the maximum in a nonempty list using a loop."""
    if not speeds:
        raise ValueError("Cannot calculate maximum speed with no measurements.")
    greatest = speeds[0]
    for speed in speeds:
        if speed > greatest:
            greatest = speed
    return greatest


def speed_std_dev(speeds: list[float]) -> float:
    """Calculate population standard deviation: square root of mean squared error."""
    average = mean_value(speeds)
    total = 0.0
    for speed in speeds:
        total += (speed - average) ** 2
    return (total / len(speeds)) ** 0.5


def mean_velocity_magnitude(data: PIVData) -> float:
    """Find magnitude of mean vector, not the same as mean of speeds."""
    mean_u = mean_value(data.u)
    mean_v = mean_value(data.v)
    return (mean_u ** 2 + mean_v ** 2) ** 0.5


def stationary_percentage(data: PIVData, cutoff: float) -> float:
    """Find percent of vector speeds strictly below a user-defined cutoff."""
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("Stationary cutoff must be nonnegative and finite.")
    stationary_count = 0
    for speed in data.speeds:
        if speed < cutoff:
            stationary_count += 1
    return stationary_count * 100.0 / len(data.speeds)


def percentage_change(before: float, after: float) -> float | None:
    """Return the relative percent change; None when the baseline is zero."""
    if before == 0:
        return None
    return (after - before) * 100.0 / before


def display_results(data: PIVData, velocity_units: str = "", cutoff: float = 0.1) -> list[str]:
    """Make labeled statistics text for the selected experiment."""
    suffix = " " + velocity_units if velocity_units else ""
    return [
        "Number of vectors: " + str(len(data.speeds)),
        "Mean U: " + format(mean_value(data.u), ".3f") + suffix,
        "Mean V: " + format(mean_value(data.v), ".3f") + suffix,
        "Mean speed: " + format(mean_value(data.speeds), ".3f") + suffix,
        "Magnitude of mean velocity: " + format(mean_velocity_magnitude(data), ".3f") + suffix,
        "Maximum speed: " + format(max_speed(data.speeds), ".3f") + suffix,
        "Minimum speed: " + format(min(data.speeds), ".3f") + suffix,
        "Speed standard deviation: " + format(speed_std_dev(data.speeds), ".3f") + suffix,
        "Nearly stationary (< " + str(cutoff) + suffix + "): " +
        format(stationary_percentage(data, cutoff), ".1f") + "%",
    ]


# ----------------------------- CSV parsing -----------------------------

def read_csv(piv: str) -> PIVData:
    """Read a CSV upload with required x,y,u,v columns and safe row errors.

    csv.reader is from Python's standard library. Unlike splitting on commas,
    it correctly handles quoted CSV cells and whitespace around fields.
    """
    if not isinstance(piv, str) or not piv.strip():
        raise ValueError("The CSV file is empty.")
    if len(piv) > MAX_CSV_CHARACTERS:
        raise ValueError("This CSV is too large. Try a smaller export (under 1.5 million characters).")

    try:
        rows = csv.reader(StringIO(piv.lstrip("\ufeff")), skipinitialspace=True, strict=True)
        header = next(rows, None)
        while header is not None and not any(cell.strip() for cell in header):
            header = next(rows, None)
        if header is None:
            raise ValueError("The CSV file has no column headers.")
        columns = [cell.strip().lower().lstrip("\ufeff") for cell in header]
        if len(columns) != len(set(columns)):
            raise ValueError("CSV column headers must not be repeated.")
        for required in ("x", "y", "u", "v"):
            if required not in columns:
                raise ValueError("The CSV must have x, y, u, and v columns.")
        positions = [columns.index(name) for name in ("x", "y", "u", "v")]

        x_values, y_values, u_values, v_values = [], [], [], []
        for row in rows:
            row_number = rows.line_num
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) != len(columns):
                raise ValueError("Row " + str(row_number) + " has the wrong number of columns.")
            numbers = []
            try:
                for position in positions:
                    number = float(row[position].strip())
                    if not math.isfinite(number):
                        raise ValueError("non-finite number")
                    numbers.append(number)
            except ValueError:
                raise ValueError("Row " + str(row_number) + " has a missing or invalid number.")
            x_values.append(numbers[0])
            y_values.append(numbers[1])
            u_values.append(numbers[2])
            v_values.append(numbers[3])
            if len(x_values) > MAX_VECTORS:
                raise ValueError("The limit is " + str(MAX_VECTORS) + " velocity vectors per upload.")
    except csv.Error as error:
        raise ValueError("Invalid CSV formatting: " + str(error))

    if not x_values:
        raise ValueError("The CSV has headers but no velocity measurements.")
    try:
        speeds = calculate_speeds(u_values, v_values)
    except ValueError as error:
        raise ValueError("Unable to calculate speeds: " + str(error))
    return PIVData(x_values, y_values, u_values, v_values, speeds)


def current_dataset(state: State) -> PIVData | None:
    """Return the active experiment, or None if no valid selection exists."""
    if 0 <= state.active_index < len(state.datasets):
        return state.datasets[state.active_index]
    return None


def filtered_vectors(data: PIVData, threshold: float) -> PIVData:
    """Copy vectors at or above threshold without changing original measurements."""
    x_values, y_values, u_values, v_values, speeds = [], [], [], [], []
    for index in range(len(data.speeds)):
        if data.speeds[index] >= threshold:
            x_values.append(data.x[index])
            y_values.append(data.y[index])
            u_values.append(data.u[index])
            v_values.append(data.v[index])
            speeds.append(data.speeds[index])
    return PIVData(x_values, y_values, u_values, v_values, speeds, data.name, data.description)


def parse_arrow_spacing(text: str) -> int:
    """Read the user's arrow spacing as a positive whole number."""
    try:
        spacing = int(text.strip())
    except (ValueError, AttributeError):
        raise ValueError("Arrow spacing must be a whole number (1 or greater).")
    if spacing < 1:
        raise ValueError("Arrow spacing must be at least 1.")
    return spacing


def sample_vectors(data: PIVData, step: int) -> PIVData:
    """Keep every nth vector in plotting only, preserving original data."""
    if not isinstance(step, int) or step < 1:
        raise ValueError("Arrow spacing must be a positive whole number.")
    x_values, y_values, u_values, v_values, speeds = [], [], [], [], []
    for index in range(len(data.speeds)):
        if index % step == 0:
            x_values.append(data.x[index])
            y_values.append(data.y[index])
            u_values.append(data.u[index])
            v_values.append(data.v[index])
            speeds.append(data.speeds[index])
    return PIVData(x_values, y_values, u_values, v_values, speeds, data.name, data.description)


def arrow_components(data: PIVData, normalize: bool) -> tuple[list[float], list[float]]:
    """Make components for plotting, optionally normalizing nonzero vectors."""
    if not normalize:
        return data.u, data.v
    new_u, new_v = [], []
    for index in range(len(data.speeds)):
        if data.speeds[index] == 0:
            new_u.append(0.0)
            new_v.append(0.0)
        else:
            new_u.append(data.u[index] / data.speeds[index])
            new_v.append(data.v[index] / data.speeds[index])
    return new_u, new_v


def effective_color_max(state: State, first: PIVData, second: PIVData | None = None) -> float:
    """Choose one finite positive color scale for single or comparison plots."""
    if state.color_max_speed > 0:
        return state.color_max_speed
    maximum = max_speed(first.speeds)
    if second is not None:
        maximum = max(maximum, max_speed(second.speeds))
    return max(0.000001, maximum)


def spatial_limits(first: PIVData, second: PIVData | None = None) -> tuple[float, float, float, float]:
    """Fit the measured domain exactly, with no artificial margins.

    A constant X or Y value needs a minimal width to make a visible axis.
    Two-experiment comparisons still use the union of both experiments.
    """
    x_values = list(first.x)
    y_values = list(first.y)
    if second is not None:
        x_values.extend(second.x)
        y_values.extend(second.y)
    smallest_x, largest_x = min(x_values), max(x_values)
    smallest_y, largest_y = min(y_values), max(y_values)
    if smallest_x == largest_x:
        smallest_x -= 0.5
        largest_x += 0.5
    if smallest_y == largest_y:
        smallest_y -= 0.5
        largest_y += 0.5
    return (smallest_x, largest_x, smallest_y, largest_y)


def matching_speed_difference(first: PIVData, second: PIVData) -> PIVData | None:
    """Return B minus A speeds at matching unique coordinates, else None.

    A dictionary pairs each (x,y) position with its speed. The datasets may
    list points in different orders, but must cover exactly the same positions.
    """
    if len(first.x) != len(second.x):
        return None
    first_positions, second_positions = {}, {}
    for index in range(len(first.x)):
        location = (first.x[index], first.y[index])
        if location in first_positions:
            return None
        first_positions[location] = first.speeds[index]
    for index in range(len(second.x)):
        location = (second.x[index], second.y[index])
        if location in second_positions:
            return None
        second_positions[location] = second.speeds[index]
    if first_positions.keys() != second_positions.keys():
        return None
    differences = []
    for index in range(len(first.x)):
        location = (first.x[index], first.y[index])
        differences.append(second_positions[location] - first_positions[location])
    return PIVData(list(first.x), list(first.y), [], [], differences,
                   "Speed difference: B minus A")


def grid_heatmap(data: PIVData) -> tuple[list[float], list[float], list[list[float]]] | None:
    """Make a speed grid where absent measurement locations remain NaN (blank).

    PIV files are commonly exported on a Cartesian grid. Using a masked grid
    instead of ordinary triangulation preserves missing internal blocks.
    Very sparse or non-grid datasets are handled by masked triangulation.
    """
    xs = sorted(set(data.x))
    ys = sorted(set(data.y))
    if len(xs) < 2 or len(ys) < 2:
        return None
    possible_positions = len(xs) * len(ys)
    if possible_positions > 30000 or len(data.x) < possible_positions * 0.35:
        return None
    measured = {}
    for index in range(len(data.x)):
        position = (data.x[index], data.y[index])
        if position in measured:
            return None
        measured[position] = data.speeds[index]

    # A missing entire row or column would otherwise be interpolated across.
    # Put an all-missing row or column inside unusually large coordinate gaps.
    def insert_missing_axis_positions(coordinates: list[float]) -> list[float]:
        steps = [coordinates[i + 1] - coordinates[i] for i in range(len(coordinates) - 1)]
        typical = sorted(steps)[len(steps) // 2]
        if typical <= 0:
            return coordinates
        result = [coordinates[0]]
        for index, step in enumerate(steps):
            if step > 1.75 * typical:
                result.append((coordinates[index] + coordinates[index + 1]) / 2)
            result.append(coordinates[index + 1])
        return result

    xs = insert_missing_axis_positions(xs)
    ys = insert_missing_axis_positions(ys)
    if len(xs) * len(ys) > 32000:
        return None
    speeds = []
    for y in ys:
        speeds.append([measured.get((x, y), math.nan) for x in xs])
    return xs, ys, speeds


def masked_heatmap_triangulation(data: PIVData) -> Triangulation | None:
    """Do not draw contour triangles across large gaps in irregular PIV data.

    Each triangle is checked against local nearest-neighbor measurement
    distances. Long connections through unsampled space are masked.
    """
    triangulation = Triangulation(data.x, data.y)
    nearest = [math.inf] * len(data.x)
    for pair in triangulation.edges:
        first, second = int(pair[0]), int(pair[1])
        length = math.hypot(data.x[first] - data.x[second],
                            data.y[first] - data.y[second])
        nearest[first] = min(nearest[first], length)
        nearest[second] = min(nearest[second], length)

    masks = []
    for triangle in triangulation.triangles:
        corners = [int(value) for value in triangle]
        across_gap = False
        for index in range(3):
            first = corners[index]
            second = corners[(index + 1) % 3]
            length = math.hypot(data.x[first] - data.x[second],
                                data.y[first] - data.y[second])
            if length > 1.9 * max(nearest[first], nearest[second]):
                across_gap = True
        masks.append(across_gap)
    if not masks or all(masks):
        return None
    triangulation.set_mask(masks)
    return triangulation


def draw_heatmap(axis: Axes, data: PIVData, limit: float):
    """Draw a smooth speed field while leaving unmeasured regions white."""
    grid = grid_heatmap(data)
    if grid is not None:
        xs, ys, values = grid
        if min(data.speeds) == max(data.speeds):
            # A constant-speed field should still fill its measured cells.
            return axis.pcolormesh(xs, ys, values, cmap="turbo", vmin=0,
                                   vmax=limit, shading="nearest")
        try:
            return axis.contourf(xs, ys, values, levels=20, cmap="turbo",
                                 vmin=0, vmax=limit, corner_mask=False)
        except (ValueError, RuntimeError):
            pass
    if len(data.x) >= 3 and min(data.speeds) != max(data.speeds):
        try:
            triangulation = masked_heatmap_triangulation(data)
            if triangulation is not None:
                return axis.tricontourf(triangulation, data.speeds, levels=20,
                                        cmap="turbo", vmin=0, vmax=limit)
        except (ValueError, RuntimeError):
            pass
    # Sparse, collinear, or otherwise unsuitable points: show only measurements.
    return axis.scatter(data.x, data.y, c=data.speeds, cmap="turbo",
                        vmin=0, vmax=limit, s=55)


# ------------------------------- Plotting -------------------------------

def draw_velocity_plot(axis: Axes, data: PIVData, state: State, limit: float,
                       bounds: tuple[float, float, float, float] | None = None) -> None:
    """Draw arrows, points, a heatmap, or a histogram on a Matplotlib axis."""
    axis.set_title(data.name)
    if state.plot_mode == "Speed histogram":
        # The histogram always uses all original speeds, not only visible arrows.
        # Histogram X range follows measured speeds, not the colorbar setting.
        highest_speed = max(max_speed(data.speeds), 0.000001)
        if limit > highest_speed and state.plot_mode == "Speed histogram":
            # Shared comparison histograms use the common measured maximum.
            highest_speed = limit
        axis.hist(data.speeds, bins=12, range=(0, highest_speed),
                  color="#2a75bb", edgecolor="white")
        axis.set_xlabel("Speed (" + state.velocity_units + ")")
        axis.set_ylabel("Vector count")
        axis.set_xlim(0, highest_speed)
    else:
        visible = filtered_vectors(data, state.minimum_speed)
        if state.plot_mode in ("Velocity vectors", "Colored vectors"):
            visible = sample_vectors(visible, state.arrow_density)
        if not visible.speeds:
            axis.text(0.5, 0.5, "No vectors above threshold", ha="center",
                      va="center", transform=axis.transAxes)
        elif state.plot_mode == "Speed map":
            art = axis.scatter(visible.x, visible.y, c=visible.speeds, cmap="viridis",
                               vmin=0, vmax=limit, s=55, edgecolors="white", linewidths=0.3)
            axis.figure.colorbar(art, ax=axis, label="Speed (" + state.velocity_units + ")",
                                 shrink=0.80)
        elif state.plot_mode == "Velocity heatmap":
            art = draw_heatmap(axis, visible, limit)
            axis.figure.colorbar(art, ax=axis,
                                 label="Speed (" + state.velocity_units + ")",
                                 shrink=0.80)
        else:
            u_values, v_values = arrow_components(visible, state.normalize_arrows)
            if bounds is None:
                bounds = spatial_limits(data)
            # Match maximum arrow length to plot dimensions, not arbitrary units.
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
            longest_arrow = 1.0 if state.normalize_arrows else max_speed(visible.speeds)
            scale = max(longest_arrow / (0.09 * span), 0.000001)
            if state.plot_mode == "Colored vectors":
                art = axis.quiver(visible.x, visible.y, u_values, v_values, visible.speeds,
                                  cmap="viridis", angles="xy", scale_units="xy", scale=scale)
                art.set_clim(0, limit)
                axis.figure.colorbar(art, ax=axis, label="Speed (" + state.velocity_units + ")",
                                     shrink=0.80)
            else:
                axis.quiver(visible.x, visible.y, u_values, v_values, color="#2766a6",
                            angles="xy", scale_units="xy", scale=scale)
        axis.set_xlabel("X position (" + state.position_units + ")")
        axis.set_ylabel("Y position (" + state.position_units + ")")
        if bounds is None:
            bounds = spatial_limits(data)
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        # A line or a single point cannot have a meaningful equal-area display.
        if min(data.x) == max(data.x) or min(data.y) == max(data.y):
            axis.set_aspect("auto")
        else:
            axis.set_aspect("equal", adjustable="box")
    axis.grid(state.show_grid, alpha=0.25)


def make_single_plot(state: State, data: PIVData) -> None:
    """Draw one finished Matplotlib figure for Drafter to turn into a Picture."""
    fig, axis = plt.subplots(figsize=(9, 5.5))
    upper = (max(max_speed(data.speeds), 0.000001) if state.plot_mode == "Speed histogram"
             else effective_color_max(state, data))
    draw_velocity_plot(axis, data, state, upper)
    fig.suptitle("PIV Flow Analyzer | " + state.plot_mode)
    fig.tight_layout()


def make_comparison_plot(state: State, first: PIVData, second: PIVData) -> None:
    """Draw comparable axes with the same spatial and color limits."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    shared_limit = effective_color_max(state, first, second)
    if state.plot_mode == "Speed histogram":
        shared_limit = max(max_speed(first.speeds), max_speed(second.speeds))
        shared_limit = max(shared_limit, 0.000001)
    bounds = spatial_limits(first, second)
    for index, dataset in enumerate([first, second]):
        draw_velocity_plot(axes[index], dataset, state, shared_limit, bounds)
        axes[index].set_title(("A: " if index == 0 else "B: ") + dataset.name)
    if state.plot_mode == "Speed histogram":
        # Both distributions share the same bins and frequency range.
        maximum_count = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(0, maximum_count)
        axes[1].set_ylim(0, maximum_count)
    fig.suptitle("Flow comparison | " + state.plot_mode)
    fig.tight_layout()


def make_difference_plot(first: PIVData, second: PIVData, state: State) -> bool:
    """Draw spatial B-minus-A speed map when measurements share a coordinate grid."""
    difference = matching_speed_difference(first, second)
    if difference is None:
        return False
    fig, axis = plt.subplots(figsize=(8, 4.8))
    maximum = max(abs(min(difference.speeds)), abs(max(difference.speeds)))
    maximum = max(maximum, 0.000001)
    art = axis.scatter(difference.x, difference.y, c=difference.speeds,
                       cmap="coolwarm", vmin=-maximum, vmax=maximum, s=100)
    fig.colorbar(art, ax=axis, label="B minus A speed (" + state.velocity_units + ")")
    bounds = spatial_limits(first, second)
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    if min(difference.x) == max(difference.x) or min(difference.y) == max(difference.y):
        axis.set_aspect("auto")
    else:
        axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X position (" + state.position_units + ")")
    axis.set_ylabel("Y position (" + state.position_units + ")")
    axis.grid(state.show_grid, alpha=0.2)
    axis.set_title("Velocity magnitude difference at matching positions")
    fig.tight_layout()
    return True


def display_velocity_vectors(data: PIVData) -> None:
    """Preserve the original helper to draw a basic quiver plot."""
    state = fresh_state()
    make_single_plot(state, data)


def display_speed(data: PIVData) -> None:
    """Preserve the original helper to draw speed-coded scatter points."""
    state = fresh_state()
    state.plot_mode = "Speed map"
    make_single_plot(state, data)


def display_velocity_field(data: PIVData) -> None:
    """Preserve the original helper to draw color-coded velocity vectors."""
    state = fresh_state()
    state.plot_mode = "Colored vectors"
    make_single_plot(state, data)


# ------------------------------- CSV export ------------------------------

def csv_cell(value: str) -> str:
    """Escape a text cell to keep commas, quotes, and line breaks CSV-safe."""
    if any(mark in value for mark in (",", '"', "\r", "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def summary_csv(data: PIVData, cutoff: float = 0.1) -> str:
    """Build summary data with the new scientific statistics included."""
    names = ("experiment,vectors,mean_u,mean_v,mean_speed,mean_vector_magnitude,"
             "max_speed,min_speed,speed_std_dev,stationary_cutoff,stationary_percent")
    values = [csv_cell(data.name), len(data.speeds), mean_value(data.u), mean_value(data.v),
              mean_value(data.speeds), mean_velocity_magnitude(data), max_speed(data.speeds),
              min(data.speeds), speed_std_dev(data.speeds), cutoff,
              stationary_percentage(data, cutoff)]
    return names + "\n" + ",".join(str(item) for item in values) + "\n"


def processed_csv(data: PIVData) -> str:
    """Export original x,y,u,v values along with their calculated magnitudes."""
    contents = "x,y,u,v,speed\n"
    for index in range(len(data.speeds)):
        row = [data.x[index], data.y[index], data.u[index], data.v[index], data.speeds[index]]
        contents += ",".join(str(number) for number in row) + "\n"
    return contents


def comparison_csv(first: PIVData, second: PIVData) -> str:
    """Create comparison statistics CSV including percentage change when defined."""
    names = "statistic," + csv_cell(first.name) + "," + csv_cell(second.name) + "\n"
    rows = [
        "vectors," + str(len(first.speeds)) + "," + str(len(second.speeds)),
        "mean_u," + str(mean_value(first.u)) + "," + str(mean_value(second.u)),
        "mean_v," + str(mean_value(first.v)) + "," + str(mean_value(second.v)),
        "mean_speed," + str(mean_value(first.speeds)) + "," + str(mean_value(second.speeds)),
        "max_speed," + str(max_speed(first.speeds)) + "," + str(max_speed(second.speeds)),
        "speed_std_dev," + str(speed_std_dev(first.speeds)) + "," + str(speed_std_dev(second.speeds)),
    ]
    difference = mean_value(second.speeds) - mean_value(first.speeds)
    percentage = percentage_change(mean_value(first.speeds), mean_value(second.speeds))
    rows.append("mean_speed_difference_B_minus_A," + str(difference) + ",")
    rows.append("mean_speed_percent_change_B_from_A," +
                (str(percentage) if percentage is not None else "undefined") + ",")
    return names + "\n".join(rows) + "\n"


def safe_filename(name: str) -> str:
    """Create a short filename using letters, digits, dashes and underscores."""
    cleaned = ""
    for char in name.lower():
        cleaned += char if char.isascii() and (char.isalnum() or char in "-_") else "_"
    return cleaned.strip("_")[:45] or "experiment"


def choice_for_index(state: State, index: int) -> str:
    """Make an unambiguous SelectBox label for a particular dataset."""
    return str(index) + " | " + state.datasets[index].name


def index_from_choice(state: State, choice: str) -> int:
    """Find the dataset corresponding to a SelectBox's label."""
    for index in range(len(state.datasets)):
        if choice_for_index(state, index) == choice:
            return index
    return -1


# -------------------------------- UI layout -------------------------------

def card(*contents: object) -> Div:
    """Display a consistent light-blue panel for text or controls."""
    return Div(*contents, style_background_color="#f3f7fc",
               style_border="1px solid #d1dfed", style_border_radius="10px",
               style_padding="14px", style_margin="9px 0")


def navbar() -> Row:
    """Create a consistent site menu for pages without unsaved form inputs."""
    return Row(Link("Home", "index"), Link("Upload", "upload_page"),
               Link("Datasets", "datasets_page"), Link("Results", "results_page"),
               Link("Visualize", "visualization_page"), Link("Compare", "comparison_page"),
               Link("Settings", "settings_page"), Link("Export", "export_page"),
               style_gap="16px", style_flex_wrap="wrap", style_padding="12px",
               style_background_color="#eaf2ff", style_border_radius="8px")


def page_header(title: str, form_page: bool = False) -> list:
    """Return a reusable title and navigation for non-form pages."""
    contents = [Header("PIV Flow Analyzer"),
                Paragraph("Explore, compare, and export fluid velocity measurements.")]
    if not form_page:
        contents.append(navbar())
    contents.append(Header(title, 2))
    return contents


def take_message(state: State) -> list:
    """Show a temporary message once instead of repeating it on every page."""
    if state.status_message:
        text = state.status_message
        state.status_message = ""
        return [card(text)]
    return []


def plot_picture() -> Picture:
    """Capture the active Matplotlib plot as Drafter's downloadable Picture.

    MatPlotLibPlot.to_picture is documented by Drafter. The resulting Picture
    can be used both by Image for the preview and Download for saving a PNG.
    """
    return MatPlotLibPlot().to_picture()


def no_experiment_page(state: State, title: str) -> Page:
    """Display an appropriate empty state instead of trying to plot no data."""
    contents = page_header(title)
    contents += take_message(state)
    contents += [card("Upload or select an experiment first."), Link("Upload a CSV", "upload_page")]
    return Page(state, contents)


# --------------------------------- Routes ---------------------------------

@route
def index(state: State) -> Page:
    """Show experiment counts, key metrics, preview plot, and next actions."""
    contents = page_header("Analysis dashboard")
    contents += take_message(state)
    contents.append(card(Header("Experiments", 3),
                         str(len(state.datasets)) + " uploaded dataset(s)"))
    selected = current_dataset(state)
    if selected is None:
        contents += [card("No experiment selected. Upload a CSV or load the two demos."),
                     Paragraph("Upload an x,y,u,v CSV to analyze particle-image velocimetry data.")]
    else:
        contents.append(card(Header("Currently selected: " + selected.name, 3),
                             "Vectors: " + str(len(selected.speeds)),
                             Paragraph("Mean speed: " + format(mean_value(selected.speeds), ".3f") +
                                       " " + state.velocity_units),
                             Paragraph("Maximum speed: " + format(max_speed(selected.speeds), ".3f") +
                                       " " + state.velocity_units)))
        # Only 9–15k arrows max; the density setting still affects this preview.
        make_single_plot(state, selected)
        contents.append(Image(plot_picture(), alt="Preview of the selected experiment's flow"))
        contents.append(Row(Link("View full results", "results_page"),
                            Link("Compare datasets", "comparison_page"),
                            Link("Open visualizations", "visualization_page"),
                            style_gap="14px", style_flex_wrap="wrap"))
    contents.append(Button("Load two sample experiments", "load_demo"))
    contents.append(Paragraph("Your uploads stay in this browser session. Export them before leaving."))
    return Page(state, contents)


@route
def load_demo(state: State) -> Page:
    """Add sample experiments only when their demo names are absent."""
    examples = [("Straight flow (demo)", DEMO_STRAIGHT,
                 "Predominantly horizontal flow with modest vertical velocity."),
                ("Rotating flow (demo)", DEMO_VORTEX,
                 "Circular velocity around a stationary center.")]
    added = 0
    for name, text, description in examples:
        found = False
        for existing in state.datasets:
            if existing.name == name:
                found = True
        if not found:
            dataset = read_csv(text)
            dataset.name = name
            dataset.description = description
            state.datasets.append(dataset)
            added += 1
    for index in range(len(state.datasets)):
        if state.datasets[index].name == "Straight flow (demo)":
            state.active_index = index
        if state.datasets[index].name == "Rotating flow (demo)":
            state.compare_index = index
    state.status_message = ("Loaded " + str(added) + " sample experiment(s)." if added
                            else "Both sample experiments are already loaded.")
    return datasets_page(state)


@route
def upload_page(state: State) -> Page:
    """Provide an experiment name, description, and CSV file picker."""
    contents = page_header("Upload an experiment", True)
    contents += take_message(state)
    contents += [Paragraph("CSV must contain x,y,u,v columns (additional columns are allowed)."),
                 card("Experiment name", TextBox("experiment_name", "My experiment"),
                      Paragraph("Experiment notes (optional)"), TextArea("description", ""),
                      Paragraph("Choose a CSV file"), FileUpload("piv_file", accept=".csv"),
                      Button("Analyze CSV", "process_upload")),
                 Link("Cancel upload", "leave_upload")]
    return Page(state, contents)


@route
def leave_upload(state: State, experiment_name: str, description: str,
                 piv_file: str | None) -> Page:
    """Leave the upload form without adding a dataset."""
    return index(state)


@route
def process_upload(state: State, experiment_name: str, description: str,
                   piv_file: str | None) -> Page:
    """Validate input and append a new experiment only after successful parsing."""
    if not experiment_name.strip():
        state.status_message = "Enter a name for this experiment."
        return upload_page(state)
    if piv_file is None or not piv_file.strip():
        state.status_message = "Choose a CSV file before uploading."
        return upload_page(state)
    try:
        dataset = read_csv(piv_file)
    except ValueError as error:
        state.status_message = "CSV upload failed: " + str(error)
        return upload_page(state)
    dataset.name = experiment_name.strip()[:90]
    dataset.description = description.strip()[:400]
    state.datasets.append(dataset)
    state.active_index = len(state.datasets) - 1
    if len(state.datasets) >= 2 and state.compare_index == -1:
        state.compare_index = 0 if state.active_index != 0 else 1
    state.status_message = ("Uploaded " + dataset.name + " — " + str(len(dataset.speeds)) +
                            " vectors. Speed range: " + format(min(dataset.speeds), ".3f") +
                            " to " + format(max_speed(dataset.speeds), ".3f") +
                            " " + state.velocity_units + ".")
    return results_page(state)


@route
def datasets_page(state: State) -> Page:
    """List PIVData instances from State and provide selection and deletion."""
    contents = page_header("Manage experiments")
    contents += take_message(state)
    if not state.datasets:
        contents.append(card("No experiments uploaded yet."))
    for index, dataset in enumerate(state.datasets):
        selected = " (selected)" if index == state.active_index else ""
        contents.append(card(Header(dataset.name + selected, 3),
                             Paragraph("Vectors: " + str(len(dataset.speeds))),
                             Paragraph("Mean speed: " + format(mean_value(dataset.speeds), ".3f") +
                                       " " + state.velocity_units),
                             Paragraph(dataset.description or "No experiment notes"),
                             Row(Button("View results", "select_dataset",
                                        [Argument("dataset_index", index)]),
                                 Button("Delete", "delete_dataset",
                                        [Argument("dataset_index", index)]),
                                 style_gap="16px")))
    contents.append(Link("Upload another experiment", "upload_page"))
    return Page(state, contents)


@route
def select_dataset(state: State, dataset_index: int) -> Page:
    """Change active_index and navigate to the newly selected results."""
    if 0 <= dataset_index < len(state.datasets):
        state.active_index = dataset_index
        state.status_message = "Selected " + state.datasets[dataset_index].name + "."
        return results_page(state)
    state.status_message = "That experiment does not exist."
    return datasets_page(state)


@route
def delete_dataset(state: State, dataset_index: int) -> Page:
    """Delete a selected dataset while repairing active and compare indices."""
    if not (0 <= dataset_index < len(state.datasets)):
        state.status_message = "Could not delete that dataset."
        return datasets_page(state)
    removed_name = state.datasets.pop(dataset_index).name
    if not state.datasets:
        state.active_index, state.compare_index = -1, -1
    else:
        if state.active_index == dataset_index:
            state.active_index = min(dataset_index, len(state.datasets) - 1)
        elif state.active_index > dataset_index:
            state.active_index -= 1
        if state.compare_index == dataset_index:
            state.compare_index = -1
        elif state.compare_index > dataset_index:
            state.compare_index -= 1
        if state.compare_index == state.active_index:
            state.compare_index = -1
    state.status_message = "Deleted " + removed_name + "."
    return datasets_page(state)


@route
def results_page(state: State) -> Page:
    """Present complete, unfiltered statistics for the active experiment."""
    dataset = current_dataset(state)
    if dataset is None:
        return no_experiment_page(state, "Experiment results")
    contents = page_header("Experiment results")
    contents += take_message(state)
    contents.append(card(Header(dataset.name, 3), dataset.description or "No description provided."))
    for line in display_results(dataset, state.velocity_units, state.stationary_cutoff):
        contents.append(card(line))
    contents += [Paragraph("Mean speed averages magnitudes; the mean U/V components may cancel."),
                 Paragraph("Statistics use all uploaded vectors, not the visualization speed filter."),
                 Row(Link("Visualize", "visualization_page"),
                     Link("Compare", "comparison_page"), Link("Export", "export_page"),
                     style_gap="16px")]
    return Page(state, contents)


@route
def visualization_page(state: State) -> Page:
    """Show a selectable plot and offer a matching downloadable PNG."""
    dataset = current_dataset(state)
    if dataset is None:
        return no_experiment_page(state, "Velocity visualizations")
    contents = page_header("Velocity visualizations", True)
    contents += take_message(state)
    contents += [Paragraph("Active experiment: " + dataset.name),
                 "Graph type:", SelectBox("plot_choice", list(PLOT_MODES), state.plot_mode),
                 Paragraph("Arrow spacing: plot every Nth vector (1 = all; try 16 or 32 for dense data)"),
                 TextBox("arrow_density_text", str(state.arrow_density)),
                 Paragraph("Arrow spacing affects vector plots only, not heatmaps or histograms."),
                 Button("Update graph", "set_visualization")]
    if state.plot_mode == "Speed histogram":
        contents.append(Paragraph("Histogram uses all measurements; the arrow filter does not apply."))
    else:
        contents.append(Paragraph("Only show vectors with speed >= " + str(state.minimum_speed) +
                                  " " + state.velocity_units + ". Every " + str(state.arrow_density) +
                                  "th vector is shown in arrow plots (when applicable)."))
        if state.plot_mode == "Velocity heatmap":
            contents.append(Paragraph("Heatmap colors estimate speeds between measured points. "
                                      "Missing measurement regions stay blank, while measured zero speed "
                                      "still appears as zero. If a contour cannot be made, measured "
                                      "points are shown instead."))
    if state.normalize_arrows and state.plot_mode in ("Velocity vectors", "Colored vectors"):
        contents.append(Paragraph("Arrows are normalized, so lengths do not show actual speeds."))
    make_single_plot(state, dataset)
    picture = plot_picture()
    contents.append(Image(picture, alt="Selected PIV velocity visualization"))
    contents.append(Download("Download this graph as PNG",
                             safe_filename(dataset.name) + "_" +
                             safe_filename(state.plot_mode) + ".png", picture,
                             content_type="image/png"))
    contents.append(Link("Return to results", "leave_visualization"))
    return Page(state, contents)


@route
def set_visualization(state: State, plot_choice: str, arrow_density_text: str) -> Page:
    """Save the selected plot and entered arrow spacing before redrawing."""
    if plot_choice not in PLOT_MODES:
        state.status_message = "Unknown graph type; keeping previous selection."
        return visualization_page(state)
    try:
        new_spacing = parse_arrow_spacing(arrow_density_text)
    except ValueError as error:
        state.status_message = str(error)
        return visualization_page(state)
    state.plot_mode = plot_choice
    state.arrow_density = new_spacing
    state.status_message = ("Plot updated to " + plot_choice + ". Showing every " +
                            str(new_spacing) + "th arrow in vector plots.")
    return visualization_page(state)


@route
def leave_visualization(state: State, plot_choice: str, arrow_density_text: str) -> Page:
    """Leave the graph page without saving unsubmitted form changes."""
    return results_page(state)


@route
def comparison_page(state: State) -> Page:
    """Show matched-axis plots, statistical differences, and an optional map."""
    if len(state.datasets) < 2:
        contents = page_header("Compare experiments") + take_message(state)
        contents += [card("At least two experiments are required."), Link("Upload a CSV", "upload_page")]
        return Page(state, contents)
    first_index = state.active_index if 0 <= state.active_index < len(state.datasets) else 0
    second_index = state.compare_index
    if not (0 <= second_index < len(state.datasets)) or second_index == first_index:
        second_index = 0 if first_index != 0 else 1
    first, second = state.datasets[first_index], state.datasets[second_index]
    options = []
    for index in range(len(state.datasets)):
        options.append(choice_for_index(state, index))
    first_mean, second_mean = mean_value(first.speeds), mean_value(second.speeds)
    difference = second_mean - first_mean
    percent = percentage_change(first_mean, second_mean)
    percent_text = (format(percent, "+.1f") + "%" if percent is not None
                    else "undefined (experiment A has zero mean speed)")
    if difference > 0:
        description = "B has a greater mean speed than A by " + format(difference, ".3f")
    elif difference < 0:
        description = "B has a lower mean speed than A by " + format(-difference, ".3f")
    else:
        description = "Both experiments have the same mean speed."
    contents = page_header("Compare experiments", True)
    contents += take_message(state)
    contents += [Paragraph("Experiment A"),
                 SelectBox("first_choice", options, choice_for_index(state, first_index)),
                 Paragraph("Experiment B"),
                 SelectBox("second_choice", options, choice_for_index(state, second_index)),
                 Button("Update comparison", "compare_datasets"),
                 card(Header("Summary", 3), description + " " + state.velocity_units + ".",
                      Paragraph("Mean speed A: " + format(first_mean, ".3f") +
                                " | B: " + format(second_mean, ".3f")),
                      Paragraph("Mean speed change (B relative to A): " + percent_text),
                      Paragraph("Max speed change (B - A): " +
                                format(max_speed(second.speeds) - max_speed(first.speeds), "+.3f")),
                      Paragraph("Speed standard deviation A: " +
                                format(speed_std_dev(first.speeds), ".3f") + " | B: " +
                                format(speed_std_dev(second.speeds), ".3f")),
                      Paragraph("Mean U A: " + format(mean_value(first.u), ".3f") +
                                " | B: " + format(mean_value(second.u), ".3f")),
                      Paragraph("Mean V A: " + format(mean_value(first.v), ".3f") +
                                " | B: " + format(mean_value(second.v), ".3f")))]
    make_comparison_plot(state, first, second)
    picture = plot_picture()
    contents += [Image(picture, alt="Side-by-side PIV comparison on common axes"),
                 Download("Download comparison PNG", "piv_comparison.png", picture,
                          content_type="image/png"),
                 Download("Download comparison CSV", "piv_comparison.csv",
                          comparison_csv(first, second), content_type="text/csv")]
    if make_difference_plot(first, second, state):
        difference_picture = plot_picture()
        contents += [Header("Spatial speed difference", 3),
                     Paragraph("At matching coordinates: positive means faster in B; negative means faster in A."),
                     Image(difference_picture, alt="Difference in measured speed at matching coordinates"),
                     Download("Download difference PNG", "piv_difference.png",
                              difference_picture, content_type="image/png")]
    else:
        contents.append(Paragraph("No difference map: the experiments do not share the same unique x-y positions."))
    contents.append(Link("Return to dashboard", "leave_comparison"))
    return Page(state, contents)


@route
def compare_datasets(state: State, first_choice: str, second_choice: str) -> Page:
    """Save two valid, distinct dataset indices and redraw their comparison."""
    first_index = index_from_choice(state, first_choice)
    second_index = index_from_choice(state, second_choice)
    if first_index == -1 or second_index == -1:
        state.status_message = "One selected experiment is no longer available."
    elif first_index == second_index:
        state.status_message = "Select two different experiments."
    else:
        state.active_index = first_index
        state.compare_index = second_index
        state.status_message = "Comparison selections updated."
    return comparison_page(state)


@route
def leave_comparison(state: State, first_choice: str, second_choice: str) -> Page:
    """Leave the comparison form without changing the chosen experiments."""
    return index(state)


@route
def settings_page(state: State) -> Page:
    """Display filter, color, density, grid, units and stationary-cutoff inputs."""
    contents = page_header("Visualization settings", True)
    contents += take_message(state)
    color_text = "" if state.color_max_speed == 0 else str(state.color_max_speed)
    contents += [card(
        Header("Velocity visualization", 3),
        Paragraph("Minimum plotted speed (nonnegative)"),
        TextBox("threshold_text", str(state.minimum_speed)),
        Paragraph("Arrow spacing: plot every Nth vector (positive whole number)"),
        TextBox("density_choice", str(state.arrow_density)),
        Paragraph("Colorbar maximum (leave blank to auto-scale)"),
        TextBox("color_max_text", color_text),
        Paragraph("Normalize arrow direction (lengths become equal for moving vectors)"),
        CheckBox("normalize_choice", state.normalize_arrows),
        Paragraph("Show gridlines"), CheckBox("grid_choice", state.show_grid)),
        card(Header("Labels and stationary vectors", 3),
             Paragraph("Position units (label only)"),
             TextBox("position_units_text", state.position_units),
             Paragraph("Velocity units (label only)"),
             TextBox("velocity_units_text", state.velocity_units),
             Paragraph("Nearly stationary cutoff (separate from the plot filter)"),
             TextBox("stationary_text", str(state.stationary_cutoff)),
             Paragraph("Measurements aren't converted when you change unit labels.")),
        Button("Save all settings", "save_settings"),
        Link("Cancel changes", "leave_settings")]
    return Page(state, contents)


@route
def save_settings(state: State, threshold_text: str, density_choice: str,
                  color_max_text: str, normalize_choice: bool, grid_choice: bool,
                  position_units_text: str, velocity_units_text: str,
                  stationary_text: str) -> Page:
    """Validate every field first; save only if the entire form is valid."""
    try:
        threshold = float(threshold_text)
        stationary = float(stationary_text)
        maximum = float(color_max_text) if color_max_text.strip() else 0.0
    except ValueError:
        state.status_message = "All numeric settings must be valid numbers."
        return settings_page(state)
    if not math.isfinite(threshold) or threshold < 0:
        state.status_message = "Minimum displayed speed must be finite and nonnegative."
        return settings_page(state)
    if not math.isfinite(stationary) or stationary < 0:
        state.status_message = "Stationary cutoff must be finite and nonnegative."
        return settings_page(state)
    if not math.isfinite(maximum) or (maximum <= 0 and color_max_text.strip()):
        state.status_message = "Color scale maximum must be positive, or leave it blank."
        return settings_page(state)
    try:
        spacing = parse_arrow_spacing(density_choice)
    except ValueError as error:
        state.status_message = str(error)
        return settings_page(state)
    if not position_units_text.strip() or not velocity_units_text.strip():
        state.status_message = "Please enter both unit labels."
        return settings_page(state)
    if len(position_units_text) > 18 or len(velocity_units_text) > 18:
        state.status_message = "Keep unit labels to 18 characters or fewer."
        return settings_page(state)
    state.minimum_speed = threshold
    state.arrow_density = spacing
    state.color_max_speed = maximum
    state.normalize_arrows = normalize_choice
    state.show_grid = grid_choice
    state.position_units = position_units_text.strip()
    state.velocity_units = velocity_units_text.strip()
    state.stationary_cutoff = stationary
    state.status_message = "Visualization settings saved."
    return settings_page(state)


@route
def leave_settings(state: State, threshold_text: str, density_choice: str,
                   color_max_text: str, normalize_choice: bool, grid_choice: bool,
                   position_units_text: str, velocity_units_text: str,
                   stationary_text: str) -> Page:
    """Discard any edits that have not been submitted with Save settings."""
    return index(state)


@route
def export_page(state: State) -> Page:
    """Offer summary, processed vectors and optional comparison CSV exports."""
    dataset = current_dataset(state)
    if dataset is None:
        return no_experiment_page(state, "Export results")
    prefix = safe_filename(dataset.name)
    contents = page_header("Export results") + take_message(state)
    contents += [card(Header("Summary statistics", 3),
                      Paragraph("Full-data statistics, including speed variation."),
                      Download("Download summary CSV", prefix + "_summary.csv",
                               summary_csv(dataset, state.stationary_cutoff), content_type="text/csv")),
                 card(Header("Processed vector data", 3),
                      Paragraph("Original measurements plus their calculated speed."),
                      Download("Download vectors CSV", prefix + "_vectors.csv",
                               processed_csv(dataset), content_type="text/csv")),
                 Paragraph("To download a PNG, visit Visualize or Compare."),
                 Link("Open visualizations", "visualization_page")]
    if 0 <= state.compare_index < len(state.datasets) and state.compare_index != state.active_index:
        contents.append(card(Download("Download current comparison CSV", "piv_comparison.csv",
                                      comparison_csv(dataset, state.datasets[state.compare_index]),
                                      content_type="text/csv")))
    return Page(state, contents)


if __name__ == "__main__":
    # Unlike the old basic_stats.py, no test plots or local files run on import.
    start_server(fresh_state())

