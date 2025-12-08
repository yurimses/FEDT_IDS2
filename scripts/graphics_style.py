"""Helper constants for keeping plot labels consistent across scripts."""

# Size used for axis labels, tick labels and titles across all plots.
LABEL_FONT_SIZE = 15
# Weight used for axis labels, tick labels and titles across all plots.
LABEL_FONT_WEIGHT = "light"
# Base multiplier (in inches) that controls how large each confusion-matrix subplot is.
CONFUSION_MATRIX_CELL_SIZE = 8
# Font size for annotations inside confusion-matrix cells.
CONFUSION_MATRIX_TEXT_SIZE = 13


def apply_default_plot_style(plt_module):
    """Apply consistent font sizes and weights to axes labels, ticks and titles."""
    plt_module.rcParams.update(
        {
            "axes.titlesize": LABEL_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.titleweight": LABEL_FONT_WEIGHT,
            "axes.labelweight": LABEL_FONT_WEIGHT,
            "xtick.labelsize": LABEL_FONT_SIZE,
            "ytick.labelsize": LABEL_FONT_SIZE,
            "font.weight": LABEL_FONT_WEIGHT,
            "legend.fontsize": LABEL_FONT_SIZE,
        }
    )
