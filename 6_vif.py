from ml_tools.VIF import compute_vif_multi

from paths import PM
from helpers.constants import TARGETS


if __name__ == "__main__":
    compute_vif_multi(input_directory=PM.mice_datasets,
                      output_plot_directory=PM.vif,
                      ignore_columns=TARGETS)
