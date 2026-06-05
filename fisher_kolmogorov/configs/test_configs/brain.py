from fisher_kolmogorov.configs.test_configs.base import BrainConfig
from fisher_kolmogorov.utilities.enum_utilities  import BrainSection

# Brain configuration factory ______________________________________________________________________________________________________________
def make_brain_config(
        section: BrainSection = BrainSection.SAGITTAL,
        config_kwargs: dict = None) -> BrainConfig:
    """
    Factory for brain simulation configs.

    Override any field via config_kwargs when
    running with different physical assumptions.

    Parameters
    ----------
    section       : BrainSection  (default: SAGITTAL)
    config_kwargs : overrides for any BrainConfig field
                    (alpha_grey, alpha_white, d_ext, d_axn, a)
    """
    base = dict(
        plane       = section,
        alpha_grey  = 0.5,
        alpha_white = 1.0,
        d_ext       = 8e-3,
        d_axn       = 8e-2,
        a           = (1.0, 1.0),
    )
    base.update(config_kwargs or {})
    return BrainConfig(**base)


