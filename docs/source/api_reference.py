from importlib import import_module

import jinja2

"""
CONFIGURING API_REFERENCE
=========================

API_REFERENCE maps each module name to the modules's __api_ref__. Each module's
__api_ref__ consists of the following components:

description (required, `None` if not needed)
    The additional description for the module to be placed under the module
    docstring, before the sections start.
sections (required)
    A list of sections, each of which consists of:
    - title (required, `None` if not needed): the section title, commonly it should
      not be `None` except for the first section of a module,
    - description (optional): the optional additional description for the section,
    - autosummary (required): an autosummary block, assuming current module is the
      current module name.

Essentially, the rendered page would look like the following:

|---------------------------------------------------------------------------------|
|     {{ module_name }}                                                           |
|     =================                                                           |
|     {{ module_docstring }}                                                      |
|     {{ description }}                                                           |
|                                                                                 |
|     {{ section_title_1 }}   <-------------- Optional if one wants the first     |
|     ---------------------                   section to directly follow          |
|     {{ section_description_1 }}             without a second-level heading.     |
|     {{ section_autosummary_1 }}                                                 |
|                                                                                 |
|     {{ section_title_2 }}                                                       |
|     ---------------------                                                       |
|     {{ section_description_2 }}                                                 |
|     {{ section_autosummary_2 }}                                                 |
|                                                                                 |
|     More sections...                                                            |
|---------------------------------------------------------------------------------|

Hooks will be automatically generated for each module and each section. For a module,
e.g., `torch_brain.data importset`, the hook would be `dataset_ref`; for a
section, e.g., "Mixins" under `torch_brain.data importset`, the hook would be
`dataset_ref-mixins`. However, note that a better way is to refer using the :mod: directive,
e.g., :mod:`torch_brain.data importset` for the module. Only in case that a section
is not a particular submodule does the hook become useful.
"""


# The API reference, grouped as the landing page presents it. Each group is a card;
# each module inside it is a row on that card, described by its own docstring.
API_GROUPS = [
    {
        "title": "Evaluation contract",
        "description": "What a submission must contain, and how it is scored.",
        "modules": [
            "ibl_bwb_eval",
            "ibl_bwb_eval.metrics",
            "ibl_bwb_eval.predictions",
            "ibl_bwb_eval.entity_ids",
        ],
    },
    {
        "title": "Core",
        "description": "The pieces every task suite builds on.",
        "modules": [
            "core.data",
            "core.dataset",
            "core.finetuning",
            "core.model",
            "core.trainer",
            "core.samplers",
            "core.transforms",
            "core.nn.metrics",
        ],
    },
    {
        "title": "Pretraining",
        "description": "Encoders pretrained once, then evaluated by each suite.",
        "modules": ["pretrain.models", "pretrain.datasets"],
    },
    {
        "title": "TS1: Behavior Prediction",
        "description": "Decoding behavior from neural population activity.",
        "modules": ["ts1", "ts1.models.single_session", "ts1.models.pretrained"],
    },
    {
        "title": "TS2: Neural Activity Prediction",
        "description": "Predicting activity across time and across neurons.",
        "modules": ["ts2", "ts2.models.single_session", "ts2.models.pretrained"],
    },
    {
        "title": "TS3: Brain Region Prediction",
        "description": "Predicting the region a single neuron was recorded in.",
        "modules": [
            "ts3",
            "ts3.models.inductive",
            "ts3.models.transductive",
            "ts3.models.supervised",
            "ts3.probes",
        ],
    },
]

# Modules to include in API reference, in landing-page order.
API_MODS = [module for group in API_GROUPS for module in group["modules"]]

API_REFERENCE = {m: import_module(m).__api_ref__ for m in API_MODS}


def _short_summary(module: str) -> str:
    """First line of the module docstring, or an empty string if it has none."""
    doc = import_module(module).__doc__ or ""
    return doc.strip().split("\n", 1)[0]


# API_GROUPS with each module paired with its short summary, for the landing page.
API_CARDS = [
    {
        **group,
        "entries": [{"module": m, "summary": _short_summary(m)} for m in group["modules"]],
    }
    for group in API_GROUPS
]


def build_api_rst():
    import pathlib

    generated = pathlib.Path(__file__).parent / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "api").mkdir(exist_ok=True)

    # rst_templates
    # kwargs: args to pass to jinja
    rst_templates: list[dict] = [
        {
            "template_path": "api/index.rst.template",
            "target_path": "generated/api/index.rst",
            "kwargs": {"API_CARDS": API_CARDS},
        },
        {
            "template_path": "api/all.rst.template",
            "target_path": "generated/api/all.rst",
            "kwargs": {"API_REFERENCE": API_REFERENCE.items()},
        },
    ]

    for module in API_REFERENCE:
        rst_templates.append(
            {
                "template_path": "api/module.rst.template",
                "target_path": f"generated/api/{module}.rst",
                "kwargs": {"module": module, "module_info": API_REFERENCE[module]},
            }
        )

    for template in rst_templates:
        # Read the corresponding template file into jinja2
        with open(template["template_path"]) as f:
            t = jinja2.Template(f.read())

        # Render the template and write to the target
        with open(template["target_path"], "w") as f:
            f.write(t.render(**template["kwargs"]))
