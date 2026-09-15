# BrainWideBench - Sample Data - NeurIPS 2026 E&D Track
 
This sample provides reviewers with a representative subset of the full BrainWideBench dataset for the purpose of inspecting data quality and format. The full dataset is available at the [primary URL](https://brain-wide-bench.s3.amazonaws.com/index.html) listed in the submission.
 
## Contents
 
The sample consists of four files, two each from pretraining/evaluation splits and `all_units`/`selected_units` variants of the dataset:
 
- **Pretraining (`all_units`)**: the first file (lexicographic order) from the pretraining bucket:
  [`004d8fd5-41e7-4f1b-a45b-0d4ad76fe446_all.h5`](https://brain-wide-bench.s3.amazonaws.com/brainsets/all_units/ibl_brain_wide_bench_2026/pretrain/004d8fd5-41e7-4f1b-a45b-0d4ad76fe446.h5)
- **Evaluation (`all_units`)**: the first file (lexicographic order) from the evaluation bucket:
  [`0802ced5-33a3-405e-8336-b65ebc5cb07c_all.h5`](https://brain-wide-bench.s3.amazonaws.com/brainsets/all_units/ibl_brain_wide_bench_2026/eval/0802ced5-33a3-405e-8336-b65ebc5cb07c.h5)
- **Pretraining (`selected_units`)**: the first file (lexicographic order) from the pretraining bucket:
  [`004d8fd5-41e7-4f1b-a45b-0d4ad76fe446_selected.h5`](https://brain-wide-bench.s3.amazonaws.com/brainsets/selected_units/ibl_brain_wide_bench_2026/pretrain/004d8fd5-41e7-4f1b-a45b-0d4ad76fe446.h5)
- **Evaluation (`selected_units`)**: the first file (lexicographic order) from the evaluation bucket:
  [`0802ced5-33a3-405e-8336-b65ebc5cb07c_selected.h5`](https://brain-wide-bench.s3.amazonaws.com/brainsets/selected_units/ibl_brain_wide_bench_2026/eval/0802ced5-33a3-405e-8336-b65ebc5cb07c.h5)
All files are hosted in the same bucket and in the same format as the full dataset.

 
Files are stored in HDF5 format after processing via [`torch_brain`](https://torch-brain.readthedocs.io/).
