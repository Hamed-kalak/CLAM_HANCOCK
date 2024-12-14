Here’s a refined and clearer version of your README file:

CLAM_HANCOCK

Attention-Based Multiple Instance Learning for the HANCOCK Dataset

This repository builds upon the CLAM framework for weakly-supervised classification of Whole-Slide Images (WSIs) and Tissue Microarrays (TMAs). It uses images from the HANCOCK dataset, which have been preprocessed according to the original CLAM documentation. Multiple pre-trained models were also evaluated as part of this work.

Data Preparation
	•	Whole-Slide Images (WSIs): Can be directly processed through the CLAM pipeline.
	•	Tissue Microarrays (TMAs):
	1.	TMAs must first be dearrayed using QuPath scripts.
	2.	Then, the TMA_constructor.py script (found in the Additional folder) reconstructs new PNG images for each patient with different stainings.
	3.	Once prepared, the data can be processed through the CLAM pipeline.

Usage

Please refer to the original CLAM documentation for pipeline setup and usage instructions.

Additional Scripts

This repository contains several additional scripts and notebooks for data preparation, analysis, and sanity checks, located in the Additional folder:
	•	CLAM_UNI/AG2_heatmap_analysis.ipynb: Contains inference code for aggregated data.
	•	Aggregation.ipynb: Combines features extracted from TMAs and WSIs.
	•	checking_out_the_clam_outputs.ipynb: Analyzes attention outputs from the CLAM model.
	•	Statistical_analysis.ipynb: Performs statistical comparisons (e.g., t-tests, U-tests) and generates related plots for the thesis.
	•	TMA_constructor.py: Assembles multi-stained TMA data used in the thesis.
	•	vizs.ipynb: Visualizes and compares results extracted from the CLAM_UNI.

License

This code is released under the GPLv3 License, in accordance with the original CLAM repository.

Let me know if you’d like any additional edits!
