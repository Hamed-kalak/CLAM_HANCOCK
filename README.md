# CLAM_HANCOCK

## Attention-based Multiple Instance Learning for HANCOCK

The code is based on the [CLAM](https://github.com/mahmoodlab/CLAM) framework for weakly-supervised classification on whole-slide images. The images were preprocessed as described in the original documentation. Please follow the original documentation from CLAM.

Additionally there are some other codes used for sanity check and comparison or etc. (Additional foldere): 
*  .../CLAM_UNI/AG2_heatmap_analysis.ipynb: include the inference part for the aggregated data
*  Aggregatio.ipynb: The part that combines the features extracted from the TMA and WSI 
*  checking_out_the_clam_outputs.ipynb: Attention analysis
*  Statistical_analysis.ipynb: plots and procedure used for Statistical part of the thesis which compares multiple distribution with t-test and u-test
*  TMA_constructor.py: is the part used to collect the TMAs with different staining and making the multi stained TMA data used in the thesis.
*  vizs.ipynb: includes the plots and comparision of the results extracted from the CLAM_UNI
    
## License

The code is released under the GPLv3 License following the original code base [here](https://github.com/mahmoodlab/CLAM).

