# CLAM_HANCOCK

## Attention-based Multiple Instance Learning for HANCOCK

The code is based on the [CLAM](https://github.com/mahmoodlab/CLAM) framework for weakly-supervised classification on whole-slide (WSI) images and Tissue Microarrays (TMA). The Images from [HANCOCK](https://hancock.research.fau.eu/) dataset were used and preprocessed as described in the original CLAM documentation. Moreover multiple pre-trained models were investigated. 

WSI data can directly be used through the CLAM pipeline and TMA data have to first be dearrayed through qupath provided [code](https://github.com/ankilab/HANCOCK_MultimodalDataset) ...qupath_scripts/dearray_tma.groovy and then be constructed back using Additional/TMA_constructor.py code which makes new PNG images for each patiinet with different staining. Afterwards it could through the CLAM pipeline.

To use the code please follow the original documentation from CLAM. 

Additionally there are some other codes used for sanity check and comparison or etc. (Additional foldere): 
*  .../CLAM_UNI/AG2_heatmap_analysis.ipynb: include the inference part for the aggregated data
*  Aggregatio.ipynb: The part that combines the features extracted from the TMA and WSI 
*  checking_out_the_clam_outputs.ipynb: Attention analysis
*  Statistical_analysis.ipynb: plots and procedure used for Statistical part of the thesis which compares multiple distribution with t-test and u-test
*  TMA_constructor.py: is the part used to collect the TMAs with different staining and making the multi stained TMA data used in the thesis.
*  vizs.ipynb: includes the plots and comparision of the results extracted from the CLAM_UNI
    
## License

The code is released under the GPLv3 License following the original code base [here](https://github.com/mahmoodlab/CLAM).

