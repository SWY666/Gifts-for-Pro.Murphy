################################################################
Dataset Information
################################################################

Dataset Name:
Breast Cancer Profiling Project, Drug Sensitivity 1: Fixed-cell GR measures of 35 breast cell lines to 34 small molecule perturbagens. Dataset 1 of 2: Normalized growth rate inhibition values.

Dataset Description:
We measured the sensitivities of two non-malignant breast cell lines and 33 breast cancer cell lines of which twenty were triple negative, six were hormone receptor positive, four were Her2 amplified, and three were established from triple negative patient-derived xenografts to 34 clinically-relevant small molecule perturbagens. A microscopy-based dose response assay was used to measure drug potency, and to quantify drug efficacy in terms of growth inhibition (GR metrics) and cell death. We treated cells with single drugs over a 9-point ½ log dilution series from a maximum dose not exceeding 10 µM and then measured cell number and viability after three days of drug exposure.

--Data in Package:
20343.csv

--Metadata in Package:
20343_Experimental_Metadata.txt
Small_Molecule_Metadata.txt
Cell_Line_Metadata.txt

################################################################
Center-specific Information
################################################################

Center-specific Name:
HMS_LINCS

Center-specific Dataset ID:
20343

Center-specific Dataset Link:
http://lincs.hms.harvard.edu/db/datasets/20343/

################################################################
Assay Information
################################################################

Assay Protocol:
1. Cells in mid-log phase of the growth cycle from 35 breast cancer cell lines were plated at appropriate densities to achieve ~40% confluence at the time of treatment; 500-2000 cells/well in 60 µL of their recommended growth medium in five 384-well plates.  The growth conditions used are detailed in this file download: <a href="http://lincs.hms.harvard.edu/data/HMS_Dataset_20343-20344_GrowthConditions.csv">Growth Conditions</a>.<br />                                                                                                                                                                                                                                                                      
2. The plated cells were grown for 36 hours at 37°C in the presence of 5% CO<sub>2</sub> (all MDAMB lines except MDAMB231 were grown in the absence of CO<sub>2</sub>) and were then treated with the indicated small molecules by pin transfer or using a D300 Digital Dispenser (Hewlett-Packard, Palo Alto, CA).<br />                                                                                                                                                                                                                                                                                
3. Cells were stained and fixed for analysis at the time of drug delivery (one plate) and after 72 hours of incubation (four plates) by adding 15 µL of staining solution (1:1000 LIVE/DEAD Far Red Dead Cell Stain (Thermo Fisher Scientific, catalog #L-34974), 2 µg/ml Hoechst 33342 (Thermo Fisher Scientific, catalog #62249), 10% OptiPrep (Sigma-Aldrich, catalog #D1556-250ML) in PBS) for 30 min at room temperature followed by fixing solution (4% formaldehyde (v/v) (Sigma-Aldrich, catalogue #F1635-500ML), 20% OptiPrep in PBS) for 20 min at room temperature. After fixation, 80 µL of supernatant per well was removed and replaced with 80 µL of PBS with an EL406 Washer Dispenser (BioTek, Winooski, VT).<br />                                                                                                                                                                                                                                                                                                                                                                                           4. The plates were scanned with a PE Operetta high-throughput plate scanner. Six fields of view covering the full well were acquired with a 10x high NA objective for all wells. The excitation and emission filters used for image acquisition were 360-400 nm and 410-480 nm for Hoechst, and 620-640 nm and 650-700 nm for LDR.<br />                                                                                                                                
5. Live and dead cell counts were obtained using Columbus software (Nuclear segmentation: module: Find Nuclei; method: C; default parameters except Individual Threshold which was set to 0.25; Channel: Hoechst. Corpse segmentation: module: Find Nuclei; method: B; default parameters except Area which was set to >80 for BT20, >100 for HCC1806, and >50 for all other cell lines; Channel: LDR; filter: Hoechst intensity < 300; output: Corpse- Number of objects. Segmented nuclei were classified as dead based on LDR texture: module: Calculate Texture Properties; method: SER Features; scale: 8px; normalized by: Region Intensity SER Spot; channel: LDR; filter: >0.001; output: DeadLDR- Number of objects, or based on size: module: Calculate Morphology Properties; method: Standard Area; population: Nuclei segmented; filter: <60-120 depending on the cell line; output: DeadSize- Number of objects. Live cells were counted as the number of nuclei segmented that did not meet the size or texture criteria for dead cells; output: Live-Number of objects.) The fraction of dead cells was calculated by taking the sum of (Corpse, DeadLDR, DeadSize counts) divided by the sum of (Corpse, DeadLDR, DeadSize, Live counts).<br />                                                                                                                                                                                                                                                             
6. Consistent with the methods reported in Hafner et al. (2016) (PMID: <a href="http://www.ncbi.nlm.nih.gov/pubmed/27135972" target = "_blank">27135972</a>), the <b>Mean Normalized Growth Rate Inhibition (GR) Values</b> were calculated according to the following formula: 2^[log2(x(c)/x0)/log2(xctrl/x0)]-1 where x(c) is the mean of the measured Live cell counts after a given treatment, x0 is the mean of the Live cell counts from the day 0 untreated plate grown in parallel until the time of treatment, and xctrl is the mean of the Live cell counts of the DMSO-treated control wells for all technical replicates.   The GR metrics calculated in HMS-LINCS dataset <a href = "http://lincs.hms.harvard.edu/db/datasets/20344/">#20344</a> are based on the GR values calculated in this step from this dataset (#20343).<br />   
7. The <b>Increased Fraction Dead</b> was calculated by subtracting the mean fraction of dead cells in the DMSO-treated control wells from the mean fraction of dead cells in the wells from a given treatment across all technical replicates. The Death_AUC calculated in HMS-LINCS dataset <a href = "http://lincs.hms.harvard.edu/db/datasets/20344/">#20344</a> are based on the Increased Fraction Dead values calculated in this step from this dataset (#20343). 

Date Updated:
2018-03-30

Date Retrieved from Center:
5/21/2018

################################################################
Metadata Information
################################################################

Metadata information regarding the entities used in the experiments is included in the accompanied metadata. A metadata file per entity category is included in the package. For example, the metadata for all the cell lines that were used in the dataset are included in the Cell_Lines_Metadata.txt file.
Descriptions for each metadata field can be found here: http://www.lincsproject.org/data/data-standards/
[/generic/datapointFile]
[/generic/experimental_metadata]
[/generic/reagents_studied]
