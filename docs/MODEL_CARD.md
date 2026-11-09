# Model card: Breast cancer screening (mammo_cbm)

**Status:** research prototype. Fill the TBD fields from the notebook output after training. Do not publish numbers you have not measured.

## Intended use
Educational and research use: a second-reader aid that proposes lesion concepts and a risk estimate for a region the clinician selects. **Not** for diagnosis, screening or treatment decisions. Not cleared by any regulator.

## Out of scope
Whole-image screening, other modalities (ultrasound, MRI, tomosynthesis), non-adult populations, scanners and vendors not represented in the training data.

## Data
CBIS-DDSM (digitised film mammograms, biopsy-proven benign/malignant labels, ROI masks, radiologist concept annotations). Patient-level splits: train 70% / validation 15% / calibration 15% of the official training set; the official test set is used once. Concepts used: lesion type, mass shape, mass margins, calcification type, calcification distribution, breast density. BI-RADS assessment and subtlety are excluded to avoid label leakage.

## Model
EfficientNet-B0 (ImageNet-pretrained) with one softmax head per concept group; logistic regression on concept probabilities (L2, grouped CV); class-conditional conformal prediction (alpha = 0.10); Ledoit-Wolf Mahalanobis out-of-distribution flag.

## Evaluation (to fill)
| Item | Value |
|---|---|
| Test AUC (patient-cluster bootstrap 95% CI) | TBD |
| Calibration / Brier score | TBD |
| Concept accuracy per head | TBD |
| True-concept ceiling AUC | TBD |
| Black-box baseline AUC | TBD |
| Verdict breakdown and abstain rate | TBD |
| External validation | TBD (not done) |
| Subgroups (density, lesion type) | TBD (not done) |

## Known limitations and risks
- Single public dataset, scanned film, limited scanner diversity.
- The decision layer is transparent, but concept recognition is a neural network and can be wrong; always check the concepts.
- Conformal guarantees hold only if deployment data resemble the calibration data.
- Automation bias: a confident-looking score can anchor the reader.

## Ethics and governance
Any use with patient data requires ethics approval, data-protection compliance and, for clinical use, regulatory clearance. Images are processed in memory only by the dashboard and nothing is logged.
