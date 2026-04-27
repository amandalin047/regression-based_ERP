# Deconvolution / Overlap-Aware Reanalysis So Far
I reanalyzed the word-position effect because the current dataset has very short SOA word presentation: each word appears every 360 ms. Under this timing, responses to adjacent words overlap heavily, so the conventional ERP pipeline may not work well. In particular, local baseline correction and fixed-window mean amplitude analysis may estimate a contaminated signal rather than a clean current-word N400 effect.

To address this, I built a <strong>time-lagged deconvolution-style regression model</strong> for Words 1–7. I excluded later/final words because some trials do not contain them, and final words are tied to the original experimental manipulation. The current model uses two predictors: an intercept/repeated-response predictor and a log-transformed word-position predictor. The log transform is used because the word-position effect appears nonlinear, with larger changes across early positions than later positions.

Because the time-lagged design matrix is highly overlapping and potentially ill-conditioned, I checked rank and condition number, then used <strong>Ridge regression</strong> instead of plain OLS.
I then compared two models:

1. intercept-only lagged model 
2. intercept + log-word-position lagged model  

Using <strong>nested cross-validation</strong>, the full model improved held-out reconstruction relative to the intercept-only model across all ROI channels (centro-parietal). This suggests that <i>log-transformed word position adds systematic explanatory/predictive structure beyond the repeated-response model</i>.

However, the coefficient-level interpretation is more cautious. When extracting the log-position beta in the N400 window, the expected positive direction was clearest at CZ and suggestive at CPZ, but not robust across all channels after correction. So the model-comparison result is stronger than the claim that every ROI channel shows a clean N400-window beta effect.

The reconstructed EEG from the fitted model looks plausible and shows repeated overlapping responses across Words 1–7. This supports the sanity of the deconvolution setup, but the reconstruction itself should not be overinterpreted as proof of a specific N400 effect.

Overall, the current conclusion is:
> <strong>The conventional baseline-corrected mean-amplitude ERP pipeline may be inappropriate for this short-SOA dataset because component overlap is severe.</strong> A time-lagged Ridge deconvolution model provides a more explicit and defensible way to test whether word-position structure is recoverable.

Next steps:
- compare preprocessing choices, especially deterministic-only preprocessing vs rule-based VEOG ICA;
- test whether the same pipeline generalizes to the two other lab datasets with longer 550 ms SOAs;
- compare the deconvolution results against the conventional baseline-corrected mean-amplitude pipeline.