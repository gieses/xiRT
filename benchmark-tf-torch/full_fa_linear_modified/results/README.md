# full_fa_linear_modified: TensorFlow reference and PyTorch

Predictions were regenerated from the validation-selected checkpoints. Train, validation and test rows retain the original grouped split; only test metrics describe held-out accuracy. Errors are in seconds.

```text
           rmse_mean  rmse_seed_sd  mae_mean  r2_mean
model                                                
legacy_tf    613.059        42.656   462.510    0.766
pytorch      630.039        25.397   469.189    0.753
```

![Held-out RMSE](plots/rmse.png)

![Predictions](plots/predictions.png)

![Residuals](plots/residuals.png)

Error bars are seed standard deviations, not confidence intervals. TensorFlow is retired from the maintained package. Its checkpoint format, source and original environment locks are retained only as archival evidence.
