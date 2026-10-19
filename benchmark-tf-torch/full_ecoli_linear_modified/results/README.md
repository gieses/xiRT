# full_ecoli_linear_modified: TensorFlow reference and PyTorch

Predictions were regenerated from the validation-selected checkpoints. Train, validation and test rows retain the original grouped split; only test metrics describe held-out accuracy. Errors are in seconds.

```text
           rmse_mean  rmse_seed_sd  mae_mean  r2_mean
model                                                
legacy_tf    160.768         3.178   109.537    0.982
pytorch      187.103        34.509   131.834    0.975
```

![Held-out RMSE](plots/rmse.png)

![Predictions](plots/predictions.png)

![Residuals](plots/residuals.png)

Error bars are seed standard deviations, not confidence intervals. TensorFlow is retired from the maintained package. Its checkpoint format, source and original environment locks are retained only as archival evidence.
