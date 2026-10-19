# Native recurrent fine-tuning

All strategies start from the same train-only checkpoint for each cohort and seed. The learning rate and adapter settings are fixed before evaluation. Early stopping and learning-rate reductions use validation loss; test rows are scored only after checkpoint selection and reconstruction. This is a separate protocol from the historical hyperparameter sweep.

```text
                                             rmse_mean  rmse_seed_sd  mae_mean  r2_mean  trainable_parameters  training_seconds_mean
dataset                    model                                                                                                    
full_ecoli_linear_modified recurrent_frozen    236.638         1.626   170.547    0.961                   289                 23.601
                           recurrent_full      183.936        28.757   128.097    0.976                  1729                106.810
                           recurrent_lokr      235.539         1.427   169.610    0.961                   441                 57.783
                           recurrent_lora      234.475         1.294   168.099    0.961                   769                 76.603
full_ecoli_modified        recurrent_frozen    291.185        11.697   203.845    0.884                   289                 10.850
                           recurrent_full      258.362         1.230   171.832    0.909                  1729                 53.068
                           recurrent_lokr      289.999        10.657   202.709    0.885                   441                 36.227
                           recurrent_lora      288.312        11.178   201.098    0.886                   769                 31.519
full_fa_cross_modified     recurrent_frozen    487.688         8.497   407.284    0.587                   289                  1.212
                           recurrent_full      287.869        10.279   227.256    0.856                  1729                  5.671
                           recurrent_lokr      486.136         7.110   407.210    0.590                   441                  4.683
                           recurrent_lora      469.467        12.992   393.069    0.618                   769                  5.794
full_fa_linear_modified    recurrent_frozen    645.911         5.507   488.767    0.741                   289                  2.091
                           recurrent_full      599.327        24.515   457.406    0.777                  1793                  9.653
                           recurrent_lokr      644.557         4.995   487.980    0.742                   441                  5.901
                           recurrent_lora      649.718         1.877   489.869    0.738                   769                  6.829
full_plink_modified        recurrent_frozen    343.421         1.436   225.860    0.850                   289                 10.785
                           recurrent_full      322.437         1.602   205.448    0.868                  1833                 48.390
                           recurrent_lokr      341.787         1.515   225.241    0.852                   441                 32.012
                           recurrent_lora      340.577         2.341   223.613    0.853                   769                 31.013
```

RMSE and MAE are seconds. Seed SD is not a confidence interval. Repeated cohorts are correlated views of two studies. These are staged within-cohort training experiments, not external transfer-learning validation.
