# MT-10 baseline: end-weight 1.0; tweak epochs, hidden_sizes (architecture), and end_inner_fraction
# Run from project root.
$epochs = @(100, 300)
$hidden_sizes = @(
  @(64, 64),
  @(128, 128)
)
$end_inner_fractions = @(0.05, 0.01, .005)

foreach ($end_inner_fraction in $end_inner_fractions) {
  foreach ($ep in $epochs) {
    foreach ($hidden in $hidden_sizes) {
      $hiddenStr = $hidden -join ","
      Write-Host "`n========== epochs=$ep hidden=[$hiddenStr] end_inner_frac=$end_inner_fraction =========="
      python train.py --suite mt10 --end-weight 1.0 --epochs $ep --end-inner-fraction $end_inner_fraction --hidden @hidden
    }
  }
}
