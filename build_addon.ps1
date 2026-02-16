param(
    [string]$Output = "loop_sculpt.zip"
)

if (Test-Path $Output) {
    Remove-Item $Output -Force
}

Compress-Archive -Path "loop_sculpt" -DestinationPath $Output
Write-Host "Created $Output"
