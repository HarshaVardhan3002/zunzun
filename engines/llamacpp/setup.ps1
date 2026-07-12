# llama.cpp Strix Halo setup (Windows 11, Vulkan). Run in PowerShell.
#   .\setup.ps1            # fetch latest Vulkan x64 release into .\bin
#   .\setup.ps1 -Rocm      # print ROCm path instructions instead (WSL/Linux)
param([switch]$Rocm)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bin  = Join-Path $here "bin"

if ($Rocm) {
  @"
ROCm path (≈2x Vulkan when stable, Linux-solid / Windows newer):
  1. Install ROCm with gfx1151 support (ROCm 6.4+ lists Strix Halo).
  2. Force the arch:  set HSA_OVERRIDE_GFX_VERSION=11.5.1
  3. Build llama.cpp:  cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 && cmake --build build -j
  4. Raise GTT / 'variable graphics memory' in BIOS so the iGPU can map a 60-90 GB pool.
Vulkan (this script's default) needs none of that and is the reliable baseline.
"@ | Write-Host
  exit 0
}

New-Item -ItemType Directory -Force -Path $bin | Out-Null
Write-Host "Querying latest llama.cpp release..."
$rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
# Windows x64 Vulkan asset (name pattern: llama-*-bin-win-vulkan-x64.zip)
$asset = $rel.assets | Where-Object { $_.name -match "win-vulkan-x64\.zip$" } | Select-Object -First 1
if (-not $asset) { throw "no win-vulkan-x64 asset in $($rel.tag_name); check github.com/ggml-org/llama.cpp/releases" }
$zip = Join-Path $env:TEMP $asset.name
Write-Host "Downloading $($asset.name) ($($rel.tag_name))..."
Invoke-WebRequest $asset.browser_download_url -OutFile $zip
Expand-Archive -Force $zip $bin
Remove-Item $zip
Write-Host "Done. Binaries in $bin"
Write-Host "Smoke test:  $bin\llama-server.exe --version"
Write-Host "Then:        coli-route (see ..\..\router\README) or launch a preset from presets.yaml"
