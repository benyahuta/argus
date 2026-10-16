#!/usr/bin/env bash
# Download the public CMS Medicare files listed in scripts/data_urls.txt (Physician &
# Other Practitioners by Provider / by Provider and Service, Part D Prescribers by
# Provider) plus the OIG LEIE exclusion list. ~25 GB total; existing files are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
date -u +%Y-%m-%d > data/raw/LEIE_DOWNLOADED_ON
curl -fsSL -o data/raw/leie.csv https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv
while read -r name url; do
  [ -s "data/raw/$name.csv" ] || curl -fL --retry 5 -o "data/raw/$name.csv" "$url"
  echo "ok $name"
done < scripts/data_urls.txt
echo ALLDONE
