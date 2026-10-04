# Downloads the raw football-data.co.uk CSVs used by every other script in
# this project. Run this FIRST, before process_data.py. Fetches every
# division in football-data.co.uk's "Main Leagues" section (the one with a
# per-season CSV per division AND bookmaker odds columns -- the site's
# separate "Extra Leagues" section, e.g. Argentina/Brazil/Japan/USA/etc.,
# uses a different one-file-per-country format and generally has no odds
# columns, so it's not usable for this project's expert-mixture approach and
# is deliberately excluded here) x 10 seasons (2016/17-2025/26) into
# data/raw/, named "<LeagueCode>_<Season>.csv", and writes a per-file
# download log to data/download_log.csv.
#
# Pipeline position: download_data.ps1 -> process_data.py -> everything else.

$ErrorActionPreference = "Stop"   # abort a single download attempt loudly instead of silently continuing

$seasons = @("1617","1718","1819","1920","2021","2122","2223","2324","2425","2526")
$leagues = @{
    "E0"  = "England_PremierLeague"
    "E1"  = "England_Championship"
    "E2"  = "England_League1"
    "E3"  = "England_League2"
    "EC"  = "England_Conference"
    "SC0" = "Scotland_Premiership"
    "SC1" = "Scotland_Championship"
    "SC2" = "Scotland_League1"
    "SC3" = "Scotland_League2"
    "D1"  = "Germany_Bundesliga"
    "D2"  = "Germany_2Bundesliga"
    "I1"  = "Italy_SerieA"
    "I2"  = "Italy_SerieB"
    "SP1" = "Spain_LaLiga"
    "SP2" = "Spain_SegundaDivision"
    "F1"  = "France_Ligue1"
    "F2"  = "France_Ligue2"
    "N1"  = "Netherlands_Eredivisie"
    "B1"  = "Belgium_JupilerLeague"
    "P1"  = "Portugal_LigaI"
    "T1"  = "Turkey_SuperLig"
    "G1"  = "Greece_SuperLeague"
}

$root = Split-Path -Parent $PSScriptRoot   # project root = parent of scripts/
$outDir = Join-Path $root "data\raw"
New-Item -ItemType Directory -Force $outDir | Out-Null
$results = @()

foreach ($season in $seasons) {
    foreach ($code in $leagues.Keys) {
        $name = $leagues[$code]
        $outFile = Join-Path $outDir "$($code)_$($season).csv"
        if (Test-Path $outFile) {
            # already downloaded in an earlier run (e.g. the original 10 leagues x 5
            # seasons) -- skip re-fetching it, keep the existing file as-is
            $results += [PSCustomObject]@{ League=$code; Name=$name; Season=$season; Status="SKIPPED (exists)"; Bytes=(Get-Item $outFile).Length }
            continue
        }
        $url = "https://www.football-data.co.uk/mmz4281/$season/$code.csv"
        try {
            Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing -TimeoutSec 30
            $size = (Get-Item $outFile).Length
            $results += [PSCustomObject]@{ League=$code; Name=$name; Season=$season; Status="OK"; Bytes=$size }
        } catch {
            $results += [PSCustomObject]@{ League=$code; Name=$name; Season=$season; Status="FAILED: $($_.Exception.Message)"; Bytes=0 }
        }
        Start-Sleep -Milliseconds 300
    }
}

$results | Format-Table -AutoSize
$results | Export-Csv -Path (Join-Path $root "data\download_log.csv") -NoTypeInformation
