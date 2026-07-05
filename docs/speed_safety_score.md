# The Speed Safety Score in plain language

*A one-page explainer for non-technical readers. For the full methodology, code, and data sources, see the [project README](../README.md).*

## What it is

Every road segment gets a **Speed Safety Score from 0 to 100**. A higher score means the posted speed limit is more out of step with what is actually safe for the people using that road - and therefore a higher priority for review. The score is a weighted sum of three plain-language ingredients. 

| Weight | Ingredient | What it measures |
|---|---|---|
| **50%** | **Misalignment** | How many km/h the posted speed limit sits above the safe speed (`V_safe`) - the speed at which a crash is survivable for whoever is exposed on that road (a pedestrian, a motorcyclist, a driver in a head-on crash, etc.), based on Safe System crash-type logic. |
| **35%** | **VRU exposure** | Whether vulnerable road users - pedestrians, cyclists, schoolchildren - are actually present, based on street-image detections, school locations, and population density. The same km/h gap matters more where more people are exposed to it. |
| **15%** | **Data confidence** | How reliable the exposure and speed-limit reading are for this specific segment. Low-confidence segments are not allowed to outrank a well-evidenced one. |

The safe speed (`V_safe`) is calculated purely from road/crash-type and who is exposed - never from the posted limit or from how fast people actually drive. This is deliberate: the challenge's own methodology warning cautions against concluding "speed limit should be lower" just because *measured* speeds are high, since the road may simply be built for higher-speed travel. Observed speed is kept as a separate diagnostic (`operating_gap`) and never mixed into the score itself.

## Priority classes

Segments are then sorted into four classes, using each country's own top 3% / 10% / 20% of scores (Thailand and Maharashtra are graded on separate curves, so one country's larger gaps can't crowd the other's genuine problem segments off the list):

- **Top Priority** - act first.
- **Priority** - act next.
- **Watch** - monitor, no action yet.
- **No Issue** - posted limit is not above the safe speed, or the composite score is low.

Within the priority classes, a further **Review Track** flag splits segments into **Review Needed** (the recorded speed limit looks reliable - a genuine policy case) versus **Field Verification Needed** (the recorded limit itself looks like a data error and should be checked on the ground before acting on it).

## Where to see it

- **Interactive map**: https://iiokentaro.github.io/adb-ai-for-safer-roads-safer-speeds-challenge/ - click any segment for its score.
- **CSV/GeoJSON lists**: `outputs/priority_review_needed.csv`, `outputs/priority_field_check.csv`, `outputs/priority_urban.csv`, `outputs/priority_rural.csv`.
- **Full methodology and code**: [`src/safety_score.py`](../src/safety_score.py), documented in the main [README](../README.md#definition-of-the-speed-safety-score).
