#!/bin/bash

# Script to check if a model in the storage directory was run with --text

if [ -z "$1" ]; then
  echo "Usage: $0 <directory_name>"
  exit 1
fi

DIR_NAME=$1

# List of env_ids that are run with --text
TEXT_ENVS=(
    "BabyAI-GoToRedBallGrey-v0"
    "BabyAI-GoToRedBall-v0"
    "BabyAI-GoToRedBallNoDists-v0"
    "BabyAI-GoToObj-v0"
    "BabyAI-GoToObjS4-v0"
    "BabyAI-GoToObjS6-v1"
    "BabyAI-GoToLocal-v0"
    "BabyAI-GoToLocalS5N2-v0"
    "BabyAI-GoToLocalS6N2-v0"
    "BabyAI-GoToLocalS6N3-v0"
    "BabyAI-GoToLocalS6N4-v0"
    "BabyAI-GoToLocalS7N4-v0"
    "BabyAI-GoToLocalS7N5-v0"
    "BabyAI-GoToLocalS8N2-v0"
    "BabyAI-GoToLocalS8N3-v0"
    "BabyAI-GoToLocalS8N4-v0"
    "BabyAI-GoToLocalS8N5-v0"
    "BabyAI-GoToLocalS8N6-v0"
    "BabyAI-GoToLocalS8N7-v0"
    "BabyAI-GoTo-v0"
    "BabyAI-GoToOpen-v0"
    "BabyAI-GoToObjMaze-v0"
    "BabyAI-GoToObjMazeOpen-v0"
    "BabyAI-GoToObjMazeS4R2-v0"
    "BabyAI-GoToObjMazeS4-v0"
    "BabyAI-GoToObjMazeS5-v0"
    "BabyAI-GoToObjMazeS6-v0"
    "BabyAI-GoToObjMazeS7-v0"
    "BabyAI-GoToImpUnlock-v0"
    "BabyAI-GoToSeq-v0"
    "BabyAI-GoToSeqS5R2-v0"
    "BabyAI-GoToRedBlueBall-v0"
    "BabyAI-GoToDoor-v0"
    "BabyAI-GoToObjDoor-v0"
    "BabyAI-Open-v0"
    "BabyAI-OpenRedDoor-v0"
    "BabyAI-OpenDoor-v0"
    "BabyAI-OpenDoorDebug-v0"
    "BabyAI-OpenDoorColor-v0"
    "BabyAI-OpenDoorLoc-v0"
    "BabyAI-OpenTwoDoors-v0"
    "BabyAI-OpenRedBlueDoors-v0"
    "BabyAI-OpenRedBlueDoorsDebug-v0"
    "BabyAI-OpenDoorsOrderN2-v0"
    "BabyAI-OpenDoorsOrderN4-v0"
    "BabyAI-OpenDoorsOrderN2Debug-v0"
    "BabyAI-OpenDoorsOrderN4Debug-v0"
    "BabyAI-Pickup-v0"
    "BabyAI-PickupLoc-v0"
    "BabyAI-PickupDist-v0"
    "BabyAI-PickupDistDebug-v0"
    "BabyAI-PickupAbove-v0"
    "BabyAI-UnblockPickup-v0"
    "BabyAI-PutNextLocal-v0"
    "BabyAI-PutNextLocalS5N3-v0"
    "BabyAI-PutNextLocalS6N4-v0"
    "BabyAI-PutNextS4N1-v0"
    "BabyAI-PutNextS5N1-v0"
    "BabyAI-PutNextS5N2-v0"
    "BabyAI-PutNextS5N2Carrying-v0"
    "BabyAI-PutNextS6N3-v0"
    "BabyAI-PutNextS6N3Carrying-v0"
    "BabyAI-PutNextS7N4-v0"
    "BabyAI-PutNextS7N4Carrying-v0"
    "BabyAI-Unlock-v0"
    "BabyAI-UnlockLocal-v0"
    "BabyAI-UnlockLocalDist-v0"
    "BabyAI-KeyInBox-v0"
    "BabyAI-UnlockPickup-v0"
    "BabyAI-UnlockPickupDist-v0"
    "BabyAI-UnlockToUnlock-v0"
    "BabyAI-BlockedUnlockPickup-v0"
    "BabyAI-ActionObjDoor-v0"
    "BabyAI-KeyCorridor-v0"
    "BabyAI-KeyCorridorS3R1-v0"
    "BabyAI-KeyCorridorS3R2-v0"
    "BabyAI-KeyCorridorS3R3-v0"
    "BabyAI-KeyCorridorS4R3-v0"
    "BabyAI-KeyCorridorS5R3-v0"
    "BabyAI-KeyCorridorS6R3-v0"
    "BabyAI-FindObjS5-v0"
    "BabyAI-FindObjS6-v0"
    "BabyAI-FindObjS7-v0"
    "BabyAI-MoveTwoAcrossS5N2-v0"
    "BabyAI-MoveTwoAcrossS8N9-v0"
    "BabyAI-OneRoomS8-v0"
    "BabyAI-OneRoomS12-v0"
    "BabyAI-OneRoomS16-v0"
    "BabyAI-OneRoomS20-v0"
    "BabyAI-Synth-v0"
    "BabyAI-SynthS5R2-v0"
    "BabyAI-SynthLoc-v0"
    "BabyAI-SynthSeq-v0"
    "BabyAI-MiniBossLevel-v0"
    "BabyAI-BossLevel-v0"
    "BabyAI-BossLevelNoUnlock-v0"
)

# Remove tae "catalog_" prefix from the directory name
NO_PREFIX=${DIR_NAME#catalog_}

# Replace underscores with hyphens and convert to original case
# This is a best guess, as case information is lost.
RECONSTRUCTED_ENV=$(echo "$NO_PREFIX" | sed 's/_/-/g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)} 1')

# A more robust way is to generate all possible model names and check for a match
  declare -A mymap
for env_id in "${TEXT_ENVS[@]}"; do
  safe_env=${env_id//-/_}
  safe_env=${safe_env,,}
  model_name="catalog_${safe_env}"

  if [ "$model_name" == "$DIR_NAME" ]; then
    avg=$(tail -n 5 "$DIR_NAME/log.csv" | awk -F'|' '{sum += $5; n++} END {if (n>0) print sum/n; else print 0}')
    mymap["$DIR_NAME"]=$avg
    echo "Match found for --text model:"
    echo "Directory: $DIR_NAME"
    echo "Original env_id: $env_id"
    exit 0
  fi
done

echo "No match found for any --text models."
exit 1
